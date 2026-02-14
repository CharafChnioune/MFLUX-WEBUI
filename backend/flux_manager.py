import os
import gc
import time
import random
import traceback
import mlx.core as mx
from PIL import Image
from backend.mflux_compat import ModelConfig
try:
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
except ModuleNotFoundError:  # pragma: no cover - optional model support
    Flux2Klein = None
    Flux2KleinEdit = None
from backend.lora_manager import process_lora_files, download_lora
from backend.ollama_manager import enhance_prompt
from backend.prompts_manager import enhance_prompt_with_mlx
from backend.mlx_utils import force_mlx_cleanup, print_memory_usage
from PIL import PngImagePlugin
from backend.generation_workflow import (
    get_generation_workflow,
    check_pre_generation,
    process_dynamic_prompt,
    get_next_seed,
    monitor_step_progress,
    save_enhanced_metadata,
    update_generation_stats
)
import json
from pathlib import Path
import numpy as np
import re
from typing import Union, Tuple, Optional
from backend.model_manager import (
    get_custom_model_config,
    resolve_local_path,
    normalize_base_model_choice,
    resolve_mflux_model_config,
    strip_quant_suffix,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_flux2_model_name(model_name: str) -> bool:
    normalized = strip_quant_suffix(model_name or "").lower()
    return normalized.startswith(("flux2-", "klein-")) or normalized == "flux2-klein"

def is_flux2_base_model_name(model_name: str) -> bool:
    normalized = strip_quant_suffix(model_name or "").lower()
    return normalized.startswith(("flux2-", "klein-")) and "-base-" in normalized


def is_flux2_dev_model_name(model_name: str) -> bool:
    normalized = strip_quant_suffix(model_name or "").lower()
    return normalized == "flux2-dev"


def save_image_with_metadata(pil_image, output_path: str, metadata: dict):
    """
    Save a PIL image with PNG textual metadata (prompt, seed, etc.).
    Falls back to a normal save if metadata embedding fails.
    """
    try:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            if value is None:
                continue
            pnginfo.add_text(str(key), str(value))
        pil_image.save(output_path, pnginfo=pnginfo)
    except Exception as e:
        print(f"Warning: Failed to embed PNG metadata for {output_path}: {e}")
        pil_image.save(output_path)

def parse_image_format(image_format):
    """
    Parse image format string to get width and height.
    Example: "Portrait (576x1024)" -> (576, 1024)
    """
    try:
        dimensions = image_format.split('(')[1].split(')')[0]
        width, height = map(int, dimensions.split('x'))
        return width, height
    except Exception as e:
        print(f"Error parsing image format: {str(e)}")
        return 512, 512

def parse_scale_factor(dimension_value: str, original_dimension: int = None) -> int:
    """
    Parse scale factor or absolute dimension value.
    Examples:
    - "2x" -> 2 * original_dimension
    - "1.5x" -> 1.5 * original_dimension  
    - "1024" -> 1024
    - "auto" -> original_dimension
    """
    if dimension_value is None or dimension_value == "":
        return 512  # Default
    
    dimension_value = str(dimension_value).strip().lower()
    
    if dimension_value == "auto":
        return original_dimension if original_dimension else 512
    
    # Check for scale factor (e.g., "2x", "1.5x")
    scale_match = re.match(r'^([0-9]*\.?[0-9]+)x$', dimension_value)
    if scale_match:
        scale = float(scale_match.group(1))
        if original_dimension:
            result = int(scale * original_dimension)
            # Align to 16-pixel boundaries for optimal results
            return ((result + 15) // 16) * 16
        else:
            return 512  # Fallback if no original dimension
    
    # Try to parse as absolute value
    try:
        return int(float(dimension_value))
    except ValueError:
        return 512  # Fallback

def calculate_dimensions_with_scale(
    width_input: Union[str, int], 
    height_input: Union[str, int], 
    original_image: Optional[Image.Image] = None
) -> Tuple[int, int]:
    """
    Calculate final dimensions supporting scale factors and mixed types.
    """
    original_width = original_image.width if original_image else None
    original_height = original_image.height if original_image else None
    
    width = parse_scale_factor(width_input, original_width)
    height = parse_scale_factor(height_input, original_height)
    
    # Safety check for very large dimensions
    MAX_DIMENSION = 2048
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        print(f"Warning: Requested dimensions ({width}x{height}) exceed recommended limit ({MAX_DIMENSION}x{MAX_DIMENSION})")
        print("This may cause memory issues or slow generation.")
    
    return width, height 

def get_or_create_flux(model, config=None, image=None, lora_paths=None, lora_scales=None, is_controlnet=False, low_ram=False, base_model_override: Optional[str] = None):
    """
    Create or retrieve a Flux model instance.
    """
    try:
        base_model = strip_quant_suffix(model)
        base_model_override = normalize_base_model_choice(base_model_override)
        model_path: Optional[object] = resolve_local_path(base_model)

        # Pre-quantized MLX repos use aliases like `flux2-klein-4b-mlx-4bit`.
        # For these, the mflux loader needs a real HF repo id (org/model), not the alias.
        config_name = base_model
        if isinstance(base_model, str) and "-mlx-" in base_model:
            config_name = base_model.split("-mlx-")[0]
            try:
                cfg = get_custom_model_config(base_model)
                if getattr(cfg, "model_name", None) and "/" in str(cfg.model_name):
                    model_path = str(cfg.model_name)
            except Exception:
                pass

        if is_flux2_dev_model_name(config_name):
            from backend.flux2_dev import _default_flux2_dev_model_config
            model_config = _default_flux2_dev_model_config()
        else:
            model_config = resolve_mflux_model_config(config_name, base_model_override)

        if "-8-bit" in model:
            quantize = 8
        elif "-4-bit" in model:
            quantize = 4
        elif "-6-bit" in model:
            quantize = 6
        elif "-3-bit" in model:
            quantize = 3
        else:
            quantize = None

        if is_flux2_model_name(base_model):
            if is_controlnet:
                print("Flux2 does not support ControlNet.")
                return None

            if is_flux2_dev_model_name(config_name):
                if image is not None:
                    print("Flux2-dev edit is not yet supported.")
                    return None
                try:
                    from backend.flux2_dev import Flux2Dev
                except Exception as exc:
                    print(f"Flux2-dev loader is unavailable: {exc}")
                    return None

                if not lora_paths:
                    lora_scales = None
                elif lora_scales is None:
                    lora_scales = []
                elif isinstance(lora_scales, tuple):
                    lora_scales = list(lora_scales)

                print(
                    f"Creating Flux2Dev with model_config={model_config}, "
                    f"quantize={quantize}, local_path={model_path}, lora_paths={lora_paths}, "
                    f"lora_scales={lora_scales}"
                )
                flux = Flux2Dev(
                    model_config=model_config,
                    quantize=quantize,
                    model_path=str(model_path) if model_path else None,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales,
                )
                return flux

            if Flux2Klein is None or Flux2KleinEdit is None:
                print("Flux2 classes are unavailable. Ensure mflux>=0.15.0 is installed.")
                return None

            FluxClass = Flux2KleinEdit if image else Flux2Klein

            if not lora_paths:
                lora_scales = None
            elif lora_scales is None:
                lora_scales = []
            elif isinstance(lora_scales, tuple):
                lora_scales = list(lora_scales)

            print(
                f"Creating {FluxClass.__name__} with model_config={model_config}, "
                f"quantize={quantize}, local_path={model_path}, lora_paths={lora_paths}, "
                f"lora_scales={lora_scales}"
            )
            try:
                flux = FluxClass(
                    model_config=model_config,
                    quantize=quantize,
                    model_path=str(model_path) if model_path else None,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales,
                )
            except TypeError:
                flux = FluxClass(
                    model_config=model_config,
                    quantize=quantize,
                    local_path=str(model_path) if model_path else None,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales,
                )
            return flux

        # Only Flux2 models are supported
        print(f"Error: Model {base_model} is not a supported Flux2 model.")
        return None
        
        # Opmerking: 'image' en 'low_ram' worden niet direct doorgegeven aan de Flux constructor
        # maar kunnen later in de workflow gebruikt worden

    except Exception as e:
        print(f"Error creating Flux instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_random_seed():
    """
    Generate a random seed for image generation.
    """
    return str(random.randint(0, 2**32 - 1))

def generate_image_batch(flux, prompt, seed, steps, height, width, guidance, num_images, model_name: Optional[str] = None):
    """
    Generate a batch of images using the Flux model.
    """
    images = []
    filenames = []
    seeds_used = []
    
    flux2_model = is_flux2_model_name(model_name or "") or flux.__class__.__name__.lower().startswith("flux2")
    if flux2_model:
        # Distilled FLUX.2 Klein models use guidance=1.0; base models allow guidance > 1.0.
        is_base = is_flux2_base_model_name(model_name or "")
        if not is_base:
            cfg = getattr(flux, "model_config", None)
            cfg_name = getattr(cfg, "model_name", "") if cfg is not None else ""
            is_base = "-base-" in str(cfg_name).lower()

        is_dev = is_flux2_dev_model_name(model_name or "") or "flux.2-dev" in str(cfg_name).lower()

        if not is_base and not is_dev and guidance != 1.0:
            print("FLUX.2 (distilled) uses guidance=1.0; overriding guidance.")
            guidance = 1.0

    for i in range(num_images):
        current_seed = seed if seed is not None else int(time.time()) + i
        seeds_used.append(current_seed)
        output_filename = f"generated_{int(time.time())}_{i}_seed_{current_seed}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        generated = flux.generate_image(
            seed=current_seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        meta = {
            "prompt": prompt,
            "seed": current_seed,
            "steps": steps,
            "guidance": guidance,
            "width": width,
            "height": height,
            "model": getattr(flux, "model_config", None) or "",
        }
        save_image_with_metadata(generated.image, output_path, meta)
        images.append(generated.image)  # Gebruik .image attribuut voor PIL Image
        filenames.append(output_filename)
    
    return images, filenames, seeds_used

def clear_flux_cache():
    """
    Clear the Flux cache to free up memory.
    """
    global flux_cache
    flux_cache = {}
    gc.collect()

def force_mlx_cleanup():
    mx.eval(mx.zeros(1))
    
    # mlx.core deprecated mx.metal.* in favor of top-level helpers.
    clear_cache = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if clear_cache:
        clear_cache()
    
    reset_peak = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if reset_peak:
        reset_peak()

    gc.collect()

def print_memory_usage(label):
    """
    Print the current memory usage.
    """
    try:
        # mlx.core deprecated mx.metal.get_*_memory in favor of mx.get_*_memory.
        get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        active_memory = get_active() / 1e6
        peak_memory = get_peak() / 1e6
        print(f"{label} - Active memory: {active_memory:.2f} MB, Peak memory: {peak_memory:.2f} MB")
    except Exception as e:
        print(f"Error getting memory usage: {str(e)}")

def simple_generate_image(
    prompt, model, image_format, lora_files,
    ollama_model=None, system_prompt=None,
    *lora_scales, num_images=1, low_ram=False
):
    """
    Simple interface for generating images.
    Uses the same dynamic prompts and seed workflow as the advanced flows.
    """
    try:
        print(f"\n--- Generating image ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Image format: {image_format}")
        print(f"LoRA files: {lora_files}")
        print(f"LoRA scales: {lora_scales}")
        print(f"Number of Images: {num_images}")
        print(f"Low-RAM mode: {low_ram}")
        print_memory_usage("Before generation")

        # Apply dynamic prompts via the shared workflow
        try:
            processed_prompt = process_dynamic_prompt(prompt)
            if processed_prompt != prompt:
                print(f"Dynamic prompt processed (simple): {processed_prompt}")
            prompt = processed_prompt
        except Exception as e:
            print(f"Warning: Dynamic prompt processing failed in simple_generate_image: {e}")
        
        # Create a Flux instance
        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_files,
            lora_scales=lora_scales if lora_scales else None,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Error: Could not create Flux instance", prompt
        
        # Parse image format
        width, height = parse_image_format(image_format)
        
        # Enhance prompt if requested
        if ollama_model and system_prompt:
            prompt = enhance_prompt(prompt, ollama_model, system_prompt)
        
        images = []
        filenames = []
        
        try:
            for i in range(int(num_images)):
                # Use unified workflow seed management (Auto Seeds + fallback)
                try:
                    seed = get_next_seed(None)
                except Exception as e:
                    print(f"Warning: get_next_seed failed in simple_generate_image, using time-based seed: {e}")
                    seed = int(time.time()) + i

                output_filename = f"generated_{int(time.time())}_{i}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                try:
                    print(f"Generating image {i+1} of {num_images} with seed {seed}...")
                    
                    is_flux2 = is_flux2_model_name(model)
                    if is_flux2:
                        guidance_value = 1.0
                        steps_value = 4
                    else:
                        # Guidance waarde instellen op basis van model type
                        # Voor "dev" model gebruiken we 4.0, voor "schnell" 0.0
                        # We zetten het nooit op None
                        is_dev_model = "dev" in model
                        guidance_value = 4.0 if is_dev_model else 0.0
                        steps_value = 4 if "schnell" in model else 20

                    generated = flux.generate_image(
                        seed=seed,
                        prompt=prompt,
                        num_inference_steps=steps_value,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                    )
                    
                    pil_image = generated.image
                    meta = {
                        "prompt": prompt,
                        "seed": seed,
                        "model": model,
                        "width": width,
                        "height": height,
                        "steps": steps_value,
                        "guidance": guidance_value,
                    }
                    save_image_with_metadata(pil_image, output_path, meta)
                    images.append(pil_image)
                    filenames.append(output_filename)
                    print(f"Saved image to {output_path}")
                except Exception as e:
                    print(f"Error generating image {i+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            return images, "\n".join(filenames), prompt
        
        except Exception as e:
            print(f"Error in generation loop: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], "Error generating images", prompt
            
    except Exception as e:
        print(f"Error in simple_generate_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
    
    finally:
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

def generate_image_gradio(
    prompt, model, base_model, seed, width, height, steps, guidance,
    lora_files, metadata, ollama_model=None, system_prompt=None,
    prompt_file=None, config_from_metadata=None, stepwise_output_dir=None,
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False, auto_seeds=None,
    progress_callback=None
):
    """
    Generate images using the Flux model through the Gradio interface with v0.9.0 features.
    """
    workflow = get_generation_workflow()
    
    try:
        print(f"\n--- Generating image (Gradio v0.9.0) ---")
        print(f"Model: {model}")
        print(f"Base Model: {base_model}")
        print(f"Original Prompt: {prompt}")
        print(f"Dimensions: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Guidance: {guidance}")
        print(f"Seed: {seed}")
        print(f"Auto Seeds: {auto_seeds}")
        print(f"LoRA files: {lora_files}")
        print(f"Number of Images: {num_images}")
        print(f"Low-RAM mode: {low_ram}")
        print_memory_usage("Before generation")

        # 1. Pre-generation checks (config validation, etc.)
        pre_checks = check_pre_generation()
        if not pre_checks["can_proceed"]:
            error_msg = "\n".join(pre_checks["errors"])
            print(f"Pre-generation check failed: {error_msg}")
            return [], error_msg, prompt
        
        if pre_checks["warnings"]:
            warnings_msg = "\n".join(pre_checks["warnings"])
            print(f"Pre-generation warnings: {warnings_msg}")

        # 2. Load prompt from file if specified (--prompt-file support)
        if prompt_file and prompt_file.strip():
            try:
                from backend.dynamic_prompts_manager import load_prompt_from_file
                file_prompt = load_prompt_from_file(prompt_file.strip())
                if file_prompt:
                    print(f"Loaded prompt from file '{prompt_file}': {file_prompt}")
                    prompt = file_prompt
                else:
                    print(f"Warning: Could not load prompt from file '{prompt_file}', using original prompt")
            except Exception as e:
                print(f"Error loading prompt file '{prompt_file}': {str(e)}")
                print("Using original prompt instead")

        # 3. Load config from metadata if specified (--config-from-metadata support)
        if config_from_metadata and config_from_metadata.strip():
            try:
                from backend.metadata_config_manager import load_config_from_image_metadata, apply_metadata_config
                loaded_config = load_config_from_image_metadata(config_from_metadata.strip())
                if loaded_config:
                    print(f"Loaded config from metadata '{config_from_metadata}': {loaded_config}")
                    
                    # Apply loaded config to current parameters
                    current_params = {
                        'prompt': prompt, 'model': model, 'seed': seed, 'width': width, 'height': height,
                        'steps': steps, 'guidance': guidance, 'lora_files': lora_files, 'low_ram': low_ram
                    }
                    updated_params = apply_metadata_config(loaded_config, current_params)
                    
                    # Update parameters with loaded values where appropriate
                    if 'prompt' in updated_params and updated_params['prompt'] != prompt:
                        prompt = updated_params['prompt']
                        print(f"Updated prompt from metadata: {prompt}")
                    if 'seed' in updated_params and updated_params['seed'] != seed:
                        seed = updated_params['seed']
                        print(f"Updated seed from metadata: {seed}")
                    if 'steps' in updated_params and updated_params['steps'] != steps:
                        steps = updated_params['steps']
                        print(f"Updated steps from metadata: {steps}")
                    if 'guidance' in updated_params and updated_params['guidance'] != guidance:
                        guidance = updated_params['guidance']
                        print(f"Updated guidance from metadata: {guidance}")
                    if 'width' in updated_params and updated_params['width'] != width:
                        width = updated_params['width']
                        print(f"Updated width from metadata: {width}")
                    if 'height' in updated_params and updated_params['height'] != height:
                        height = updated_params['height']
                        print(f"Updated height from metadata: {height}")
                else:
                    print(f"Warning: Could not load config from metadata '{config_from_metadata}'")
            except Exception as e:
                print(f"Error loading config from metadata '{config_from_metadata}': {str(e)}")
                print("Using original parameters instead")

        # 4. Setup stepwise output if specified (--stepwise-image-output-dir support)
        stepwise_enabled = False
        if stepwise_output_dir and stepwise_output_dir.strip():
            try:
                from backend.stepwise_output_manager import setup_stepwise_output
                session_name = f"generation_{int(time.time())}"
                stepwise_enabled = setup_stepwise_output(stepwise_output_dir.strip(), session_name)
                if stepwise_enabled:
                    print(f"Stepwise output enabled: {stepwise_output_dir.strip()}/{session_name}")
                else:
                    print(f"Warning: Could not setup stepwise output directory '{stepwise_output_dir}'")
            except Exception as e:
                print(f"Error setting up stepwise output '{stepwise_output_dir}': {str(e)}")
                stepwise_enabled = False

        # 5. Setup VAE tiling if specified (--vae-tiling & --vae-tiling-split support)
        vae_tiling_enabled = False
        try:
            from backend.vae_tiling_manager import setup_vae_tiling, should_use_vae_tiling
            auto_tiling = should_use_vae_tiling(width, height)
        except Exception as e:
            print(f"Error importing VAE tiling manager: {str(e)}")
            auto_tiling = False
            
        if vae_tiling or auto_tiling:
            try:
                tile_size = 512  # Default tile size
                overlap = 64     # Default overlap
                split_factor = max(1, int(vae_tiling_split)) if vae_tiling_split else 1
                
                vae_tiling_enabled = setup_vae_tiling(True, tile_size, overlap, split_factor)
                if vae_tiling_enabled:
                    print(f"VAE tiling enabled: tile_size={tile_size}, split_factor={split_factor}")
                    # Enable low RAM mode for large images with tiling
                    if width * height > 1024 * 1024:
                        low_ram = True
                        print("Enabled low RAM mode for large image with VAE tiling")
                else:
                    print("Warning: Could not setup VAE tiling")
            except Exception as e:
                print(f"Error setting up VAE tiling: {str(e)}")
                vae_tiling_enabled = False

        # 6. Process dynamic prompts
        processed_prompt = process_dynamic_prompt(prompt)
        if processed_prompt != prompt:
            print(f"Dynamic prompt processed: {processed_prompt}")
        prompt = processed_prompt

        # 7. Process LoRA files and scales (docs: gradiodocs/docs-image/image.md for file inputs)
        lora_paths = process_lora_files(lora_files)
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        base_model_override = normalize_base_model_choice(base_model)
        
        # 8. Enhance prompt with Ollama if requested
        if ollama_model and system_prompt:
            prompt = enhance_prompt(prompt, ollama_model, system_prompt)
            print(f"Ollama enhanced prompt: {prompt}")
            
        # 9. Determine seeds to use (auto-seeds integration)
        seeds = []

        # Treat auto_seeds as a simple on/off flag:
        # when enabled we always use automatic seeds (ignoring explicit seed),
        # otherwise we respect an explicit seed if provided.
        try:
            auto_seeds_enabled = bool(auto_seeds) and int(auto_seeds) > 0
        except (TypeError, ValueError):
            auto_seeds_enabled = False

        # Normalize number of images
        try:
            total_images = int(num_images) if num_images else 1
        except (TypeError, ValueError):
            total_images = 1
        if total_images < 1:
            total_images = 1

        # Normalize seed to an optional int
        seed_int: Optional[int] = None
        if seed not in (None, "", "None"):
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                print(f"Warning: Invalid seed value '{seed}', ignoring explicit seed")
                seed_int = None

        for _ in range(total_images):
            try:
                if auto_seeds_enabled:
                    # Force automatic seed selection (Auto Seeds manager or random)
                    seeds.append(get_next_seed(None))
                else:
                    # Use provided seed if any, otherwise let workflow decide
                    seeds.append(get_next_seed(seed_int))
            except Exception as e:
                print(f"Warning: get_next_seed failed in generate_image_gradio, using random seed: {e}")
                seeds.append(random.randint(0, 2**32 - 1))
        
        all_images = []
        all_seeds = []
        all_metadata = []
        generation_start_time = time.time()
        
        for i, current_seed in enumerate(seeds):
            try:
                # Check if we should stop/pause before each generation
                progress_status = monitor_step_progress(i, len(seeds))
                if not progress_status["should_continue"]:
                    print(f"Generation stopped: {progress_status.get('stop_reason', 'Unknown')}")
                    break
                
                if progress_status["should_pause"]:
                    print(f"Generation paused: {progress_status.get('pause_reason', 'Unknown')}")
                    workflow.handle_generation_pause()
                    continue

                # Initialize Flux model
                if progress_callback:
                    progress_callback("stage", "loading_model")
                flux = get_or_create_flux(
                    model=model,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales_float,
                    low_ram=low_ram,
                    base_model_override=base_model
                )
                
                if not flux:
                    print("Failed to initialize Flux model")
                    update_generation_stats(success=False)
                    continue
                    
                # Prepare generation parameters
                if guidance is None:
                    is_dev_model = "dev" in model
                    guidance_value = 4.0 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)

                if is_flux2_model_name(model):
                    if not is_flux2_base_model_name(model):
                        if guidance_value != 1.0:
                            print("FLUX.2 (distilled) uses guidance=1.0; overriding guidance.")
                        guidance_value = 1.0
                
                steps_int = 4 if not steps or steps.strip() == "" else int(steps)
                
                # Generate the image with progress monitoring
                if progress_callback:
                    progress_callback("image_start", {
                        "current_image": i + 1,
                        "total_images": len(seeds),
                        "seed": current_seed,
                    })
                print(f"Generating image {i+1}/{len(seeds)} with seed: {current_seed}")
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    num_inference_steps=steps_int,
                    height=height,
                    width=width,
                    guidance=guidance_value,
                )
                
                # Process the result
                pil_image = generated.image
                timestamp = int(time.time())
                filename = f"generated_{timestamp}_{current_seed}.png"
                
                # Prepare enhanced metadata
                generation_metadata = {
                    "prompt": prompt,
                    "original_prompt": prompt if processed_prompt == prompt else processed_prompt,
                    "seed": current_seed,
                    "steps": steps_int,
                    "guidance": guidance_value,
                    "width": width,
                    "height": height,
                    "model": model,
                    "generation_time": str(time.ctime()),
                    "generation_duration": time.time() - generation_start_time,
                    "lora_files": lora_paths,
                    "lora_scales": lora_scales_float,
                    "low_ram_mode": low_ram,
                    "filename": filename
                }
                
                # Save the image with embedded metadata
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                output_path = os.path.join(OUTPUT_DIR, filename)
                save_image_with_metadata(pil_image, output_path, generation_metadata)
                
                # Save enhanced metadata using workflow
                if metadata:
                    save_enhanced_metadata(Path(output_path), generation_metadata)
                
                all_images.append(pil_image)
                all_seeds.append(current_seed)
                all_metadata.append(generation_metadata)
                
                # Update statistics
                update_generation_stats(success=True)
                
                if progress_callback:
                    progress_callback("image_complete", {
                        "current_image": i + 1,
                        "total_images": len(seeds),
                    })
                print(f"Successfully generated image {i+1} with seed {current_seed}")

            except Exception as e:
                if progress_callback:
                    progress_callback("image_error", {
                        "current_image": i + 1,
                        "error": str(e),
                    })
                print(f"Error generating image {i+1} with seed {current_seed}: {str(e)}")
                import traceback
                traceback.print_exc()
                update_generation_stats(success=False)
            
            finally:
                # Cleanup after each generation
                if 'flux' in locals():
                    del flux
                gc.collect()
                force_mlx_cleanup()
        
        # Final cleanup and results
        workflow.reset_workflow_state()
        
        if all_images:
            # Prepare result information
            seed_info = []
            for i, (seed, meta) in enumerate(zip(all_seeds, all_metadata)):
                duration = meta.get('generation_duration', 0)
                seed_info.append(f"Image {i+1}: Seed {seed} ({duration:.1f}s)")
            
            result_info = "\n".join(seed_info)
            if pre_checks["warnings"]:
                result_info += "\n\nWarnings:\n" + "\n".join(pre_checks["warnings"])
            
            print(f"Generation completed: {len(all_images)} images generated")
            return all_images, result_info, prompt
        else:
            return [], "No images were generated successfully", prompt
            
    except Exception as e:
        print(f"Error in generate_image_gradio: {str(e)}")
        import traceback
        traceback.print_exc()
        update_generation_stats(success=False)
        return [], f"Generation error: {str(e)}", prompt
    
    finally:
        # Final cleanup
        gc.collect()
        force_mlx_cleanup()
        print_memory_usage("After generation")

def generate_image_controlnet_gradio(
    prompt, control_image, model, base_model, seed, height, width, steps, guidance,
    controlnet_strength, lora_files, metadata, save_canny,
    prompt_file=None, config_from_metadata=None, stepwise_output_dir=None,
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False,
    progress_callback=None
):
    """
    Generate an image with controlnet guidance.
    """
    try:
        print(f"\n--- Generating image with ControlNet ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Controlnet strength: {controlnet_strength}")
        print(f"Seed: {seed}")
        print(f"Dimensions: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Guidance: {guidance}")
        print(f"LoRA files: {lora_files}")
        print(f"LoRA scales: {lora_scales}")
        print(f"Save canny: {save_canny}")
        print(f"Low-RAM mode: {low_ram}")
        print_memory_usage("Before controlnet generation")

        if is_flux2_model_name(model):
            return [], "FLUX.2 does not support ControlNet.", prompt

        # Load prompt from file if specified (CLI-style --prompt-file)
        if prompt_file and str(prompt_file).strip():
            try:
                from backend.dynamic_prompts_manager import load_prompt_from_file
                file_prompt = load_prompt_from_file(str(prompt_file).strip())
                if file_prompt:
                    print(f"Loaded prompt from file '{prompt_file}': {file_prompt}")
                    prompt = file_prompt
                else:
                    print(f"Warning: Could not load prompt from file '{prompt_file}', using original prompt")
            except Exception as e:
                print(f"Error loading prompt file '{prompt_file}' in ControlNet: {str(e)}")
                print("Using original prompt instead")

        # Apply dynamic prompt processing so ControlNet matches main workflow
        try:
            processed_prompt = process_dynamic_prompt(prompt)
            if processed_prompt != prompt:
                print(f"Dynamic prompt processed (ControlNet): {processed_prompt}")
            prompt = processed_prompt
        except Exception as e:
            print(f"Warning: Dynamic prompt processing failed in ControlNet: {e}")
        
        # Generate temporary file path for control image
        control_image_path = os.path.join(OUTPUT_DIR, f"controlnet_input_{int(time.time())}.png")
        control_image.save(control_image_path)
        
        # Process LoRA selections and normalize base model override
        lora_paths = process_lora_files(lora_files)
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        base_model_override = normalize_base_model_choice(base_model)
        
        # Determine seed via shared workflow (Auto Seeds + fallback)
        seed_int: Optional[int] = None
        if seed not in (None, "", "None"):
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                print(f"Warning: Invalid seed value '{seed}' in ControlNet, falling back to auto-seed")
                seed_int = None

        try:
            seed = get_next_seed(seed_int)
        except Exception as e:
            print(f"Warning: get_next_seed failed in ControlNet, using random seed: {e}")
            seed = int(time.time())
        
        # Initialize Flux1Controlnet
        if progress_callback:
            progress_callback("stage", "loading_model")
        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
            is_controlnet=True,
            low_ram=low_ram,
            base_model_override=base_model_override
        )
        
        if not flux:
            return [], "Could not initialize ControlNet", prompt
            
        try:
            if progress_callback:
                progress_callback("image_start", {
                    "current_image": 1,
                    "total_images": 1,
                    "seed": seed,
                })
            print(f"Generating controlnet image with seed: {seed}")
            current_seed = seed

            # Guidance waarde: we zorgen ervoor dat het nooit None is
            if guidance is None:
                is_dev_model = "dev" in model
                guidance_value = 4.0 if is_dev_model else 0.0
            else:
                guidance_value = float(guidance)

            is_flux2 = is_flux2_model_name(model)
            if is_flux2:
                guidance_value = 1.0

            steps_int = 4 if not steps or steps.strip() == "" else int(steps)
            controlnet_strength_float = float(controlnet_strength)

            # Generate the image
            generated = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                controlnet_image_path=control_image_path,
                num_inference_steps=steps_int,
                height=height,
                width=width,
                guidance=guidance_value,
                controlnet_strength=controlnet_strength_float,
            )
            
            # Process results
            pil_image = generated.image
            timestamp = int(time.time())
            filename = f"controlnet_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            meta = {
                "prompt": prompt,
                "seed": current_seed,
                "steps": steps_int,
                "guidance": guidance_value,
                "width": width,
                "height": height,
                "model": model,
                "controlnet_strength": controlnet_strength_float,
                "lora_files": lora_paths,
                "lora_scales": lora_scales_float,
            }
            save_image_with_metadata(pil_image, output_path, meta)
            
            # Save canny reference if requested
            if save_canny:
                # Maak de canny edge detectie afbeelding met ControlnetUtil
                try:
                    from mflux.models.flux.variants.controlnet.controlnet_util import ControlnetUtil
                except ModuleNotFoundError:
                    from mflux.controlnet.controlnet_util import ControlnetUtil  # type: ignore
                from PIL import Image
                import numpy as np
                
                # Laad de originele afbeelding
                orig_control_image = Image.open(control_image_path)
                
                # Pas de afbeelding aan de afmetingen aan
                # Dit is dezelfde logica die in ControlnetUtil._scale_image wordt gebruikt
                aspect_ratio = orig_control_image.width / orig_control_image.height
                new_width = width
                new_height = height
                
                # Resize terwijl de aspect ratio behouden blijft
                if aspect_ratio > 1:
                    new_height = int(height / aspect_ratio)
                else:
                    new_width = int(width * aspect_ratio)
                
                # Rond af naar een veelvoud van 16
                new_width = 16 * (new_width // 16)
                new_height = 16 * (new_height // 16)
                
                # Resize de afbeelding
                resized_image = orig_control_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Maak de canny afbeelding
                canny_image = ControlnetUtil._preprocess_canny(resized_image)
                
                # Converteer numpy array naar PIL Image (in geval het een numpy array is)
                if isinstance(canny_image, np.ndarray):
                    # Zorg ervoor dat het array waarden tussen 0-255 heeft (wat nodig is voor PIL)
                    if canny_image.max() > 0:  # vermijd delen door nul
                        # Normaliseer naar 0-255 bereik als het nog niet in dat bereik zit
                        if canny_image.max() != 255 or canny_image.min() != 0:
                            canny_image = ((canny_image - canny_image.min()) / (canny_image.max() - canny_image.min()) * 255).astype(np.uint8)
                    # Converteer naar RGB beeld als het een enkel kanaal is
                    if len(canny_image.shape) == 2:  # grijswaarden afbeelding
                        canny_image = np.stack([canny_image, canny_image, canny_image], axis=-1)
                    # Converteer naar PIL afbeelding
                    canny_image = Image.fromarray(canny_image)
                
                # Sla de canny afbeelding op
                canny_path = os.path.join(OUTPUT_DIR, f"canny_{timestamp}_{current_seed}.png")
                canny_image.save(canny_path)
                print(f"Canny edge detection image saved to {canny_path}")
                
            # Save metadata if requested
            if metadata:
                metadata_path = os.path.join(OUTPUT_DIR, f"controlnet_{timestamp}_{current_seed}.json")
                metadata_dict = {
                    "prompt": prompt,
                    "seed": current_seed,
                    "steps": steps_int,
                    "guidance": guidance_value,
                    "width": width,
                    "height": height,
                    "model": model,
                    "controlnet_strength": controlnet_strength_float,
                    "generation_time": str(time.ctime()),
                    "lora_files": lora_paths,
                    "lora_scales": lora_scales_float
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                    
            if progress_callback:
                progress_callback("image_complete", {
                    "current_image": 1,
                    "total_images": 1,
                })
            print(f"Generated controlnet image saved to {output_path}")
            return [pil_image], filename, prompt

        except Exception as e:
            if progress_callback:
                progress_callback("image_error", {"current_image": 1, "error": str(e)})
            print(f"Error in controlnet image generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], f"Error: {str(e)}", prompt
            
    except Exception as e:
        print(f"Error in controlnet preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

def generate_image_i2i_gradio(
    prompt, input_image, model, base_model, seed, height, width, steps, guidance,
    image_strength, lora_files, metadata,
    prompt_file=None, config_from_metadata=None, stepwise_output_dir=None,
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False,
    progress_callback=None
):
    """
    Generate an image based on an input image.
    """
    try:
        print(f"\n--- Generating image-to-image ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Image strength: {image_strength}")
        print(f"Seed: {seed}")
        print(f"Dimensions: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Guidance: {guidance}")
        print(f"LoRA files: {lora_files}")
        print(f"LoRA scales: {lora_scales}")
        print(f"Low-RAM mode: {low_ram}")
        print_memory_usage("Before image-to-image generation")

        # Load prompt from file if specified
        if prompt_file and str(prompt_file).strip():
            try:
                from backend.dynamic_prompts_manager import load_prompt_from_file
                file_prompt = load_prompt_from_file(str(prompt_file).strip())
                if file_prompt:
                    print(f"Loaded prompt from file '{prompt_file}': {file_prompt}")
                    prompt = file_prompt
                else:
                    print(f"Warning: Could not load prompt from file '{prompt_file}', using original prompt")
            except Exception as e:
                print(f"Error loading prompt file '{prompt_file}' in image-to-image: {str(e)}")
                print("Using original prompt instead")

        # Apply dynamic prompt processing for image-to-image
        try:
            processed_prompt = process_dynamic_prompt(prompt)
            if processed_prompt != prompt:
                print(f"Dynamic prompt processed (image-to-image): {processed_prompt}")
            prompt = processed_prompt
        except Exception as e:
            print(f"Warning: Dynamic prompt processing failed in image-to-image: {e}")

        # Generate temporary file path for input image
        input_image_path = os.path.join(OUTPUT_DIR, f"i2i_input_{int(time.time())}.png")
        input_image.save(input_image_path)
        
        # Process LoRA files and normalize base model override
        lora_paths = process_lora_files(lora_files)
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        base_model_override = normalize_base_model_choice(base_model)
        
        # Determine seed via shared workflow (Auto Seeds + fallback)
        seed_int: Optional[int] = None
        if seed not in (None, "", "None"):
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                print(f"Warning: Invalid seed value '{seed}' in image-to-image, falling back to auto-seed")
                seed_int = None

        try:
            seed = get_next_seed(seed_int)
        except Exception as e:
            print(f"Warning: get_next_seed failed in image-to-image, using random seed: {e}")
            seed = int(time.time())
        
        # Initialize Flux
        if progress_callback:
            progress_callback("stage", "loading_model")
        flux = get_or_create_flux(
            model=model,
            image=input_image_path,  # Merk op: hier gebruiken we image parameter
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
            low_ram=low_ram,
            base_model_override=base_model_override
        )
        
        if not flux:
            return [], "Could not initialize Flux for image-to-image", prompt
            
        try:
            if progress_callback:
                progress_callback("image_start", {
                    "current_image": 1,
                    "total_images": 1,
                    "seed": seed,
                })
            print(f"Generating image-to-image with seed: {seed}")
            current_seed = seed

            # Guidance waarde: we zorgen ervoor dat het nooit None is
            if guidance is None:
                is_dev_model = "dev" in model
                guidance_value = 4.0 if is_dev_model else 0.0
            else:
                guidance_value = float(guidance)
                
            steps_int = 4 if not steps or steps.strip() == "" else int(steps)
            image_strength_float = float(image_strength)
            
            # Generate the image
            # Opmerking: In MFLUX v0.6.0 gebruiken we image_path en image_strength
            # in plaats van init_image en init_image_strength
            if is_flux2:
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    num_inference_steps=steps_int,
                    height=height,
                    width=width,
                    guidance=guidance_value,
                    image_paths=[Path(input_image_path)],
                    image_strength=image_strength_float,
                )
            else:
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    num_inference_steps=steps_int,
                    height=height,
                    width=width,
                    guidance=guidance_value,
                    image_path=Path(input_image_path),
                    image_strength=image_strength_float,
                )
            
            # Process results
            pil_image = generated.image
            timestamp = int(time.time())
            filename = f"i2i_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            meta = {
                "prompt": prompt,
                "seed": current_seed,
                "steps": steps_int,
                "guidance": guidance_value,
                "width": width,
                "height": height,
                "model": model,
                "image_strength": image_strength_float,
                "lora_files": lora_paths,
                "lora_scales": lora_scales_float,
            }
            save_image_with_metadata(pil_image, output_path, meta)
            
            # Save metadata if requested
            if metadata:
                metadata_path = os.path.join(OUTPUT_DIR, f"i2i_{timestamp}_{current_seed}.json")
                metadata_dict = {
                    "prompt": prompt,
                    "seed": current_seed,
                    "steps": steps_int,
                    "guidance": guidance_value,
                    "width": width,
                    "height": height,
                    "model": model,
                    "image_strength": image_strength_float,
                    "generation_time": str(time.ctime()),
                    "lora_files": lora_paths,
                    "lora_scales": lora_scales_float
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                    
            if progress_callback:
                progress_callback("image_complete", {
                    "current_image": 1,
                    "total_images": 1,
                })
            print(f"Generated image-to-image saved to {output_path}")
            return [pil_image], filename, prompt

        except Exception as e:
            if progress_callback:
                progress_callback("image_error", {"current_image": 1, "error": str(e)})
            print(f"Error in image-to-image generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], f"Error: {str(e)}", prompt
            
    except Exception as e:
        print(f"Error in image-to-image preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

def generate_image_in_context_lora_gradio(
    prompt, reference_image, model, base_model, seed, height, width, steps, guidance,
    lora_style, lora_files, metadata,
    prompt_file=None, config_from_metadata=None, stepwise_output_dir=None,
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False
):
    """
    Generate an image with in-context LoRA.
    """
    try:
        print("")
        print("--- Generating with In-Context LoRA ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"LoRA style: {lora_style}")
        print(f"Seed: {seed}")
        print(f"Dimensions: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Guidance: {guidance}")
        print(f"LoRA files: {lora_files}")
        print(f"LoRA scales: {lora_scales}")
        print(f"Low-RAM mode: {low_ram}")
        print_memory_usage("Before in-context LoRA generation")

        if is_flux2_model_name(model):
            return [], "FLUX.2 does not support In-Context LoRA.", prompt

        if reference_image is None:
            return [], "Reference image is required", prompt

        if prompt_file and str(prompt_file).strip():
            try:
                from backend.dynamic_prompts_manager import load_prompt_from_file
                file_prompt = load_prompt_from_file(str(prompt_file).strip())
                if file_prompt:
                    print(f"Loaded prompt from file '{prompt_file}': {file_prompt}")
                    prompt = file_prompt
                else:
                    print(f"Warning: Could not load prompt from file '{prompt_file}', using original prompt")
            except Exception as e:
                print(f"Error loading prompt file '{prompt_file}' in in-context LoRA: {str(e)}")
                print("Using original prompt instead")

        try:
            processed_prompt = process_dynamic_prompt(prompt)
            if processed_prompt != prompt:
                print(f"Dynamic prompt processed (in-context LoRA): {processed_prompt}")
            prompt = processed_prompt
        except Exception as e:
            print(f"Warning: Dynamic prompt processing failed in in-context LoRA: {e}")

        if isinstance(reference_image, (str, Path)):
            ref_image_path = str(reference_image)
        else:
            ref_image_path = os.path.join(OUTPUT_DIR, f"in_context_ref_{int(time.time())}.png")
            reference_image.save(ref_image_path)

        lora_paths = process_lora_files(lora_files) or []
        lora_scales_float = process_lora_files(lora_files, lora_scales) if lora_paths else []

        if lora_style:
            try:
                from mflux.models.flux.variants.in_context.utils.in_context_loras import get_lora_path
                style_entry = get_lora_path(lora_style)
                style_paths = process_lora_files([style_entry]) or []
                if style_paths:
                    lora_paths.extend(style_paths)
                    lora_scales_float = list(lora_scales_float)
                    lora_scales_float.extend([1.0] * len(style_paths))
            except Exception as e:
                print(f"Warning: Failed to resolve in-context LoRA style '{lora_style}': {e}")

        if not lora_paths:
            lora_paths = None
            lora_scales_float = None
        else:
            if lora_scales_float is None:
                lora_scales_float = []
            if len(lora_scales_float) < len(lora_paths):
                lora_scales_float.extend([1.0] * (len(lora_paths) - len(lora_scales_float)))
            elif len(lora_scales_float) > len(lora_paths):
                lora_scales_float = lora_scales_float[:len(lora_paths)]

        base_model_override = normalize_base_model_choice(base_model)
        model_config = resolve_mflux_model_config(model, base_model_override)
        model_path = resolve_local_path(strip_quant_suffix(model))

        quantize = None
        if "-8-bit" in model:
            quantize = 8
        elif "-4-bit" in model:
            quantize = 4
        elif "-6-bit" in model:
            quantize = 6
        elif "-3-bit" in model:
            quantize = 3
        elif low_ram:
            quantize = 8

        try:
            from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev
        except ModuleNotFoundError:
            from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev  # type: ignore

        try:
            flux = Flux1InContextDev(
                model_config=model_config,
                quantize=quantize,
                model_path=str(model_path) if model_path else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales_float,
            )
        except TypeError:
            flux = Flux1InContextDev(
                model_config=model_config,
                quantize=quantize,
                local_path=str(model_path) if model_path else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales_float,
            )

        seeds = []
        try:
            seed_int = None
            if seed not in (None, "", "None"):
                seed_int = int(seed)
            total_images = int(num_images) if num_images else 1
            for _ in range(total_images):
                seeds.append(get_next_seed(seed_int))
        except Exception as e:
            print(f"Warning: get_next_seed failed in in-context LoRA, using random seed: {e}")
            total_images = int(num_images) if num_images else 1
            seeds = [random.randint(0, 2**32 - 1) for _ in range(total_images)]

        results = []
        filenames = []

        for i, current_seed in enumerate(seeds, start=1):
            print(f"Generating in-context LoRA image {i}/{len(seeds)} with seed {current_seed}")
            if guidance is None:
                is_dev_model = "dev" in model
                guidance_value = 7.0 if is_dev_model else 0.0
            else:
                guidance_value = float(guidance)

            steps_int = 25 if not steps or str(steps).strip() == "" else int(steps)

            if "[IMAGE1]" not in prompt and "[LEFT]" not in prompt and "[REFERENCE]" not in prompt:
                in_context_prompt = f"This is a set of two side-by-side images. [LEFT] {prompt} [RIGHT] {prompt}"
            else:
                in_context_prompt = prompt

            generated = flux.generate_image(
                seed=current_seed,
                prompt=in_context_prompt,
                num_inference_steps=steps_int,
                height=height,
                width=width,
                guidance=guidance_value,
                image_path=ref_image_path,
            )

            output_image = generated.get_right_half().image if hasattr(generated, "get_right_half") else generated.image
            timestamp = int(time.time())
            filename = f"in_context_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            meta = {
                "prompt": prompt,
                "seed": current_seed,
                "steps": steps_int,
                "guidance": guidance_value,
                "width": width,
                "height": height,
                "model": model,
                "lora_style": lora_style,
                "generation_time": str(time.ctime()),
            }
            save_image_with_metadata(output_image, output_path, meta)

            if metadata:
                metadata_path = os.path.join(OUTPUT_DIR, f"in_context_{timestamp}_{current_seed}.json")
                metadata_dict = {
                    **meta,
                    "lora_files": lora_paths,
                    "lora_scales": lora_scales_float,
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)

            results.append(output_image)
            filenames.append(filename)

        if results:
            return results, ", ".join(filenames), prompt
        return [], "Error: Failed to generate any images", prompt

    except Exception as e:
        print(f"Error in in-context LoRA preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt

    finally:
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()


def generate_image_kontext_gradio(
    prompt, reference_image, model, seed, width, height, steps, guidance,
    lora_files, metadata, *lora_scales, num_images=1, low_ram=False
):
    """
    Generate an image with FLUX.1 Kontext for image editing via text instructions.
    """
    try:
        print("")
        print("--- Generating with FLUX.1 Kontext ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")

        if not reference_image:
            return [], "Reference image is required for Kontext generation", prompt

        if isinstance(reference_image, (str, Path)):
            ref_image_path = str(reference_image)
            with Image.open(ref_image_path) as ref_img:
                ref_width, ref_height = ref_img.size
        else:
            ref_image_path = os.path.join(OUTPUT_DIR, f"kontext_ref_{int(time.time())}.png")
            reference_image.save(ref_image_path)
            ref_width, ref_height = reference_image.size

        if not width or width == 0:
            width = ref_width
        if not height or height == 0:
            height = ref_height

        lora_paths = process_lora_files(lora_files)
        lora_scales_float = process_lora_files(lora_files, lora_scales) if lora_paths else None

        model_path = resolve_local_path(strip_quant_suffix(model))
        quantize = None
        if "-8-bit" in model:
            quantize = 8
        elif "-4-bit" in model:
            quantize = 4
        elif "-6-bit" in model:
            quantize = 6
        elif "-3-bit" in model:
            quantize = 3
        elif low_ram:
            quantize = 8

        try:
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
        except ModuleNotFoundError:
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext  # type: ignore

        model_config = ModelConfig.dev_kontext()
        try:
            flux = Flux1Kontext(
                model_config=model_config,
                quantize=quantize,
                model_path=str(model_path) if model_path else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales_float,
            )
        except TypeError:
            flux = Flux1Kontext(
                model_config=model_config,
                quantize=quantize,
                local_path=str(model_path) if model_path else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales_float,
            )

        seeds = []
        try:
            seed_int = None
            if seed not in (None, "", "None"):
                seed_int = int(seed)
            total_images = int(num_images) if num_images else 1
            for _ in range(total_images):
                seeds.append(get_next_seed(seed_int))
        except Exception as e:
            print(f"Warning: get_next_seed failed in Kontext, using random seed: {e}")
            total_images = int(num_images) if num_images else 1
            seeds = [random.randint(0, 2**32 - 1) for _ in range(total_images)]
        results = []
        filenames = []

        for current_seed in seeds:
            if guidance is None:
                guidance_value = 3.0
            else:
                guidance_value = float(guidance)
            steps_int = 25 if not steps or str(steps).strip() == "" else int(steps)

            generated = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                num_inference_steps=steps_int,
                height=height,
                width=width,
                guidance=guidance_value,
                image_path=ref_image_path,
            )

            pil_image = generated.image
            timestamp = int(time.time())
            filename = f"kontext_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            pil_image.save(output_path)

            if metadata:
                metadata_path = os.path.join(OUTPUT_DIR, f"kontext_{timestamp}_{current_seed}.json")
                metadata_dict = {
                    "prompt": prompt,
                    "seed": current_seed,
                    "steps": steps_int,
                    "guidance": guidance_value,
                    "width": width,
                    "height": height,
                    "model": "dev-kontext",
                    "generation_time": str(time.ctime()),
                    "reference_image_used": True,
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)

            results.append(pil_image)
            filenames.append(filename)

        if results:
            return results, ", ".join(filenames), prompt
        return [], "Error: No images were generated", prompt

    except Exception as e:
        print(f"Error in Kontext generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt

    finally:
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()
