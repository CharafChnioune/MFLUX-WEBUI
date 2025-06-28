import os
import gc
import time
import random
import traceback
import mlx.core as mx
from PIL import Image
from mflux.config.config import Config
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from backend.lora_manager import process_lora_files, download_lora
from backend.ollama_manager import enhance_prompt
from backend.prompts_manager import enhance_prompt_with_mlx
from backend.mlx_utils import force_mlx_cleanup, print_memory_usage
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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def get_or_create_flux(model, config=None, image=None, lora_paths=None, lora_scales=None, is_controlnet=False, low_ram=False):
    """
    Create or retrieve a Flux1 instance.
    """
    try:
        base_model = model.replace("-8-bit", "").replace("-4-bit", "").replace("-6-bit", "").replace("-3-bit", "")
        
        try:
            from backend.model_manager import get_custom_model_config
            custom_config = get_custom_model_config(base_model)
            if base_model in ["dev", "schnell", "dev-8-bit", "dev-4-bit", "dev-6-bit", "dev-3-bit", 
                             "schnell-8-bit", "schnell-4-bit", "schnell-6-bit", "schnell-3-bit"]:
                model_path = None
                # Base model architecture is the same as model name for official models
                base_model_arch = base_model.split("-")[0] if "-" in base_model else base_model
            else:
                model_path = os.path.join("models", base_model)
                # For third-party models, get base architecture from custom_config
                base_model_arch = custom_config.base_arch if hasattr(custom_config, 'base_arch') else 'schnell'
        except ValueError:
            from backend.model_manager import CustomModelConfig
            custom_config = CustomModelConfig(base_model, base_model, 1000, 512)
            model_path = os.path.join("models", base_model)
            base_model_arch = 'schnell'  # Default to schnell architecture for custom models

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

        FluxClass = Flux1Controlnet if is_controlnet else Flux1
        print(f"Creating {FluxClass.__name__} with model_config={custom_config}, quantize={quantize}, local_path={model_path}, lora_paths={lora_paths}, lora_scales={lora_scales}")
        try:
            # Special handling for lora_scales to work with mflux library's internals.
            # mflux library does: flux_model.lora_scales = (lora_scales or []) + [1.0] * len(hf_lora_paths)
            # So we need to pass lora_scales as a LIST instead of a tuple to allow this to work
            
            # If no lora paths, don't pass any scales
            if not lora_paths:
                lora_scales = None
            # If lora paths but no scales, pass an empty list to allow concatenation
            elif lora_scales is None:
                lora_scales = []
            # If lora paths and scales, ensure scales is a list
            elif isinstance(lora_scales, tuple):
                lora_scales = list(lora_scales)
            
            flux = FluxClass(
                model_config=custom_config,
                quantize=quantize,
                local_path=model_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales
            )
            return flux
        except Exception as e:
            print(f"Error instantiating {FluxClass.__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
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

def generate_image_batch(flux, prompt, seed, steps, height, width, guidance, num_images):
    """
    Generate a batch of images using the Flux model.
    """
    images = []
    filenames = []
    seeds_used = []
    
    for i in range(num_images):
        current_seed = seed if seed is not None else int(time.time()) + i
        seeds_used.append(current_seed)
        output_filename = f"generated_{int(time.time())}_{i}_seed_{current_seed}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        generated = flux.generate_image(
            seed=current_seed,
            prompt=prompt,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            ),
        )
        # Sla het GeneratedImage object op als PIL Image
        generated.image.save(output_path)
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
    
    if hasattr(mx.metal, 'clear_cache'):
        mx.metal.clear_cache()
    
    if hasattr(mx.metal, 'reset_peak_memory'):
        mx.metal.reset_peak_memory()

    gc.collect()

def print_memory_usage(label):
    """
    Print the current memory usage.
    """
    try:
        active_memory = mx.metal.get_active_memory() / 1e6
        peak_memory = mx.metal.get_peak_memory() / 1e6
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
                seed = int(time.time()) + i
                output_filename = f"generated_{int(time.time())}_{i}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                try:
                    print(f"Generating image {i+1} of {num_images} with seed {seed}...")
                    
                    # Guidance waarde instellen op basis van model type
                    # Voor "dev" model gebruiken we 4.0, voor "schnell" 0.0
                    # We zetten het nooit op None
                    is_dev_model = "dev" in model
                    guidance_value = 4.0 if is_dev_model else 0.0
                    
                    generated = flux.generate_image(
                        seed=seed,
                        prompt=prompt,
                        config=Config(
                            num_inference_steps=4 if "schnell" in model else 20,
                            height=height,
                            width=width,
                            guidance=guidance_value,
                        ),
                    )
                    
                    pil_image = generated.image
                    pil_image.save(output_path)
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
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False, auto_seeds=None
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

        # 1. Pre-generation checks (battery, config validation, etc.)
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

        # 7. Process LoRA files and scales
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # 8. Enhance prompt with Ollama if requested
        if ollama_model and system_prompt:
            prompt = enhance_prompt(prompt, ollama_model, system_prompt)
            print(f"Ollama enhanced prompt: {prompt}")
            
        # 9. Determine seeds to use (auto-seeds integration)
        seeds = []
        if auto_seeds and auto_seeds > 0:
            # Legacy auto-seeds support
            seeds = [random.randint(1, 1000000000) for _ in range(auto_seeds)]
        elif seed is not None:
            seeds = [seed]
        else:
            # Use workflow seed management (integrates with auto-seeds manager)
            seeds = [get_next_seed(seed) for _ in range(num_images)]
        
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
                flux = get_or_create_flux(
                    model=model,
                    lora_paths=lora_files,
                    lora_scales=lora_scales_float,
                    low_ram=low_ram
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
                
                steps_int = 4 if not steps or steps.strip() == "" else int(steps)
                
                # Generate the image with progress monitoring
                print(f"Generating image {i+1}/{len(seeds)} with seed: {current_seed}")
                generation_config = Config(
                    num_inference_steps=steps_int,
                    height=height,
                    width=width,
                    guidance=guidance_value,
                )
                
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    config=generation_config,
                )
                
                # Process the result
                pil_image = generated.image
                timestamp = int(time.time())
                filename = f"generated_{timestamp}_{current_seed}.png"
                
                # Save the image
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
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
                    "lora_files": lora_files,
                    "lora_scales": lora_scales_float,
                    "low_ram_mode": low_ram,
                    "filename": filename
                }
                
                # Save enhanced metadata using workflow
                if metadata:
                    save_enhanced_metadata(Path(output_path), generation_metadata)
                
                all_images.append(pil_image)
                all_seeds.append(current_seed)
                all_metadata.append(generation_metadata)
                
                # Update statistics
                update_generation_stats(success=True)
                
                print(f"Successfully generated image {i+1} with seed {current_seed}")
                
            except Exception as e:
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
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False
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
        
        # Generate temporary file path for control image
        control_image_path = os.path.join(OUTPUT_DIR, f"controlnet_input_{int(time.time())}.png")
        control_image.save(control_image_path)
        
        # Process lora files and scales
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # Auto-generate seed if not provided
        if not seed:
            seed = get_random_seed()
        else:
            try:
                seed = int(seed)
            except ValueError:
                seed = get_random_seed()
                print(f"Invalid seed, using random seed: {seed}")
        
        # Initialize Flux1Controlnet
        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_files,
            lora_scales=lora_scales_float,
            is_controlnet=True,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Could not initialize ControlNet", prompt
            
        try:
            print(f"Generating controlnet image with seed: {seed}")
            current_seed = seed
            
            # Guidance waarde: we zorgen ervoor dat het nooit None is
            if guidance is None:
                is_dev_model = "dev" in model
                guidance_value = 4.0 if is_dev_model else 0.0
            else:
                guidance_value = float(guidance)
                
            steps_int = 4 if not steps or steps.strip() == "" else int(steps)
            controlnet_strength_float = float(controlnet_strength)
            
            # Generate the image
            generated = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                controlnet_image_path=control_image_path,
                config=Config(
                    num_inference_steps=steps_int,
                    height=height,
                    width=width,
                    guidance=guidance_value,
                    controlnet_strength=controlnet_strength_float,
                ),
            )
            
            # Process results
            pil_image = generated.image
            timestamp = int(time.time())
            filename = f"controlnet_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            pil_image.save(output_path)
            
            # Save canny reference if requested
            if save_canny:
                # Maak de canny edge detectie afbeelding met ControlnetUtil
                from mflux.controlnet.controlnet_util import ControlnetUtil
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
                    "lora_files": lora_files,
                    "lora_scales": lora_scales_float
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                    
            print(f"Generated controlnet image saved to {output_path}")
            return [pil_image], filename, prompt
            
        except Exception as e:
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
    vae_tiling=False, vae_tiling_split=1, *lora_scales, num_images=1, low_ram=False
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
        
        # Generate temporary file path for input image
        input_image_path = os.path.join(OUTPUT_DIR, f"i2i_input_{int(time.time())}.png")
        input_image.save(input_image_path)
        
        # Process lora files and scales
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # Auto-generate seed if not provided
        if not seed:
            seed = get_random_seed()
        else:
            try:
                seed = int(seed)
            except ValueError:
                seed = get_random_seed()
                print(f"Invalid seed, using random seed: {seed}")
        
        # Initialize Flux
        flux = get_or_create_flux(
            model=model,
            image=input_image_path,  # Merk op: hier gebruiken we image parameter
            lora_paths=lora_files,
            lora_scales=lora_scales_float,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Could not initialize Flux for image-to-image", prompt
            
        try:
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
            generated = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps_int,
                    height=height,
                    width=width,
                    guidance=guidance_value,
                    image_path=Path(input_image_path),
                    image_strength=image_strength_float,
                ),
            )
            
            # Process results
            pil_image = generated.image
            timestamp = int(time.time())
            filename = f"i2i_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            pil_image.save(output_path)
            
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
                    "lora_files": lora_files,
                    "lora_scales": lora_scales_float
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                    
            print(f"Generated image-to-image saved to {output_path}")
            return [pil_image], filename, prompt
            
        except Exception as e:
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
        print(f"\n--- Generating with In-Context LoRA ---")
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

        # Process parameters
        from mflux.community.in_context_lora.in_context_loras import LORA_NAME_MAP, LORA_REPO_ID
        from backend.lora_manager import download_lora
        
        # Save reference image
        ref_image_path = os.path.join(OUTPUT_DIR, f"in_context_ref_{int(time.time())}.png")
        reference_image.save(ref_image_path)
        
        # Process lora files and scales
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # Download the style LoRA if specified
        if lora_style:
            lora_filename = LORA_NAME_MAP.get(lora_style)
            if lora_filename:
                print(f"Downloading LoRA style: {lora_style} ({lora_filename})")
                lora_path = download_lora(lora_filename, repo_id=LORA_REPO_ID)
                if lora_path and os.path.exists(lora_path):
                    # Add the style LoRA to the list
                    if lora_files is None:
                        lora_files = [lora_path]
                    else:
                        lora_files.append(lora_path)
                    
                    # Add a default scale for the style LoRA
                    if lora_scales_float is None:
                        lora_scales_float = [1.0]
                    else:
                        lora_scales_float.append(1.0)
                        
                    print(f"Added style LoRA: {lora_path}")
        
        # Auto-generate seed if not provided
        if not seed:
            seed = get_random_seed()
        else:
            try:
                seed = int(seed)
            except ValueError:
                seed = get_random_seed()
                print(f"Invalid seed, using random seed: {seed}")
        
        # Initialize Flux
        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_files,
            lora_scales=lora_scales_float,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Could not initialize Flux for in-context LoRA", prompt
            
        try:
            print(f"Generating with in-context LoRA, seed: {seed}")
            current_seed = seed
            
            # Guidance waarde: we zorgen ervoor dat het nooit None is
            if guidance is None:
                is_dev_model = "dev" in model
                guidance_value = 7.0 if is_dev_model else 0.0  # In-context LoRA works better with higher guidance
            else:
                guidance_value = float(guidance)
                
            steps_int = 25 if not steps or steps.strip() == "" else int(steps)  # In-context LoRA needs more steps
            
            # Generate the image
            # Aangepaste implementatie voor in-context LoRA generatie
            
            # 1. Laad de referentie afbeelding
            ref_image = Image.open(ref_image_path)
            
            # 2. Bereid een prompt voor met "side-by-side" beschrijving (belangrijk voor in-context LoRA)
            # In-context LoRA werkt door een prompt te gebruiken die beide afbeeldingen beschrijft
            if "[IMAGE1]" not in prompt and "[LEFT]" not in prompt and "[REFERENCE]" not in prompt:
                # Als er geen markers zijn, voeg ze toe
                in_context_prompt = f"This is a set of two side-by-side images. [LEFT] {prompt} [RIGHT] {prompt}"
            else:
                # Gebruik de bestaande markers in de prompt
                in_context_prompt = prompt
            
            # 3. Stel de breedte in op twee keer de normale breedte om ruimte te maken voor beide afbeeldingen
            combined_width = width * 2
            
            # 4. Genereer de afbeelding met de side-by-side prompt
            generated = flux.generate_image(
                seed=current_seed,
                prompt=in_context_prompt,
                config=Config(
                    num_inference_steps=steps_int,
                    height=height,
                    width=combined_width,
                    guidance=guidance_value,
                ),
            )
            
            # 5. Knip de rechterkant van de gegenereerde afbeelding om alleen het resultaat te tonen
            # In-context LoRA genereert een side-by-side afbeelding met links de referentie en rechts het resultaat
            full_image = generated.image
            right_half = full_image.crop((width, 0, combined_width, height))
            
            # Gebruik deze rechterkant als het uiteindelijke resultaat
            pil_image = right_half
            
            # Process results
            timestamp = int(time.time())
            filename = f"in_context_{timestamp}_{current_seed}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            pil_image.save(output_path)
            
            # Save metadata if requested
            if metadata:
                metadata_path = os.path.join(OUTPUT_DIR, f"in_context_{timestamp}_{current_seed}.json")
                metadata_dict = {
                    "prompt": prompt,
                    "seed": current_seed,
                    "steps": steps_int,
                    "guidance": guidance_value,
                    "width": width,
                    "height": height,
                    "model": model,
                    "lora_style": lora_style,
                    "generation_time": str(time.ctime()),
                    "lora_files": lora_files,
                    "lora_scales": lora_scales_float
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                    
            print(f"Generated in-context LoRA image saved to {output_path}")
            return [pil_image], filename, prompt
            
        except Exception as e:
            print(f"Error in in-context LoRA generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], f"Error: {str(e)}", prompt
            
    except Exception as e:
        print(f"Error in in-context LoRA preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()



def generate_image_kontext_gradio(
    prompt, reference_image, seed, height, width, steps, guidance,
    metadata, prompt_file=None, config_from_metadata=None, stepwise_output_dir=None, 
    vae_tiling=False, vae_tiling_split="horizontal", num_images=1, low_ram=False
):
    """
    Generate an image with FLUX.1 Kontext for image editing via text instructions.
    Kontext automatically uses the dev-kontext model and requires a reference image.
    """
    try:
        print(f"\n--- Generating with FLUX.1 Kontext ---")
        print(f"Reference image provided: {reference_image is not None}")
        print(f"Prompt: {prompt}")
        print(f"Guidance: {guidance} (recommended 2.0-4.0 for Kontext)")
        
        if not reference_image:
            error_msg = "Reference image is required for Kontext generation"
            print(f"Error: {error_msg}")
            return [], error_msg, prompt
        
        # Handle dynamic prompts from file if specified
        if prompt_file and os.path.exists(prompt_file):
            from backend.managers.dynamic_prompts_manager import load_prompt_from_file
            try:
                loaded_prompt = load_prompt_from_file(prompt_file)
                if loaded_prompt:
                    prompt = loaded_prompt
                    print(f"Loaded prompt from file: {prompt}")
            except Exception as e:
                print(f"Warning: Could not load prompt from file {prompt_file}: {e}")
        
        # Handle config from metadata if specified
        if config_from_metadata and os.path.exists(config_from_metadata):
            from backend.managers.metadata_config_manager import extract_config_from_metadata
            try:
                config_dict = extract_config_from_metadata(config_from_metadata)
                if config_dict:
                    # Apply extracted config (override current settings)
                    height = config_dict.get('height', height)
                    width = config_dict.get('width', width)
                    steps = config_dict.get('steps', steps)
                    guidance = config_dict.get('guidance', guidance)
                    print(f"Applied config from metadata: {config_dict}")
            except Exception as e:
                print(f"Warning: Could not extract config from metadata {config_from_metadata}: {e}")
        
        # Validate and convert parameters
        try:
            height = int(height)
            width = int(width)
            steps_int = int(steps)
            guidance_value = float(guidance)
        except ValueError as e:
            error_msg = f"Invalid parameter values: {str(e)}"
            print(f"Error: {error_msg}")
            return [], error_msg, prompt
        
        # Kontext-specific validation
        if guidance_value < 1.0 or guidance_value > 6.0:
            print(f"Warning: Guidance {guidance_value} is outside recommended range 2.0-4.0 for Kontext")
        
        # Process reference image
        try:
            if isinstance(reference_image, str) and os.path.exists(reference_image):
                reference_pil = Image.open(reference_image).convert('RGB')
            elif hasattr(reference_image, 'save'):  # PIL Image
                reference_pil = reference_image.convert('RGB')
            else:
                error_msg = "Invalid reference image format"
                print(f"Error: {error_msg}")
                return [], error_msg, prompt
        except Exception as e:
            error_msg = f"Error processing reference image: {str(e)}"
            print(f"Error: {error_msg}")
            return [], error_msg, prompt
        
        # Auto-generate seed if not provided
        if not seed:
            seed = get_random_seed()
        else:
            try:
                seed = int(seed)
            except ValueError:
                seed = get_random_seed()
                print(f"Invalid seed, using random seed: {seed}")
        
        # Initialize Flux with dev-kontext model (Kontext automatically uses this model)
        flux = get_or_create_flux(
            model="dev-kontext",  # Kontext always uses dev-kontext model
            lora_paths=None,      # Kontext doesn't use LoRA files
            lora_scales=None,
            low_ram=low_ram
        )
        
        generated_images = []
        filenames = []
        
        for i in range(num_images):
            current_seed = seed + i if num_images > 1 else seed
            print(f"Generating Kontext image {i+1}/{num_images} with seed {current_seed}")
            
            try:
                # Generate image with Kontext
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                        init_image_strength=0.95  # Standard for Kontext editing
                    ),
                    init_image=reference_pil
                )
                
                # Convert to PIL
                if isinstance(generated, list) and len(generated) > 0:
                    pil_image = generated[0]
                else:
                    pil_image = generated
                
                if not isinstance(pil_image, Image.Image):
                    pil_image = Image.fromarray((pil_image * 255).astype(np.uint8))
                
                generated_images.append(pil_image)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"kontext_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                filenames.append(filename)
                
                # Handle stepwise output if specified
                if stepwise_output_dir:
                    from backend.managers.stepwise_output_manager import save_stepwise_output
                    try:
                        save_stepwise_output(
                            stepwise_output_dir, 
                            pil_image, 
                            current_seed, 
                            steps_int,
                            "kontext"
                        )
                    except Exception as e:
                        print(f"Warning: Could not save stepwise output: {e}")
                
                # Save metadata if requested
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
                        "kontext_editing": True
                    }
                    
                    # Add MFLUX v0.9.0 metadata
                    if prompt_file:
                        metadata_dict["prompt_file"] = prompt_file
                    if config_from_metadata:
                        metadata_dict["config_from_metadata"] = config_from_metadata
                    if stepwise_output_dir:
                        metadata_dict["stepwise_output_dir"] = stepwise_output_dir
                    if vae_tiling:
                        metadata_dict["vae_tiling"] = vae_tiling
                        metadata_dict["vae_tiling_split"] = vae_tiling_split
                    
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                
                print(f"Generated Kontext image saved to {output_path}")
                
            except Exception as e:
                print(f"Error generating Kontext image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if generated_images:
            return generated_images, ", ".join(filenames), prompt
        else:
            return [], "Error: No images were generated", prompt
            
    except Exception as e:
        print(f"Error in Kontext generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

