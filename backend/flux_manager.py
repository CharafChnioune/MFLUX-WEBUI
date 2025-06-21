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
import json
from pathlib import Path
import numpy as np

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
            # Convert lora_scales to tuple if it's a list to avoid type error
            if lora_scales is not None and not isinstance(lora_scales, tuple):
                lora_scales = tuple(lora_scales)
            
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
    prompt, model, seed, width, height, steps, guidance, 
    lora_files, metadata, ollama_model=None, system_prompt=None,
    *lora_scales, num_images=1, low_ram=False, auto_seeds=None
):
    """
    Generate images using the Flux model through the Gradio interface.
    """
    try:
        print(f"\n--- Generating image (Gradio) ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Dimensions: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Guidance: {guidance}")
        print(f"Seed: {seed}")
        print(f"Auto Seeds: {auto_seeds}")
        print(f"LoRA files: {lora_files}")
        print(f"LoRA scales: {lora_scales}")
        print(f"Number of Images: {num_images}")
        print(f"Low-RAM mode: {low_ram}")
        print_memory_usage("Before generation")

        # Omzetten van lora_scales van tuple van strings naar lijst van floats
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # Enhance prompt if requested
        if ollama_model and system_prompt:
            prompt = enhance_prompt(prompt, ollama_model, system_prompt)
            
        # Auto seeds verwerken
        if auto_seeds and auto_seeds > 0:
            seeds = [random.randint(1, 1000000000) for _ in range(auto_seeds)]
        else:
            seeds = [seed] if seed else [get_random_seed()]
        
        all_images = []
        all_seeds = []
        
        for current_seed in seeds:
            # Flux model initialiseren
            flux = get_or_create_flux(
                model=model,
                lora_paths=lora_files,
                lora_scales=lora_scales_float,
                low_ram=low_ram
            )
            
            if not flux:
                return [], "", prompt
                
            try:
                # Guidance waarde: we zorgen ervoor dat het nooit None is
                if guidance is None:
                    is_dev_model = "dev" in model
                    guidance_value = 4.0 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)
                
                steps_int = 4 if not steps or steps.strip() == "" else int(steps)
                
                # Genereer de afbeelding
                print(f"Generating with seed: {current_seed}, steps: {steps_int}, guidance: {guidance_value}")
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                    ),
                )
                
                # Verwerk het resultaat
                pil_image = generated.image
                timestamp = int(time.time())
                filename = f"generated_{timestamp}_{current_seed}.png"
                
                # Sla de afbeelding op
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                # Optioneel: metadata opslaan
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"generated_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "prompt": prompt,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "model": model,
                        "generation_time": str(time.ctime()),
                        "lora_files": lora_files,
                        "lora_scales": lora_scales_float
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                
                all_images.append(pil_image)
                all_seeds.append(current_seed)
                
                print(f"Generated image with seed {current_seed} saved to {output_path}")
                
            except Exception as e:
                print(f"Error generating image with seed {current_seed}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Cleanup
                if 'flux' in locals():
                    del flux
                gc.collect()
                force_mlx_cleanup()
        
        # Bereid de output voor
        if all_images:
            seed_info = [f"Seed: {seed}" for seed in all_seeds]
            return all_images, "\n".join(seed_info), prompt
        else:
            return [], "No images were generated", prompt
            
    except Exception as e:
        print(f"Error in generate_image_gradio: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt

def generate_image_controlnet_gradio(
    prompt, control_image, model, seed, height, width, steps, guidance,
    controlnet_strength, lora_files, metadata, save_canny,
    *lora_scales, num_images=1, low_ram=False
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
    prompt, input_image, model, seed, height, width, steps, guidance,
    image_strength, lora_files, metadata, *lora_scales, num_images=1, low_ram=False
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
    prompt, reference_image, model, seed, height, width, steps, guidance,
    lora_style, lora_files, metadata, *lora_scales, num_images=1, low_ram=False
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
