import os
import time
import gc
from PIL import Image
import gradio as gr
from backend.flux_manager import (
    get_or_create_flux,
    generate_image_batch,
    clear_flux_cache,
    force_mlx_cleanup,
    print_memory_usage
)
from backend.lora_manager import process_lora_files
from mflux.config.config import Config

def update_guidance_visibility(model):
    """
    Example: Make guidance visible if 'dev' is in the model name.
    """
    return gr.update(visible="dev" in model)

def simple_generate_image(prompt, model, image_format, lora_files, ollama_model, system_prompt, *lora_scales_and_num_images):
    """
    Simple interface to generate an image without too many parameters.
    """
    num_images = lora_scales_and_num_images[-1]
    lora_scales_list = lora_scales_and_num_images[:-1]

    print(f"\n--- Generating image (Easy) ---")
    print(f"Model: {model}")
    print(f"Format: {image_format}")
    
    # Convert format selection to actual dimensions
    width, height = get_dimensions_from_format(image_format)
    
    # Process LoRA files if specified
    lora_scales = None
    if lora_files and len(lora_files) > 0:
        if lora_scales_list and len(lora_scales_list) > 0:
            # Filter out non-visible scales (0.0)
            valid_loras = []
            valid_scales = []
            for i, lora_file in enumerate(lora_files):
                if i < len(lora_scales_list) and float(lora_scales_list[i]) > 0:
                    valid_loras.append(lora_file)
                    valid_scales.append(float(lora_scales_list[i]))
            
            lora_files = valid_loras
            lora_scales = valid_scales
    
    # Process the image format
    try:
        # Determine steps based on model name
        if model and ("schnell" in model.lower() or "4-bit" in model.lower() or "4bit" in model.lower()):
            steps = 4  # Fewer steps for faster models
        else:
            steps = 20  # More steps for better quality with dev model
            
        # Generate the output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Low RAM mode = False for simple interface (could be a parameter in future)
        low_ram = False
        
        # Enhance the prompt using Ollama if specified
        if ollama_model and system_prompt:
            from backend.ollama_manager import enhance_prompt as enhance_prompt_ollama
            enhanced_prompt = enhance_prompt_ollama(prompt, ollama_model, system_prompt)
            if enhanced_prompt:
                prompt = enhanced_prompt
        
        # Generate images
        output_path = generate_image_batch(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            num_inference_steps=steps,
            num_images=int(num_images) if num_images else 1,
            lora_files=lora_files,
            lora_scales=lora_scales,
            output_dir=output_dir,
            low_ram=low_ram
        )
        
        # Return both the gallery paths and the original prompt (potentially enhanced)
        return output_path, output_path, prompt
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in simple_generate_image: {str(e)}")
        print(error_trace)
        return f"Error: {str(e)}", [], prompt

def get_dimensions_from_format(format_str):
    """Get width and height dimensions from a format string."""
    dimensions = {
        "Portrait (576x1024)": (576, 1024),
        "Landscape (1024x576)": (1024, 576),
        "Background (1920x1080)": (1920, 1080),
        "Square (1024x1024)": (1024, 1024),
        "Poster (1080x1920)": (1080, 1920),
        "Wide Screen (2560x1440)": (2560, 1440),
        "Ultra Wide Screen (3440x1440)": (3440, 1440),
        "Banner (728x90)": (728, 90),
    }
    
    return dimensions.get(format_str, (576, 1024))  # Default to portrait if not found

def generate_images(prompt, model, num_images=1, steps=None, width=1024, height=1024,
                   lora_files=None, lora_scales=None, guidance=None,
                   seed=None, auto_seeds=None, metadata=False, low_ram=False,
                   negative_prompt=None, scheduler=None, controlnet_params=None):
    """
    Generates images using the MFLUX backend with specified parameters.
    
    Args:
        prompt: Text description of the image to generate
        model: The model to use (from user UI)
        num_images: Number of images to generate
        steps: Number of inference steps (default: 4 for schnell, 20 for dev)
        width: Width of the output image
        height: Height of the output image
        lora_files: List of LoRA files to use (from user UI)
        lora_scales: List of scale factors for each LoRA effect (from user UI)
        guidance: Guidance scale (only used for dev model)
        seed: Random seed for image generation
        auto_seeds: Number of auto-generated random seeds
        metadata: Whether to save image metadata
        low_ram: Enable low RAM mode to reduce memory usage
        negative_prompt: Text description of what to avoid in the image
        scheduler: The scheduler to use for generation
        controlnet_params: Parameters for ControlNet
    
    Returns:
        A string with the path to the generated image(s)
    """
    try:
        # Check if it's a JSON string that can be parsed
        if isinstance(prompt, str) and prompt.strip().startswith("{") and prompt.strip().endswith("}"):
            try:
                import json
                prompt_data = json.loads(prompt)
                # Extract values from JSON
                if isinstance(prompt_data, dict):
                    prompt_text = prompt_data.get("prompt", prompt)
                    if "lora_scales" in prompt_data and isinstance(prompt_data["lora_scales"], list):
                        lora_scales = prompt_data["lora_scales"]
                    if "lora_files" in prompt_data and isinstance(prompt_data["lora_files"], list):
                        lora_files = prompt_data["lora_files"]
                    # Using the provided prompt text
                    prompt = prompt_text
            except:
                # It was not a valid JSON string, use original prompt
                pass
    except:
        pass
    
    # Generate images
    try:
        # Check parameters and use defaults only if they're not provided at all
        if steps is None:
            if model and ("schnell" in model.lower() or "4-bit" in model.lower() or "4bit" in model.lower()):
                steps = 4  # Fewer steps for faster models
            else:
                steps = 20  # More steps for better quality with dev model
        
        # If guidance is None, set default based on model
        if guidance is None and model and "dev" in model.lower():
            try:
                # Try to parse JSON config for guidance parameter
                if isinstance(prompt, str) and prompt.strip().startswith("{") and prompt.strip().endswith("}"):
                    try:
                        import json
                        data = json.loads(prompt)
                        if isinstance(data, dict) and "guidance" in data:
                            guidance = float(data["guidance"])
                    except:
                        # Not valid json, use original
                        pass
            except:
                pass
            
            # Default guidance for dev model if not specified
            if guidance is None:
                guidance = 3.5
        
        # Set the output directory
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Check and process lora_files to ensure we have a valid list
        if isinstance(lora_files, str):
            try:
                # Try to parse JSON for lora_files
                if lora_files.strip().startswith("[") and lora_files.strip().endswith("]"):
                    import json
                    lora_files = json.loads(lora_files)
                else:
                    # Make sure lora_files is a list
                    lora_files = [lora_files]
            except:
                # If JSON parsing fails, treat as single string
                lora_files = [lora_files]
        
        # If we got a None or empty string for lora_files
        if not lora_files or (isinstance(lora_files, str) and not lora_files.strip()):
            lora_files = None
        
        # Check lora_scales parameter
        if lora_scales is not None:
            if isinstance(lora_scales, str):
                try:
                    # Try to parse JSON for lora_scales
                    if lora_scales.strip().startswith("[") and lora_scales.strip().endswith("]"):
                        import json
                        lora_scales = json.loads(lora_scales)
                    else:
                        # Convert comma-separated string to list
                        lora_scales = [float(s) for s in lora_scales.split(",")]
                except:
                    # If parsing fails, use default scale of 1.0
                    if lora_files and isinstance(lora_files, list):
                        lora_scales = [1.0] * len(lora_files)
                    else:
                        lora_scales = None
        
        # Make sure the list has the right length
        if lora_files and lora_scales and len(lora_scales) < len(lora_files):
            lora_scales.extend([1.0] * (len(lora_files) - len(lora_scales)))
        
        # Convert all scales to float values
        if lora_scales:
            lora_scales = [float(s) for s in lora_scales]
        
        # Use process_lora_files to get the actual file paths
        if lora_files:
            lora_files = process_lora_files(lora_files)
            if not lora_files:  # If no valid files found, reset lora_scales too
                lora_scales = None
        
        # Process the LoRA files
        lora_paths = []
        if lora_files:
            print(f"Using LoRA files: {lora_files}")
            if lora_scales:
                print(f"Using LoRA scales: {lora_scales}")
            lora_paths = lora_files
        
        # Use the existing MFLUX backend to generate images
        from mflux import Flux1, Config
        # Create the Flux instance
        try:
            flux = get_or_create_flux(model_name=model, lora_paths=lora_paths, lora_scales=lora_scales)
        except Exception as e:
            print(f"Error initializing Flux: {e}")
            # Try again with memory cleanup
            clear_flux_cache()
            force_mlx_cleanup()
            flux = get_or_create_flux(model_name=model, lora_paths=lora_paths, lora_scales=lora_scales)
        
        # Generate images
        if flux:
            output_paths = generate_image_batch(
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                num_inference_steps=steps,
                num_images=num_images,
                guidance=guidance,
                seed=seed,
                auto_seeds=auto_seeds,
                metadata=metadata,
                lora_files=lora_paths,
                lora_scales=lora_scales,
                low_ram=low_ram,
                negative_prompt=negative_prompt,
                scheduler=scheduler,
                controlnet_params=controlnet_params
            )
            
            # Wait a moment to ensure the image is generated
            time.sleep(0.5)
            
            # Log the absolute path of the generated images
            if isinstance(output_paths, list):
                for path in output_paths:
                    print(f"Generated image at: {os.path.abspath(path)}")
            else:
                print(f"Generated image at: {os.path.abspath(output_paths)}")
                
            # Check if the output_path is a list (multiple images) or a single string
            results = output_paths
            if isinstance(results, list) and len(results) == 1:
                results = results[0]  # Return single path for single image
                
            return results
        else:
            return "Error: Failed to initialize Flux model"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_images: {e}")
        print(error_trace)
        return f"Error: {str(e)}\n{error_trace}"

def parse_image_format(image_format):
    import re
    match = re.search(r"\((\d+)x(\d+)\)", image_format)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return (width, height)
    return (512, 512)

def generate_image(model, prompt, image_format="Portrait (576x1024)", num_images=1, steps=None, lora_files=None, lora_scales=None, guidance=None):
    """
    Functie voor CrewAI integratie.
    Genereert afbeeldingen via de MFLUX backend met opgegeven parameters.
    
    Args:
        model: Het te gebruiken model (schnell-4-bit, dev-4-bit)
        prompt: De tekstbeschrijving voor het beeld
        image_format: Formaat string in de vorm "Naam (WxH)"
        num_images: Aantal afbeeldingen om te genereren
        steps: Aantal inferentiestappen (standaard: 4 voor schnell, 20 voor dev)
        lora_files: Lijst met LoRA bestanden
        lora_scales: Lijst met schaalfactoren voor LoRA effecten
        guidance: Guidance schaal (standaard: 7.5 voor dev)
        
    Returns:
        Een string met het pad naar de gegenereerde afbeelding(en)
    """
    print(f"\n--- Generating image (CrewAI) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Image Format: {image_format}")
    print(f"Steps: {steps}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")
    
    # Verwerk het afbeeldingsformaat
    width, height = parse_image_format(image_format)
    
    # Verwerk LoRA bestanden
    valid_loras = process_lora_files(lora_files) if lora_files else None
    valid_lora_scales = lora_scales[:len(valid_loras)] if valid_loras and lora_scales else None
    
    # Stel standaardwaarden in
    if steps is None:
        steps = 4 if "schnell" in model else 20
    elif isinstance(steps, str):
        steps = int(steps)
    
    if guidance is None:
        guidance = 1.0 if "schnell" in model else 7.5
    
    try:
        # Maak de Flux-instantie
        flux = get_or_create_flux(model, None, None, valid_loras, valid_lora_scales)
        
        # Genereer afbeeldingen
        images, filenames, seeds = generate_image_batch(
            flux=flux,
            prompt=prompt,
            seed=None,
            steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            num_images=int(num_images)
        )
        
        # Voer geheugenopruiming uit
        clear_flux_cache()
        force_mlx_cleanup()
        
        # Geef de bestandsnamen terug
        if len(filenames) == 1:
            return filenames[0]
        return filenames
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_image_gradio(prompt, model, seed, width, height, steps, guidance, lora_files, metadata, ollama_model, system_prompt, *lora_scales_and_num_images):
    """
    Uitgebreide methode met handmatige parameters
    """
    num_images = lora_scales_and_num_images[-1]
    lora_scales_list = lora_scales_and_num_images[:-1]

    print(f"\n--- Generating image (Advanced) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales_list}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    valid_loras = process_lora_files(lora_files)
    lora_scales = lora_scales_list[:len(valid_loras)] if valid_loras else None

    try:
        seed_int = None if seed.strip() == "" else int(seed)
        steps_int = 4 if not steps or steps.strip() == "" else int(steps)

        flux = get_or_create_flux(model, None, None, valid_loras, lora_scales)
        images, filenames, seeds = generate_image_batch(
            flux=flux,
            prompt=prompt,
            seed=seed_int,
            steps=steps_int,
            height=height,
            width=width,
            guidance=guidance,
            num_images=int(num_images)
        )

        clear_flux_cache()
        force_mlx_cleanup()

        # Maak een informatieve output string
        output_info = []
        for filename, seed in zip(filenames, seeds):
            output_info.append(f"File: {filename} (Seed: {seed})")

        return images, "\n".join(output_info), prompt

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return [], "", prompt

def generate_image_controlnet_gradio(prompt, control_image, model, seed, height, width, steps, guidance, controlnet_strength, lora_files, metadata, save_canny, ollama_model, system_prompt, *lora_scales_and_num_images):
    """
    ControlNet image generation
    """
    num_images = lora_scales_and_num_images[-1]
    lora_scales_list = lora_scales_and_num_images[:-1]

    print(f"\n--- Generating image (ControlNet) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"ControlNet strength: {controlnet_strength}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales_list}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    valid_loras = process_lora_files(lora_files)
    lora_scales = lora_scales_list[:len(valid_loras)] if valid_loras else None

    try:
        seed_int = None if seed.strip() == "" else int(seed)
        steps_int = 4 if not steps or steps.strip() == "" else int(steps)

        config = Config()
        config.controlnet_conditioning_scale = float(controlnet_strength)
        config.save_controlnet_canny = bool(save_canny)

        flux = get_or_create_flux(model, config, control_image, valid_loras, lora_scales)
        images, filenames, seeds = generate_image_batch(
            flux=flux,
            prompt=prompt,
            seed=seed_int,
            steps=steps_int,
            height=height,
            width=width,
            guidance=guidance,
            num_images=int(num_images)
        )

        clear_flux_cache()
        force_mlx_cleanup()

        # Maak een informatieve output string
        output_info = []
        for filename, seed in zip(filenames, seeds):
            output_info.append(f"File: {filename} (Seed: {seed})")

        return images, "\n".join(output_info), prompt

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return [], "", prompt

def generate_image_i2i_gradio(prompt, init_image, init_image_strength, model, seed, width, height, steps, guidance, lora_files, metadata, ollama_model, system_prompt, *lora_scales_and_num_images):
    """
    Image-to-Image generation
    """
    num_images = lora_scales_and_num_images[-1]
    lora_scales_list = lora_scales_and_num_images[:-1]

    if init_image is None:
        print("Error: No initial image provided")
        return [], "", prompt

    print(f"\n--- Generating image (Image-to-Image) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"Image-to-Image strength: {init_image_strength}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales_list}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    valid_loras = process_lora_files(lora_files)
    lora_scales = lora_scales_list[:len(valid_loras)] if valid_loras else None

    try:
        seed_int = None if seed.strip() == "" else int(seed)
        steps_int = 4 if not steps or steps.strip() == "" else int(steps)

        config = Config()
        config.image_to_image = True
        config.image_to_image_strength = float(init_image_strength)

        flux = get_or_create_flux(model, config, init_image, valid_loras, lora_scales)
        images, filenames, seeds = generate_image_batch(
            flux=flux,
            prompt=prompt,
            seed=seed_int,
            steps=steps_int,
            height=height,
            width=width,
            guidance=guidance,
            num_images=int(num_images)
        )

        clear_flux_cache()
        force_mlx_cleanup()

        # Maak een informatieve output string
        output_info = []
        for filename, seed in zip(filenames, seeds):
            output_info.append(f"File: {filename} (Seed: {seed})")

        return images, "\n".join(output_info), prompt

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return [], "", prompt

def process_controlnet_image(flux_model, cnn_image, additional_prompt, prompt_strength=0.5, controlnet_strength=0.5):
    """
    Process a control image with controlnet.
    """
    CONFIG = {}
    # Controlnet config
    config = Config()
    config.height = 1024
    config.width = 1024
    config.guidance = 7.0
    config.controlnet_strength = controlnet_strength
