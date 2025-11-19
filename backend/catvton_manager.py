import os
import gc
import time
import random
import traceback
import mlx.core as mx
from PIL import Image
import numpy as np
from pathlib import Path
import json
from mflux.config.config import Config
from backend.mlx_utils import force_mlx_cleanup, print_memory_usage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_or_create_flux_catvton(model, quantize=None, low_ram=False, base_model_override=None):
    """
    Create or retrieve a Flux instance for CatVTON (Virtual Try-On).
    """
    try:
        try:
            from mflux.flux.flux import Flux1
        except ModuleNotFoundError:
            from mflux.models.flux.variants.txt2img.flux import Flux1
        from backend.model_manager import (
    CustomModelConfig,
    get_custom_model_config,
    normalize_base_model_choice,
    resolve_local_path,
)
        
        base_model = model.replace("-8-bit", "").replace("-4-bit", "").replace("-6-bit", "").replace("-3-bit", "")
        
        try:
            custom_config = get_custom_model_config(base_model)
            if base_model_override and base_model_override != custom_config.base_arch:
                custom_config = CustomModelConfig(
                    model_name=custom_config.model_name,
                    alias=custom_config.alias,
                    num_train_steps=custom_config.num_train_steps,
                    max_sequence_length=custom_config.max_sequence_length,
                    base_arch=base_model_override,
                    local_dir=custom_config.local_dir,
                )
            model_path = str(custom_config.local_dir) if custom_config.local_dir else None
        except ValueError:
            local_dir = resolve_local_path(base_model)
            custom_config = CustomModelConfig(
                base_model,
                base_model,
                1000,
                512,
                base_arch=base_model_override or "schnell",
                local_dir=local_dir,
            )
            model_path = str(local_dir) if local_dir else None
            
        if "-8-bit" in model:
            quantize = 8
        elif "-4-bit" in model:
            quantize = 4
        elif "-6-bit" in model:
            quantize = 6
        elif "-3-bit" in model:
            quantize = 3
            
        print(f"Creating Flux CatVTON with model_config={custom_config}, quantize={quantize}")
        
        flux = Flux1(
            model_config=custom_config,
            quantize=quantize,
            local_path=model_path,
            enable_catvton=True  # Enable CatVTON mode
        )
        
        return flux
        
    except Exception as e:
        print(f"Error creating Flux CatVTON instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_catvton_gradio(
    person_image, clothing_image, model, base_model, seed, height, width, steps, guidance,
    prompt_style, metadata, num_images=1, low_ram=False
):
    """
    Generate virtual try-on images using CatVTON.
    """
    try:
        # Parse inputs
        if not person_image:
            return [], "Error: Person image is required"
            
        if not clothing_image:
            return [], "Error: Clothing image is required"
            
        base_model_override = normalize_base_model_choice(base_model)
        # Get or create flux instance
        flux = get_or_create_flux_catvton(
            model=model,
            low_ram=low_ram,
            base_model_override=base_model_override,
        )
        
        if not flux:
            return [], "Error: Failed to initialize CatVTON model"
            
        # Load images
        person_img = Image.open(person_image).convert("RGB")
        clothing_img = Image.open(clothing_image).convert("RGB")
        
        generated_images = []
        
        for i in range(num_images):
            try:
                # Use provided seed or generate random one
                if seed and seed.strip():
                    if seed.lower() == "random":
                        current_seed = random.randint(0, 2**32 - 1)
                    else:
                        current_seed = int(seed) + i
                else:
                    current_seed = random.randint(0, 2**32 - 1)
                    
                print(f"Generating CatVTON try-on {i+1}/{num_images} with seed: {current_seed}")
                
                # Parse dimensions - use person image dimensions if not specified
                if not height or height == 0:
                    height = person_img.height
                if not width or width == 0:
                    width = person_img.width
                    
                # Ensure dimensions are multiples of 64
                height = (height // 64) * 64
                width = (width // 64) * 64
                
                # Resize images if needed
                if person_img.size != (width, height):
                    person_img_resized = person_img.resize((width, height), Image.LANCZOS)
                else:
                    person_img_resized = person_img
                    
                if clothing_img.size != (width, height):
                    clothing_img_resized = clothing_img.resize((width, height), Image.LANCZOS)
                else:
                    clothing_img_resized = clothing_img
                    
                # Create prompt based on style
                if prompt_style == "Professional":
                    prompt = "Professional fashion photoshoot, studio lighting, high quality"
                elif prompt_style == "Casual":
                    prompt = "Casual everyday wear, natural lighting, lifestyle photography"
                elif prompt_style == "Fashion":
                    prompt = "High fashion editorial, dramatic lighting, vogue style"
                elif prompt_style == "Outdoor":
                    prompt = "Outdoor fashion photography, natural environment, golden hour"
                else:  # Custom
                    prompt = prompt_style
                    
                # CatVTON works best with dev model
                is_dev_model = "dev" in model
                
                # Determine default guidance based on model
                if not guidance or guidance == "":
                    guidance_value = 5.0 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)
                    
                steps_int = 30 if not steps or steps.strip() == "" else int(steps)
                
                # Generate the virtual try-on image
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    person_image=person_img_resized,
                    clothing_image=clothing_img_resized,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                    ),
                )
                
                pil_image = generated.image
                
                # Save the image
                timestamp = int(time.time())
                filename = f"catvton_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                # Save metadata if requested
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"catvton_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "prompt": prompt,
                        "prompt_style": prompt_style,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "model": model,
                        "base_model_override": base_model_override,
                        "generation_time": str(time.ctime()),
                        "person_image": os.path.basename(person_image),
                        "clothing_image": os.path.basename(clothing_image)
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated CatVTON try-on saved to {output_path}")
                generated_images.append(pil_image)
                
            except Exception as e:
                print(f"Error generating CatVTON image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        if generated_images:
            return generated_images, f"Generated {len(generated_images)} try-on image(s)"
        else:
            return [], "Error: Failed to generate any try-on images"
            
    except Exception as e:
        print(f"Error in CatVTON generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}"
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

# Expose this module as `catvton_manager`
import sys as _sys
catvton_manager = _sys.modules[__name__]
