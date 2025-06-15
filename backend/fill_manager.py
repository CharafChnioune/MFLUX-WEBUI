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

def get_or_create_flux_fill(quantize=None, low_ram=False):
    """
    Create or retrieve a Flux Fill instance.
    """
    try:
        from mflux.flux.flux import Flux1
        
        # Fill model is always FLUX.1-Fill-dev
        model_config = "flux.1-fill-dev"
        
        print(f"Creating Flux Fill with model_config={model_config}, quantize={quantize}")
        
        flux = Flux1.from_huggingface(
            model_name=model_config,
            quantize=quantize
        )
        
        return flux
        
    except Exception as e:
        print(f"Error creating Flux Fill instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_fill_gradio(
    prompt, image_path, mask_path, seed, height, width, steps, guidance,
    metadata, num_images=1, low_ram=False
):
    """
    Generate images using the Fill tool for inpainting/outpainting.
    """
    try:
        # Parse inputs
        if not prompt or not prompt.strip():
            return [], "Error: Prompt is required", prompt
            
        if not image_path:
            return [], "Error: Input image is required", prompt
            
        if not mask_path:
            return [], "Error: Mask image is required", prompt
            
        # Get or create flux fill instance
        flux = get_or_create_flux_fill(quantize=8 if low_ram else None, low_ram=low_ram)
        if not flux:
            return [], "Error: Failed to initialize Fill model", prompt
            
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
                    
                print(f"Generating fill image {i+1}/{num_images} with seed: {current_seed}")
                
                # Load images
                original_image = Image.open(image_path).convert("RGB")
                mask_image = Image.open(mask_path).convert("L")  # Grayscale mask
                
                # Parse dimensions
                if not height or height == 0:
                    height = original_image.height
                if not width or width == 0:
                    width = original_image.width
                    
                # Resize images if needed
                if original_image.size != (width, height):
                    original_image = original_image.resize((width, height), Image.LANCZOS)
                    mask_image = mask_image.resize((width, height), Image.LANCZOS)
                    
                # Fill tool parameters
                guidance_value = float(guidance) if guidance else 30.0  # Fill works best with high guidance
                steps_int = int(steps) if steps else 25
                
                # Generate the filled image
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    image=original_image,
                    mask=mask_image,
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
                filename = f"fill_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                # Save metadata if requested
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"fill_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "prompt": prompt,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "generation_time": str(time.ctime()),
                        "fill_type": "inpaint/outpaint"
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated fill image saved to {output_path}")
                generated_images.append(pil_image)
                
            except Exception as e:
                print(f"Error generating fill image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        if generated_images:
            return generated_images, f"Generated {len(generated_images)} image(s)", prompt
        else:
            return [], "Error: Failed to generate any images", prompt
            
    except Exception as e:
        print(f"Error in fill generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()
