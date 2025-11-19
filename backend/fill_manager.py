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
from backend.model_manager import normalize_base_model_choice

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_or_create_flux_fill(quantize=None, low_ram=False):
    """
    Create or retrieve a Flux Fill instance.
    """
    try:
        try:
            from mflux.flux.flux import Flux1
        except ModuleNotFoundError:
            from mflux.models.flux.variants.txt2img.flux import Flux1
        
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

def _ensure_image(value, mode="RGB"):
    """
    Normalize incoming image data (path, PIL image, or sketch dict) to a PIL.Image.
    """
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert(mode)
    if isinstance(value, dict):
        # Sketch tool returns {"image": pil, "mask": pil}
        if "mask" in value and value["mask"] is not None:
            return _ensure_image(value["mask"], mode)
        if "image" in value and value["image"] is not None:
            return _ensure_image(value["image"], mode)
        return None
    if hasattr(value, "read"):
        return Image.open(value).convert(mode)
    return Image.open(value).convert(mode)


def generate_fill_gradio(
    prompt,
    image_input,
    mask_input,
    base_model,
    seed,
    height,
    width,
    steps,
    guidance,
    metadata,
    num_images=1,
    low_ram=False,
):
    """
    Generate images using the Fill tool for inpainting/outpainting.
    """
    try:
        # Parse inputs
        if not prompt or not prompt.strip():
            return [], "Error: Prompt is required", prompt
            
        if not image_input:
            return [], "Error: Input image is required", prompt
            
        if not mask_input:
            return [], "Error: Mask image is required", prompt

        original_image = _ensure_image(image_input, "RGB")
        mask_image = _ensure_image(mask_input, "L")

        if original_image is None or mask_image is None:
            return [], "Error: Could not read image or mask", prompt

        base_model_override = normalize_base_model_choice(base_model)
        if base_model_override and base_model_override not in {"dev", "dev-fill", "krea-dev"}:
            print(f"Warning: Fill only supports dev variants. Ignoring base_model={base_model_override}.")
            base_model_override = None
            
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
                        "base_model_override": base_model_override or "dev",
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
