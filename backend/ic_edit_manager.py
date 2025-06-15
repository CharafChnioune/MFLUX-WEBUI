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

def get_or_create_flux_ic_edit(model, quantize=None, low_ram=False):
    """
    Create or retrieve a Flux instance for IC-Edit (In-Context Editing).
    """
    try:
        from mflux.flux.flux import Flux1
        from backend.model_manager import get_custom_model_config
        
        base_model = model.replace("-8-bit", "").replace("-4-bit", "").replace("-6-bit", "").replace("-3-bit", "")
        
        try:
            custom_config = get_custom_model_config(base_model)
            if base_model in ["dev", "schnell"]:
                model_path = None
            else:
                model_path = os.path.join("models", base_model)
        except ValueError:
            from backend.model_manager import CustomModelConfig
            custom_config = CustomModelConfig(base_model, base_model, 1000, 512)
            model_path = os.path.join("models", base_model)
            
        if "-8-bit" in model:
            quantize = 8
        elif "-4-bit" in model:
            quantize = 4
        elif "-6-bit" in model:
            quantize = 6
        elif "-3-bit" in model:
            quantize = 3
            
        print(f"Creating Flux IC-Edit with model_config={custom_config}, quantize={quantize}")
        
        flux = Flux1(
            model_config=custom_config,
            quantize=quantize,
            local_path=model_path,
            enable_ic_edit=True  # Enable IC-Edit mode
        )
        
        return flux
        
    except Exception as e:
        print(f"Error creating Flux IC-Edit instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_ic_edit_gradio(
    original_image, edit_instruction, model, seed, height, width, steps, guidance,
    edit_strength, preserve_original, metadata, num_images=1, low_ram=False
):
    """
    Generate edited images using IC-Edit (In-Context Editing).
    """
    try:
        # Parse inputs
        if not original_image:
            return [], "Error: Original image is required"
            
        if not edit_instruction or not edit_instruction.strip():
            return [], "Error: Edit instruction is required"
            
        # Get or create flux instance
        flux = get_or_create_flux_ic_edit(
            model=model,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Error: Failed to initialize IC-Edit model"
            
        # Load original image
        orig_img = Image.open(original_image).convert("RGB")
        
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
                    
                print(f"Generating IC-Edit {i+1}/{num_images} with seed: {current_seed}")
                
                # Parse dimensions - use original image dimensions if not specified
                if not height or height == 0:
                    height = orig_img.height
                if not width or width == 0:
                    width = orig_img.width
                    
                # Ensure dimensions are multiples of 64
                height = (height // 64) * 64
                width = (width // 64) * 64
                
                # IC-Edit requires a special prompt format that describes the edit
                # Format: "Original: [description]. Edit: [instruction]"
                if preserve_original:
                    # Preserve more of the original image structure
                    ic_edit_prompt = f"Carefully edit the image while preserving most details. Edit: {edit_instruction}"
                else:
                    # More freedom in editing
                    ic_edit_prompt = f"Transform the image. Edit: {edit_instruction}"
                    
                # For IC-Edit, we need to prepare the image in a special way
                # The model expects a side-by-side format: original | edited
                combined_width = width * 2
                
                # Create a canvas for the side-by-side image
                canvas = Image.new('RGB', (combined_width, height))
                
                # Resize original image if needed
                if orig_img.size != (width, height):
                    orig_img_resized = orig_img.resize((width, height), Image.LANCZOS)
                else:
                    orig_img_resized = orig_img
                    
                # Place original image on the left side
                canvas.paste(orig_img_resized, (0, 0))
                # The right side will be generated
                
                # IC-Edit works best with dev model
                is_dev_model = "dev" in model
                
                # Determine default guidance based on model
                if not guidance or guidance == "":
                    guidance_value = 7.0 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)
                    
                steps_int = 25 if not steps or steps.strip() == "" else int(steps)
                edit_strength_float = float(edit_strength) if edit_strength else 0.8
                
                # Generate the edited image
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=ic_edit_prompt,
                    ic_edit_image=canvas,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=combined_width,
                        guidance=guidance_value,
                        edit_strength=edit_strength_float,
                    ),
                )
                
                # Extract the edited part (right side) from the generated image
                full_image = generated.image
                edited_image = full_image.crop((width, 0, combined_width, height))
                
                # Save the edited image
                timestamp = int(time.time())
                filename = f"ic_edit_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                edited_image.save(output_path)
                
                # Also save the full comparison image if requested
                if metadata:
                    comparison_filename = f"ic_edit_comparison_{timestamp}_{current_seed}.png"
                    comparison_path = os.path.join(OUTPUT_DIR, comparison_filename)
                    full_image.save(comparison_path)
                    
                    metadata_path = os.path.join(OUTPUT_DIR, f"ic_edit_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "edit_instruction": edit_instruction,
                        "ic_edit_prompt": ic_edit_prompt,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "model": model,
                        "edit_strength": edit_strength_float,
                        "preserve_original": preserve_original,
                        "generation_time": str(time.ctime()),
                        "original_image": os.path.basename(original_image),
                        "comparison_image": comparison_filename
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated IC-Edit saved to {output_path}")
                generated_images.append(edited_image)
                
            except Exception as e:
                print(f"Error generating IC-Edit image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        if generated_images:
            return generated_images, f"Generated {len(generated_images)} edited image(s)"
        else:
            return [], "Error: Failed to generate any edited images"
            
    except Exception as e:
        print(f"Error in IC-Edit generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}"
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

# Expose this module as `ic_edit_manager`
import sys as _sys
ic_edit_manager = _sys.modules[__name__]
