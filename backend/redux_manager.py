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
from backend.mflux_compat import Config
from backend.mlx_utils import force_mlx_cleanup, print_memory_usage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_or_create_flux_redux(model, quantize=None, lora_paths=None, lora_scales=None, low_ram=False):
    """
    Create or retrieve a Flux instance for Redux (image variation).
    """
    try:
        try:
            from mflux.flux.flux import Flux1
        except ModuleNotFoundError:
            from mflux.models.flux.variants.txt2img.flux import Flux1
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
            
        print(f"Creating Flux Redux with model_config={custom_config}, quantize={quantize}")
        
        flux = Flux1(
            model_config=custom_config,
            quantize=quantize,
            local_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            enable_redux=True  # Enable Redux mode
        )
        
        return flux
        
    except Exception as e:
        print(f"Error creating Flux Redux instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_redux_gradio(
    prompt, reference_image, model, seed, height, width, steps, guidance,
    redux_strength, lora_files, metadata, *lora_scales, num_images=1, low_ram=False
):
    """
    Generate image variations using Redux.
    """
    try:
        # Parse inputs
        if not reference_image:
            return [], "Error: Reference image is required", prompt
            
        # Process LoRA files
        from backend.lora_manager import process_lora_files
        lora_paths, lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # Get or create flux instance
        flux = get_or_create_flux_redux(
            model=model,
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Error: Failed to initialize model", prompt
            
        # Load reference image
        ref_image = Image.open(reference_image).convert("RGB")
        
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
                    
                print(f"Generating Redux variation {i+1}/{num_images} with seed: {current_seed}")
                
                # Parse dimensions - use reference image dimensions if not specified
                if not height or height == 0:
                    height = ref_image.height
                if not width or width == 0:
                    width = ref_image.width
                    
                # Resize reference image if needed
                if ref_image.size != (width, height):
                    ref_image_resized = ref_image.resize((width, height), Image.LANCZOS)
                else:
                    ref_image_resized = ref_image
                    
                # Redux works best with dev model
                is_dev_model = "dev" in model
                
                # Use prompt if provided, otherwise use empty prompt for pure Redux
                final_prompt = prompt if prompt and prompt.strip() else ""
                
                # Determine default guidance based on model
                if not guidance or guidance == "":
                    guidance_value = 3.5 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)
                    
                steps_int = 20 if not steps or steps.strip() == "" else int(steps)
                redux_strength_float = float(redux_strength) if redux_strength else 0.8
                
                # Generate the image variation
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=final_prompt,
                    redux_image=ref_image_resized,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                        redux_strength=redux_strength_float,
                    ),
                )
                
                pil_image = generated.image
                
                # Save the image
                timestamp = int(time.time())
                filename = f"redux_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                # Save metadata if requested
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"redux_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "prompt": final_prompt,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "model": model,
                        "redux_strength": redux_strength_float,
                        "generation_time": str(time.ctime()),
                        "lora_files": lora_files,
                        "lora_scales": lora_scales_float
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated Redux variation saved to {output_path}")
                generated_images.append(pil_image)
                
            except Exception as e:
                print(f"Error generating Redux image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        if generated_images:
            return generated_images, f"Generated {len(generated_images)} variation(s)", prompt
        else:
            return [], "Error: Failed to generate any variations", prompt
            
    except Exception as e:
        print(f"Error in Redux generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

# Expose this module as `redux_manager` for easy import
import sys as _sys
redux_manager = _sys.modules[__name__]
