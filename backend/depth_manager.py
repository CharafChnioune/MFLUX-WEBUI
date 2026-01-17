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
from backend.model_manager import (
    CustomModelConfig,
    get_custom_model_config,
    resolve_local_path,
    normalize_base_model_choice,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_depth_map(image_path):
    """
    Generate a depth map from an image using Depth Pro.
    """
    try:
        try:
            from mflux.models.depth_pro.model.depth_pro import DepthPro
        except ModuleNotFoundError:
            try:
                from mflux.models.depth_pro.depth_pro import DepthPro  # type: ignore
            except ModuleNotFoundError:
                from mflux.depth.depth_pro import DepthPro  # type: ignore
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Initialize Depth Pro
        depth_pro = DepthPro()
        
        # Generate depth map
        depth_map = depth_pro.generate_depth_map(image)
        
        return depth_map
        
    except Exception as e:
        print(f"Error generating depth map: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_or_create_flux_depth(model, quantize=None, lora_paths=None, lora_scales=None, low_ram=False, base_model_override=None):
    """
    Create or retrieve a Flux instance for depth-controlled generation.
    """
    try:
        try:
            from mflux.controlnet.flux_controlnet import Flux1Controlnet
        except ModuleNotFoundError:
            from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
        
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
                model_name=base_model,
                alias=base_model,
                num_train_steps=1000,
                max_sequence_length=512,
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
            
        print(f"Creating Flux Depth with model_config={custom_config}, quantize={quantize}")
        
        flux = Flux1Controlnet(
            model_config=custom_config,
            quantize=quantize,
            local_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            control_type="depth"  # Specify depth control
        )
        
        return flux
        
    except Exception as e:
        print(f"Error creating Flux Depth instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_depth_gradio(
    prompt, control_image, model, base_model, seed, height, width, steps, guidance,
    depth_strength, lora_files, metadata, save_depth_map,
    *lora_scales, num_images=1, low_ram=False
):
    """
    Generate images using depth control.
    """
    try:
        # Parse inputs
        if not prompt or not prompt.strip():
            return [], None, "Error: Prompt is required", prompt
            
        if not control_image:
            return [], None, "Error: Control image is required", prompt
            
        # Process LoRA files
        from backend.lora_manager import process_lora_files
        lora_paths = process_lora_files(lora_files)
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        base_model_override = normalize_base_model_choice(base_model)
        
        # Get or create flux instance
        flux = get_or_create_flux_depth(
            model=model,
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
            low_ram=low_ram,
            base_model_override=base_model_override
        )
        
        if not flux:
            return [], None, "Error: Failed to initialize model", prompt
            
        # Generate depth map from control image
        depth_map = generate_depth_map(control_image)
        if depth_map is None:
            return [], None, "Error: Failed to generate depth map", prompt
            
        # Save depth map if requested
        depth_map_path = None
        if save_depth_map:
            timestamp = int(time.time())
            depth_filename = f"depth_map_{timestamp}.png"
            depth_map_path = os.path.join(OUTPUT_DIR, depth_filename)
            depth_map.save(depth_map_path)
            print(f"Depth map saved to {depth_map_path}")
            
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
                    
                print(f"Generating depth-controlled image {i+1}/{num_images} with seed: {current_seed}")
                
                # Parse dimensions
                if not height or height == 0:
                    height = 512
                if not width or width == 0:
                    width = 512
                    
                # Resize depth map if needed
                if depth_map.size != (width, height):
                    depth_map_resized = depth_map.resize((width, height), Image.LANCZOS)
                else:
                    depth_map_resized = depth_map
                    
                # Determine default guidance based on model
                if not guidance or guidance == "":
                    is_dev_model = "dev" in model
                    guidance_value = 3.5 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)
                    
                steps_int = 4 if not steps or steps.strip() == "" else int(steps)
                depth_strength_float = float(depth_strength) if depth_strength else 1.0
                
                # Generate the image
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    control_image=depth_map_resized,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                        control_strength=depth_strength_float,
                    ),
                )
                
                pil_image = generated.image
                
                # Save the image
                timestamp = int(time.time())
                filename = f"depth_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                # Save metadata if requested
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"depth_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "prompt": prompt,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "model": model,
                        "depth_strength": depth_strength_float,
                        "generation_time": str(time.ctime()),
                        "lora_files": lora_files,
                        "lora_scales": lora_scales_float
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated depth-controlled image saved to {output_path}")
                generated_images.append(pil_image)
                
            except Exception as e:
                print(f"Error generating depth image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        if generated_images:
            return generated_images, depth_map if save_depth_map else None, f"Generated {len(generated_images)} image(s)", prompt
        else:
            return [], None, "Error: Failed to generate any images", prompt
            
    except Exception as e:
        print(f"Error in depth generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], None, f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

# Expose this module as `depth_manager` for easy import in frontend components
import sys as _sys
depth_manager = _sys.modules[__name__]
