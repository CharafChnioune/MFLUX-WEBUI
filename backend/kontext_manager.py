import os
import gc
import time
import json
from pathlib import Path
from PIL import Image
from mflux.config.config import Config
from backend.flux_manager import get_or_create_flux, get_random_seed, calculate_dimensions_with_scale, OUTPUT_DIR
from backend.lora_manager import process_lora_files
from backend.mlx_utils import force_mlx_cleanup

def generate_image_kontext_gradio(
    prompt, reference_image, model, seed, width, height, steps, guidance,
    lora_files, metadata, *lora_scales, num_images=1, low_ram=False
):
    """
    Generate an image with FLUX.1 Kontext for character consistency and local editing.
    """
    if reference_image is None:
        return [], "Error: Reference image is required for Kontext generation", prompt
        
    try:
        print(f"\n--- Generating with FLUX.1 Kontext ---")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Reference image: {reference_image}")
        
        # Process LoRA files and scales (skip in-context helpers to avoid double-stacking)
        lora_paths = process_lora_files(lora_files)
        lora_scales_float = process_lora_files(lora_files, lora_scales)
        if lora_paths:
            filtered_paths = []
            filtered_scales = []
            for idx, path in enumerate(lora_paths):
                if "in_context_lora" in Path(path).name.lower():
                    continue
                filtered_paths.append(path)
                if lora_scales_float and idx < len(lora_scales_float):
                    filtered_scales.append(lora_scales_float[idx])
            lora_paths = filtered_paths
            lora_scales_float = filtered_scales if filtered_scales else None
        
        # Calculate dimensions with scale factor support
        ref_img = Image.open(reference_image)
        final_width, final_height = calculate_dimensions_with_scale(width, height, ref_img)
        
        # Use dev-kontext model configuration
        kontext_model = model.replace("dev", "dev-kontext") if "dev" in model else "dev-kontext"
        
        results = []
        filenames = []
        
        for i in range(num_images):
            try:
                current_seed = seed + i if seed is not None else get_random_seed()
                print(f"Generating image {i+1}/{num_images} with seed {current_seed}")
                
                # Get Flux instance for Kontext
                flux = get_or_create_flux(
                    kontext_model,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales_float,
                    low_ram=low_ram
                )
                
                # Kontext-specific guidance (higher for dev models)
                is_dev_model = "dev" in model
                guidance_value = float(guidance) if guidance else (7.0 if is_dev_model else 0.0)
                steps_int = int(steps) if steps else 25
                
                # Generate with Kontext capabilities
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=prompt,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=final_height,
                        width=final_width,
                        guidance=guidance_value,
                    ),
                    # Kontext-specific parameters
                    reference_image=ref_img
                )
                
                pil_image = generated.image
                
                # Process and save results
                timestamp = int(time.time())
                filename = f"kontext_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                results.append(pil_image)
                filenames.append(filename)
                
                # Save metadata if requested
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"kontext_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "prompt": prompt,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": final_width,
                        "height": final_height,
                        "model": kontext_model,
                        "reference_image": reference_image,
                        "generation_time": str(time.ctime()),
                        "lora_files": lora_files,
                        "lora_scales": lora_scales_float,
                        "kontext_enabled": True
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated Kontext image {i+1}/{num_images} saved to {output_path}")
                
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        if results:
            success_msg = f"Generated {len(results)} Kontext image(s): {', '.join(filenames)}"
            return results, success_msg, prompt
        else:
            return [], "Error: Failed to generate any images", prompt
            
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
