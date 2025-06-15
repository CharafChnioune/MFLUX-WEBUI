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

def parse_concept_attention_prompt(prompt):
    """
    Parse a prompt with concept attention syntax.
    Example: "A dog<mid> and a cat<high> in a garden<low>"
    Returns: parsed prompt with attention weights
    """
    import re
    
    # Patterns for different attention levels
    high_pattern = r'<high>|<h>'
    mid_pattern = r'<mid>|<m>|<medium>'
    low_pattern = r'<low>|<l>'
    
    # Replace patterns with attention weights
    # High attention: 1.5x weight
    prompt = re.sub(r'(\w+)\s*' + high_pattern, r'(\1:1.5)', prompt, flags=re.IGNORECASE)
    # Medium attention: 1.2x weight
    prompt = re.sub(r'(\w+)\s*' + mid_pattern, r'(\1:1.2)', prompt, flags=re.IGNORECASE)
    # Low attention: 0.8x weight
    prompt = re.sub(r'(\w+)\s*' + low_pattern, r'(\1:0.8)', prompt, flags=re.IGNORECASE)
    
    return prompt

def get_or_create_flux_concept(model, quantize=None, lora_paths=None, lora_scales=None, low_ram=False):
    """
    Create or retrieve a Flux instance with concept attention support.
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
            
        print(f"Creating Flux with Concept Attention, model_config={custom_config}, quantize={quantize}")
        
        flux = Flux1(
            model_config=custom_config,
            quantize=quantize,
            local_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            enable_concept_attention=True  # Enable concept attention
        )
        
        return flux
        
    except Exception as e:
        print(f"Error creating Flux Concept Attention instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_concept_attention_gradio(
    prompt, concept_examples, model, seed, height, width, steps, guidance,
    attention_strength, lora_files, metadata, *lora_scales, num_images=1, low_ram=False
):
    """
    Generate images using concept attention.
    """
    try:
        # Parse inputs
        if not prompt or not prompt.strip():
            return [], "Error: Prompt is required", prompt
            
        # Process LoRA files
        from backend.lora_manager import process_lora_files
        lora_paths, lora_scales_float = process_lora_files(lora_files, lora_scales)
        
        # Get or create flux instance
        flux = get_or_create_flux_concept(
            model=model,
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
            low_ram=low_ram
        )
        
        if not flux:
            return [], "Error: Failed to initialize model", prompt
            
        # Process concept examples
        if concept_examples and concept_examples.strip():
            # Apply concept attention based on examples
            # Format: "concept1:high, concept2:low, concept3:mid"
            concepts = [c.strip() for c in concept_examples.split(',')]
            
            # Build enhanced prompt with concept attention
            enhanced_prompt = prompt
            for concept in concepts:
                if ':' in concept:
                    concept_name, attention_level = concept.split(':', 1)
                    concept_name = concept_name.strip()
                    attention_level = attention_level.strip().lower()
                    
                    # Apply attention tags to concepts in the prompt
                    if concept_name.lower() in prompt.lower():
                        if attention_level in ['high', 'h']:
                            enhanced_prompt = enhanced_prompt.replace(concept_name, f"{concept_name}<high>")
                        elif attention_level in ['mid', 'medium', 'm']:
                            enhanced_prompt = enhanced_prompt.replace(concept_name, f"{concept_name}<mid>")
                        elif attention_level in ['low', 'l']:
                            enhanced_prompt = enhanced_prompt.replace(concept_name, f"{concept_name}<low>")
                            
            # Parse the enhanced prompt with attention tags
            final_prompt = parse_concept_attention_prompt(enhanced_prompt)
        else:
            # Use prompt as-is, but still parse any existing attention tags
            final_prompt = parse_concept_attention_prompt(prompt)
            
        print(f"Final prompt with concept attention: {final_prompt}")
        
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
                    
                print(f"Generating concept attention image {i+1}/{num_images} with seed: {current_seed}")
                
                # Parse dimensions
                if not height or height == 0:
                    height = 512
                if not width or width == 0:
                    width = 512
                    
                # Determine default guidance based on model
                if not guidance or guidance == "":
                    is_dev_model = "dev" in model
                    guidance_value = 3.5 if is_dev_model else 0.0
                else:
                    guidance_value = float(guidance)
                    
                steps_int = 4 if not steps or steps.strip() == "" else int(steps)
                attention_strength_float = float(attention_strength) if attention_strength else 1.0
                
                # Generate the image with concept attention
                generated = flux.generate_image(
                    seed=current_seed,
                    prompt=final_prompt,
                    config=Config(
                        num_inference_steps=steps_int,
                        height=height,
                        width=width,
                        guidance=guidance_value,
                        attention_strength=attention_strength_float,
                    ),
                )
                
                pil_image = generated.image
                
                # Save the image
                timestamp = int(time.time())
                filename = f"concept_attention_{timestamp}_{current_seed}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                pil_image.save(output_path)
                
                # Save metadata if requested
                if metadata:
                    metadata_path = os.path.join(OUTPUT_DIR, f"concept_attention_{timestamp}_{current_seed}.json")
                    metadata_dict = {
                        "original_prompt": prompt,
                        "final_prompt": final_prompt,
                        "concept_examples": concept_examples,
                        "seed": current_seed,
                        "steps": steps_int,
                        "guidance": guidance_value,
                        "width": width,
                        "height": height,
                        "model": model,
                        "attention_strength": attention_strength_float,
                        "generation_time": str(time.ctime()),
                        "lora_files": lora_files,
                        "lora_scales": lora_scales_float
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_dict, f, indent=2)
                        
                print(f"Generated concept attention image saved to {output_path}")
                generated_images.append(pil_image)
                
            except Exception as e:
                print(f"Error generating concept attention image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        if generated_images:
            return generated_images, f"Generated {len(generated_images)} image(s)", final_prompt
        else:
            return [], "Error: Failed to generate any images", prompt
            
    except Exception as e:
        print(f"Error in concept attention generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", prompt
        
    finally:
        # Cleanup
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

# Expose this module as `concept_attention_manager`
import sys as _sys
concept_attention_manager = _sys.modules[__name__]
