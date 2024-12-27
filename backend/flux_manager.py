import os
import gc
import time
import random
import traceback
import mlx.core as mx
from PIL import Image
from mflux.config.config import Config, ConfigControlnet
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from backend.lora_manager import process_lora_files
from backend.ollama_manager import enhance_prompt
from backend.prompts_manager import enhance_prompt_with_mlx
from backend.mlx_utils import force_mlx_cleanup, print_memory_usage

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

def get_or_create_flux(model, config=None, init_image=None, lora_paths=None, lora_scales=None, is_controlnet=False):
    """
    Create or retrieve a Flux1 instance.
    """
    try:
        base_model = model.replace("-8-bit", "").replace("-4-bit", "")
        
        try:
            from backend.model_manager import get_custom_model_config
            custom_config = get_custom_model_config(base_model)
            if base_model in ["dev", "schnell", "dev-8-bit", "dev-4-bit", "schnell-8-bit", "schnell-4-bit"]:
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
        else:
            quantize = None

        FluxClass = Flux1Controlnet if is_controlnet else Flux1
        flux = FluxClass(
            model_config=custom_config,
            quantize=quantize,
            local_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        return flux

    except Exception as e:
        print(f"Error creating Flux instance: {str(e)}")
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
    for i in range(num_images):
        current_seed = seed if seed is not None else int(time.time()) + i
        output_filename = f"generated_{int(time.time())}_{i}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        image = flux.generate_image(
            seed=current_seed,
            prompt=prompt,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            ),
        )
        image.save(output_path)
        images.append(image)
        filenames.append(output_filename)
    return images, filenames

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
    *lora_scales, num_images=1
):
    """
    Generate images with the given parameters.
    """
    try:
        start_time = time.time()
        
        width, height = parse_image_format(image_format)
        
        base_model = model.replace("-4-bit", "").replace("-8-bit", "")
        if "schnell" in base_model:
            steps = 4
        else:
            steps = 20
        
        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_files,
            lora_scales=lora_scales if lora_scales else None
        )
        
        images = []
        filenames = []
        
        for i in range(num_images):
            current_seed = random.randint(0, 2**32 - 1)
            output_filename = f"output_{int(time.time())}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            image = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=7.5,
                ),
            )
            
            image.image.save(output_path)
            images.append(image.image)
            filenames.append(output_filename)
        
        del flux
        gc.collect()
        force_mlx_cleanup()
        
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.2f} seconds")
        
        return images, "\n".join(filenames), prompt
        
    except Exception as e:
        print(f"Error in image generation: {str(e)}")
        return [], "", prompt
    finally:
        force_mlx_cleanup()
        gc.collect()

def generate_image_gradio(
    prompt, model, seed, width, height, steps, guidance, 
    lora_files, metadata, ollama_model=None, system_prompt=None,
    *lora_scales, num_images=1
):
    """
    Generate an image with the given parameters.
    """
    print(f"\n--- Generating image ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    try:
        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_files,
            lora_scales=lora_scales if lora_scales else None
        )

        if not flux:
            return [], None, prompt

        images = []
        filenames = []
        for i in range(int(num_images)):
            current_seed = int(seed) if seed and str(seed).strip() else int(time.time()) + i
            output_filename = f"generated_{int(time.time())}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            image = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=int(steps) if steps and str(steps).strip() else 20,
                    height=int(height - (height % 16)),
                    width=int(width - (width % 16)),
                    guidance=guidance if guidance is not None else 7.5,
                ),
            )

            image.image.save(output_path)
            images.append(image.image)
            filenames.append(output_filename)

        del flux
        gc.collect()
        
        force_mlx_cleanup()
        print_memory_usage("After generation")
        
        return images, "\n".join(filenames), prompt

    except Exception as e:
        error_message = f"Error generating images: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return [], None, prompt
    finally:
        force_mlx_cleanup()
        gc.collect()

def generate_image_controlnet_gradio(
    prompt, control_image, model, seed, height, width, steps, guidance,
    controlnet_strength, lora_files, metadata, save_canny,
    *lora_scales, num_images=1
):
    """
    Generate an image with ControlNet.
    """
    print(f"\n--- Generating image (ControlNet) ---")
    print(f"Received parameters:")
    print(f"- prompt: {prompt}")
    print(f"- model: {model}")
    print(f"- seed: {seed}")
    print(f"- height: {height}")
    print(f"- width: {width}") 
    print(f"- steps: {steps}")
    print(f"- guidance: {guidance}")
    print(f"- controlnet_strength: {controlnet_strength}")
    print(f"- lora_files: {lora_files}")
    print(f"- lora_scales: {lora_scales}")
    print(f"- save_canny: {save_canny}")
    print(f"- num_images: {num_images}")

    print_memory_usage("Before generation")
    start_time = time.time()
    generated_images = []
    filenames = []
    canny_image_to_return = None
    try:
        seed = None if seed == "" else int(seed)
        steps = None if steps == "" else int(steps)
        if steps is None:
            steps = 4 if "schnell" in model else 14

        flux = get_or_create_flux(
            model=model,
            lora_paths=lora_files,
            lora_scales=lora_scales if lora_scales else None,
            is_controlnet=True
        )

        timestamp = int(time.time())
        control_image_path = os.path.join(OUTPUT_DIR, f"control_image_{timestamp}.png")
        control_image.save(control_image_path)

        for i in range(int(num_images)):
            current_seed = seed if seed is not None else int(time.time()) + i
            output_filename = f"generated_controlnet_{int(time.time())}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            generated_image = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                controlnet_image_path=control_image_path,
                config=ConfigControlnet(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    controlnet_strength=controlnet_strength,
                ),
                output=output_path,
                controlnet_save_canny=save_canny
            )

            generated_images.append(generated_image.image)
            filenames.append(output_filename)

        print_memory_usage("After generating images")

        if os.path.exists(control_image_path):
            os.remove(control_image_path)

        print(f"Generation completed in {time.time() - start_time:.2f}s")
        return generated_images, "\n".join(filenames), prompt, canny_image_to_return

    except Exception as e:
        print(f"\nError in ControlNet generation: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        return [], "", prompt, None

    finally:
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

def generate_image_i2i_gradio(
    prompt, input_image, model, seed, height, width, steps, guidance,
    strength, lora_files, metadata, *lora_scales, num_images=1
):
    """
    Generate an image from another image (img2img).
    """
    if input_image is None:
        print("Error: No initial image provided")
        return [], "", prompt

    print(f"\n--- Generating image (img2img) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Initial Image Strength: {strength}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    start_time = time.time()

    try:
        seed = None if seed == "" else int(seed)
        if not steps or steps.strip() == "":
            base_model = model.replace("-4-bit", "").replace("-8-bit", "")
            if "schnell" in base_model:
                steps = 4
            elif "dev" in base_model:
                steps = 20
            else:
                steps = 20
        else:
            steps = int(steps)

        flux = get_or_create_flux(
            model=model,
            init_image=input_image,
            lora_paths=lora_files,
            lora_scales=lora_scales if lora_scales else None
        )

        if not flux:
            return [], "Error: Could not create Flux instance", prompt

        images = []
        filenames = []
        for i in range(int(num_images)):
            current_seed = seed if seed is not None else int(time.time()) + i
            output_filename = f"generated_{int(time.time())}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            image = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    init_image_strength=strength,
                ),
            )
            
            pil_image = image.image
            pil_image.save(output_path)
            images.append(pil_image)
            filenames.append(output_filename)

        del flux
        gc.collect()
        force_mlx_cleanup()

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        return images, "\n".join(filenames), prompt

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        traceback.print_exc()
        return [], "", prompt

    finally:
        force_mlx_cleanup()
        gc.collect()
