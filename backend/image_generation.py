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
    Voorbeeld: Guidance visible maken indien 'dev' in de modelnaam zit.
    """
    return gr.update(visible="dev" in model)

def simple_generate_image(prompt, model, image_format, lora_files, ollama_model, system_prompt, *lora_scales_and_num_images):
    """
    Simpele interface om een afbeelding te genereren zonder al te veel parameters.
    """
    num_images = lora_scales_and_num_images[-1]
    lora_scales_list = lora_scales_and_num_images[:-1]

    print(f"\n--- Generating image (Easy) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Image Format: {image_format}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales_list}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    width, height = parse_image_format(image_format)

    valid_loras = process_lora_files(lora_files)
    lora_scales = lora_scales_list[:len(valid_loras)] if valid_loras else None

    steps = 20 if "dev" in model else 4

    flux = get_or_create_flux(model, None, None, valid_loras, lora_scales)

    images, filenames, seeds = generate_image_batch(
        flux=flux,
        prompt=prompt,
        seed=None,      
        steps=steps,
        height=height,
        width=width,
        guidance=7.5,
        num_images=int(num_images)
    )

    clear_flux_cache()
    force_mlx_cleanup()

    # Maak een informatieve output string
    output_info = []
    for filename, seed in zip(filenames, seeds):
        output_info.append(f"File: {filename} (Seed: {seed})")

    return images, "\n".join(output_info), prompt

def parse_image_format(image_format):
    import re
    match = re.search(r"\((\d+)x(\d+)\)", image_format)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return (width, height)
    return (512, 512)

def generate_image_gradio(prompt, model, base_model, seed, width, height, steps, guidance, lora_files, metadata, ollama_model, system_prompt, *lora_scales_and_num_images):
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
