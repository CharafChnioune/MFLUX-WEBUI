import gradio as gr
from PIL import Image
import numpy as np
from backend.model_manager import get_updated_models, get_base_model_choices
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    MAX_LORAS,
    process_lora_files,
    refresh_lora_choices
)
from backend.prompts_manager import enhance_prompt
from backend.flux_manager import (
    generate_image_controlnet_gradio,
    get_random_seed
)
from backend.post_processing import (
    update_dimensions_on_image_change,
    update_dimensions_on_scale_change,
    update_height_with_aspect_ratio,
    update_width_with_aspect_ratio,
    scale_dimensions
)
from frontend.components.llmsettings import create_llm_settings
import time
import os

def update_dimensions_on_image_change(image):
    """Update width and height when image is uploaded"""
    if image is not None:
        width, height = image.size
        return width, height, width, height
    return None, None, None, None

def update_dimensions_on_scale_change(scale_factor, original_width, original_height):
    """Update width and height when scale factor changes"""
    if original_width is not None and original_height is not None:
        width = int(original_width * scale_factor)
        height = int(original_height * scale_factor)
        return width, height
    return None, None

def update_steps_based_on_model(model_name):
    if "schnell" in model_name.lower():
        return "4"
    elif "dev" in model_name.lower():
        return "20"
    return ""

def create_controlnet_tab():
    """Create the ControlNet tab interface"""
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=2)
            with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                llm_components = create_llm_settings(tab_name="controlnet", parent_accordion=llm_section)
            with gr.Row():
                enhance_prompt_btn = gr.Button("Enhance prompt with LLM")
            
            control_image = gr.Image(label="Control Image", type="pil")
            
            def debug_enhance_prompt(p, t, m1, m2, sp, img):
                print("[DEBUG] Enhance prompt called")
                print(f"[DEBUG] Image uploaded: {img is not None}")
                if img is not None:
                    print(f"[DEBUG] Image type: {type(img)}")
                    print(f"[DEBUG] Image size: {img.size}")
                    # Convert PIL image to base64 for MLX-VLM
                    if t == "MLX-VLM":
                        import io
                        import base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        img = img_str
                return enhance_prompt(p, t, m1, m2, sp, img)
            
            enhance_prompt_btn.click(
                fn=debug_enhance_prompt,
                inputs=[
                    prompt,
                    llm_components[0],  # llm_type
                    llm_components[1],  # ollama_model
                    llm_components[4],  # mlx_model
                    llm_components[2],  # system_prompt
                    control_image
                ],
                outputs=prompt
            )

            model_cn = gr.Dropdown(
                choices=get_updated_models(),
                label="Model",
                value="schnell-4-bit"
            )

            # Vervang de canny image door een tekstmelding die alleen zichtbaar is als save_canny is ingeschakeld
            canny_notification = gr.Textbox(label="Canny Edge Detection", visible=False)

            # Store original dimensions
            original_width_cn = gr.State()
            original_height_cn = gr.State()

            # Width and height inputs
            width_cn = gr.Number(label="Width", value=0, precision=0)
            height_cn = gr.Number(label="Height", value=0, precision=0)

            # Scale factor slider
            scale_factor_cn = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Scale Factor"
            )

            # Update dimensions when image is uploaded
            control_image.change(
                fn=update_dimensions_on_image_change,
                inputs=[control_image],
                outputs=[width_cn, height_cn, original_width_cn, original_height_cn]
            )

            # Update dimensions when scale factor changes
            scale_factor_cn.change(
                fn=update_dimensions_on_scale_change,
                inputs=[scale_factor_cn, original_width_cn, original_height_cn],
                outputs=[width_cn, height_cn]
            )

            guidance_cn = gr.Number(label="Guidance Scale", value=3.5, visible=False)
            controlnet_strength = gr.Number(label="ControlNet Strength", value=0.5)
            
            steps_cn = gr.Textbox(label="Inference Steps (optional)", value="4")
            model_cn.change(fn=update_steps_based_on_model, inputs=[model_cn], outputs=[steps_cn])

            with gr.Row():
                seed_cn = gr.Textbox(
                    label="Seed (optional)",
                    value="",
                    scale=9,
                    container=True
                )
                random_seed_cn = gr.Button(
                    "🎲",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )
            random_seed_cn.click(fn=get_random_seed, outputs=[seed_cn])

            with gr.Row():
                lora_files_cn = gr.Dropdown(
                    choices=get_lora_choices(),
                    label="Select LoRA Files",
                    multiselect=True,
                    value=[],
                    interactive=True,
                    scale=9
                )
                refresh_lora_cn = gr.Button(
                    "🔄",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )

            refresh_lora_cn.click(
                fn=refresh_lora_choices,
                inputs=[],
                outputs=[lora_files_cn]
            )

            lora_scales_cn = [
                gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) 
                for _ in range(MAX_LORAS)
            ]
            
            lora_files_cn.change(
                fn=update_lora_scales,
                inputs=[lora_files_cn],
                outputs=lora_scales_cn
            )

            num_images_cn = gr.Number(label="Number of Images", value=1, precision=0)

            metadata_cn = gr.Checkbox(label="Export Metadata as JSON", value=False)
            save_canny = gr.Checkbox(label="Save Canny Edge Detection Image", value=False)
            low_ram_cn = gr.Checkbox(label="Low RAM Mode (reduces memory usage)", value=False)
            
            # MFLUX v0.9.0 Features
            with gr.Accordion("⚡ MFLUX v0.9.0 Features", open=False):
                base_model_cn = gr.Dropdown(
                    choices=["Auto"] + get_base_model_choices(),
                    label="Base Model (for third-party HuggingFace models)",
                    value="Auto"
                )
                prompt_file_cn = gr.Textbox(
                    label="Prompt File Path (--prompt-file)",
                    placeholder="Path to text/JSON file with prompts",
                    value=""
                )
                config_from_metadata_cn = gr.Textbox(
                    label="Config from Metadata (--config-from-metadata)", 
                    placeholder="Path to image file to extract config from",
                    value=""
                )
                stepwise_output_dir_cn = gr.Textbox(
                    label="Stepwise Output Directory (--stepwise-image-output-dir)",
                    placeholder="Directory to save intermediate steps",
                    value=""
                )
                vae_tiling_cn = gr.Checkbox(
                    label="VAE Tiling (--vae-tiling)",
                    value=False,
                    info="Enable tiling for large images to reduce memory usage"
                )
                vae_tiling_split_cn = gr.Dropdown(
                    choices=["horizontal", "vertical"],
                    label="VAE Tiling Split (--vae-tiling-split)",
                    value="horizontal",
                    visible=False
                )
                
                vae_tiling_cn.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[vae_tiling_cn],
                    outputs=[vae_tiling_split_cn]
                )
            
            generate_button_cn = gr.Button("Generate Image", variant='primary')

        with gr.Column():
            output_gallery_cn = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery_controlnet",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            output_message_cn = gr.Textbox(label="Saved Image Filenames")

        def generate_with_loras(*args):
            prompt, control_image, model, base_model, seed, height, width, steps, guidance, controlnet_strength, lora_files, metadata, save_canny, low_ram = args[:14]
            prompt_file, config_from_metadata, stepwise_output_dir, vae_tiling, vae_tiling_split = args[14:19]
            lora_scales_and_num = args[19:]
            num_images = lora_scales_and_num[-1]
            lora_scales = lora_scales_and_num[:-1]
            
            # Process LoRA files and get valid paths
            valid_loras = process_lora_files(lora_files) if lora_files else None
            if valid_loras:
                # Only use scales for valid LoRAs
                valid_scales = lora_scales[:len(valid_loras)]
            else:
                valid_scales = []
            
            # Generate images
            result = generate_image_controlnet_gradio(
                prompt,
                control_image,
                model,
                base_model if base_model not in ("Auto", "None", "", None) else None,
                seed,
                height,
                width,
                steps,
                guidance,
                controlnet_strength,
                valid_loras,
                metadata,
                save_canny,
                prompt_file,
                config_from_metadata,
                stepwise_output_dir,
                vae_tiling,
                vae_tiling_split,
                *valid_scales,
                num_images=num_images,
                low_ram=low_ram
            )
            
            # Update canny image visibility based on save_canny checkbox
            # Controlnet retourneert nu [images], filename, prompt 
            # zonder expliciet het canny image terug te geven
            # daarom maken we een afbeelding in het geval save_canny ingeschakeld is
            if save_canny and len(result[0]) > 0:
                # Als er controlnet is gebruikt, toon een melding dat de canny-afbeelding apart is opgeslagen
                timestamp = int(time.time())
                canny_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", f"canny_{timestamp}_{seed}.png")
                canny_notification = f"Canny edge image opgeslagen in output map als 'canny_{timestamp}_{seed}.png'"
                # Zet de zichtbaarheid van het textvak direct in
                canny_notification = gr.update(value=canny_notification, visible=True)
            else:
                # Verberg het textvak en maak het leeg
                canny_notification = gr.update(value="", visible=False)
            
            # Return results
            return (
                result[0],             # Generated images
                result[1],             # Filenames
                result[2],             # Original prompt
                canny_notification     # Bericht over canny image
            )

        generate_button_cn.click(
            fn=generate_with_loras,
            inputs=[
                prompt, control_image, model_cn, base_model_cn, seed_cn, height_cn, width_cn, 
                steps_cn, guidance_cn, controlnet_strength, lora_files_cn, 
                metadata_cn, save_canny, low_ram_cn,
                prompt_file_cn, config_from_metadata_cn, stepwise_output_dir_cn, 
                vae_tiling_cn, vae_tiling_split_cn,
                *lora_scales_cn, num_images_cn
            ],
            outputs=[output_gallery_cn, output_message_cn, prompt, canny_notification]
        )

        return {
            'prompt': prompt,
            'model': model_cn,
            'lora_files': lora_files_cn,
            'output_gallery': output_gallery_cn,
            'output_message': output_message_cn,
            'ollama_components': llm_components,
            'control_image': control_image,
            'canny_notification': canny_notification
        } 
