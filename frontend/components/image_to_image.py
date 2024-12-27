import gradio as gr
from PIL import Image
from backend.model_manager import get_updated_models
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    MAX_LORAS,
    process_lora_files
)
from backend.prompts_manager import enhance_prompt
from backend.flux_manager import (
    generate_image_i2i_gradio,
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

def update_steps_based_on_model(model_name):
    if "schnell" in model_name.lower():
        return "4"
    elif "dev" in model_name.lower():
        return "20"
    return ""

def create_image_to_image_tab():
    """Create the Image to Image tab interface"""
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=2)
            with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                llm_components = create_llm_settings(tab_name="image-to-image", parent_accordion=llm_section)
            with gr.Row():
                enhance_prompt_btn = gr.Button("Enhance prompt with LLM")
            
            input_image = gr.Image(label="Input Image", type="pil")
            
            def debug_enhance_prompt(p, t, m1, m2, sp, img):
                print("[DEBUG] Enhance prompt called")
                print(f"[DEBUG] Image uploaded: {img is not None}")
                if img is not None:
                    print(f"[DEBUG] Image type: {type(img)}")
                    print(f"[DEBUG] Image size: {img.size}")
                    # Pass image to both MLX-VLM and Ollama
                    if t == "MLX-VLM":
                        import io
                        import base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        img = img_str
                return enhance_prompt(p, t, m1, m2, sp, img, tab_name="image-to-image")
            
            enhance_prompt_btn.click(
                fn=debug_enhance_prompt,
                inputs=[
                    prompt,
                    llm_components[0],  # llm_type
                    llm_components[1],  # ollama_model
                    llm_components[4],  # mlx_model
                    llm_components[2],  # system_prompt
                    input_image  # Add input image
                ],
                outputs=prompt
            )

            model_i2i = gr.Dropdown(
                choices=get_updated_models(),
                label="Model",
                value="schnell-4-bit"
            )

            scale_factor_i2i = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Scale Factor (%)"
            )

            original_width_i2i = gr.State()
            original_height_i2i = gr.State()

            width_i2i = gr.Number(label="Width", precision=0)
            height_i2i = gr.Number(label="Height", precision=0)

            input_image.change(
                fn=update_dimensions_on_image_change,
                inputs=[input_image],
                outputs=[width_i2i, height_i2i, original_width_i2i, original_height_i2i]
            )

            scale_factor_i2i.change(
                fn=update_dimensions_on_scale_change,
                inputs=[scale_factor_i2i, original_width_i2i, original_height_i2i],
                outputs=[width_i2i, height_i2i]
            )

            width_i2i.change(
                fn=update_height_with_aspect_ratio,
                inputs=[width_i2i, original_width_i2i, original_height_i2i],
                outputs=[height_i2i]
            )

            height_i2i.change(
                fn=update_width_with_aspect_ratio,
                inputs=[height_i2i, original_width_i2i, original_height_i2i],
                outputs=[width_i2i]
            )

            guidance_i2i = gr.Number(label="Guidance Scale", value=3.5, visible=False)
            strength_i2i = gr.Number(label="Strength", value=0.5)
            
            steps_i2i = gr.Textbox(label="Inference Steps (optional)", value="4")
            model_i2i.change(fn=update_steps_based_on_model, inputs=[model_i2i], outputs=[steps_i2i])

            with gr.Row():
                seed_i2i = gr.Textbox(
                    label="Seed (optional)",
                    value="",
                    scale=9,
                    container=True
                )
                random_seed_i2i = gr.Button(
                    "🎲",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )
            random_seed_i2i.click(fn=get_random_seed, outputs=[seed_i2i])

            with gr.Row():
                lora_files_i2i = gr.Dropdown(
                    choices=get_lora_choices(),
                    label="Select LoRA Files",
                    multiselect=True,
                    allow_custom_value=True,
                    value=[],
                    interactive=True,
                    scale=9
                )
                refresh_lora_i2i = gr.Button(
                    "🔄",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )

            lora_scales_i2i = [
                gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) 
                for _ in range(MAX_LORAS)
            ]
            
            lora_files_i2i.change(
                fn=update_lora_scales,
                inputs=[lora_files_i2i],
                outputs=lora_scales_i2i
            )

            num_images_i2i = gr.Number(label="Number of Images", value=1, precision=0)

            metadata_i2i = gr.Checkbox(label="Export Metadata as JSON", value=False)
            generate_button_i2i = gr.Button("Generate Image", variant='primary')

        with gr.Column():
            output_gallery_i2i = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery_i2i",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            output_message_i2i = gr.Textbox(label="Saved Image Filenames")

        def generate_with_loras(*args):
            prompt, input_image, model, seed, height, width, steps, guidance, strength, lora_files, metadata, *lora_scales_and_num = args
            num_images = lora_scales_and_num[-1]
            lora_scales = lora_scales_and_num[:-1]
            
            # Process LoRA files and get valid paths
            valid_loras = process_lora_files(lora_files) if lora_files else None
            if valid_loras:
                # Only use scales for valid LoRAs
                valid_scales = lora_scales[:len(valid_loras)]
            else:
                valid_scales = []
            
            # Pass arguments in correct order to avoid confusion
            return generate_image_i2i_gradio(
                prompt,
                input_image,
                model,
                seed,
                height,
                width,
                steps,
                guidance,
                strength,
                valid_loras,
                metadata,
                *valid_scales,
                num_images=num_images
            )

        generate_button_i2i.click(
            fn=generate_with_loras,
            inputs=[
                prompt, input_image, model_i2i, seed_i2i, height_i2i, width_i2i, steps_i2i,
                guidance_i2i, strength_i2i, lora_files_i2i, metadata_i2i,
                *lora_scales_i2i, num_images_i2i
            ],
            outputs=[output_gallery_i2i, output_message_i2i, prompt]
        )

        return {
            'prompt': prompt,
            'model': model_i2i,
            'lora_files': lora_files_i2i,
            'output_gallery': output_gallery_i2i,
            'output_message': output_message_i2i,
            'ollama_components': llm_components,
            'input_image': input_image
        } 