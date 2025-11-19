import gradio as gr
from PIL import Image
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
    generate_image_in_context_lora_gradio,
    get_random_seed
)
from backend.post_processing import (
    update_dimensions_on_image_change,
    update_dimensions_on_scale_change,
    update_height_with_aspect_ratio,
    update_width_with_aspect_ratio,
    scale_dimensions,
    update_guidance_visibility
)
from frontend.components.llmsettings import create_llm_settings

def update_steps_based_on_model(model_name):
    if "schnell" in model_name.lower():
        return "4"
    elif "dev" in model_name.lower():
        return "25"  # Higher steps recommended for In-Context LoRA
    return ""

def create_in_context_lora_tab():
    """Create the In-Context LoRA tab interface"""
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe both the reference image and the image you want to generate. Use markers like [IMAGE1]/[IMAGE2], [LEFT]/[RIGHT], etc.")
            with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                llm_components = create_llm_settings(tab_name="in-context-lora", parent_accordion=llm_section)
            with gr.Row():
                enhance_prompt_btn = gr.Button("Enhance prompt with LLM")
            
            reference_image = gr.Image(label="Reference Image", type="pil")
            
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
                return enhance_prompt(p, t, m1, m2, sp, img, tab_name="in-context-lora")
            
            enhance_prompt_btn.click(
                fn=debug_enhance_prompt,
                inputs=[
                    prompt,
                    llm_components[0],  # llm_type
                    llm_components[1],  # ollama_model
                    llm_components[4],  # mlx_model
                    llm_components[2],  # system_prompt
                    reference_image
                ],
                outputs=prompt
            )

            model_icl = gr.Dropdown(
                choices=get_updated_models(),
                label="Model",
                value="dev-4-bit"  # Default to dev for In-Context LoRA
            )

            # Store original dimensions
            original_width_icl = gr.State()
            original_height_icl = gr.State()

            # Width and height inputs
            with gr.Row():
                width_icl = gr.Number(label="Width", value=1024, precision=0)
                height_icl = gr.Number(label="Height", value=1024, precision=0)

            guidance_icl = gr.Number(label="Guidance Scale", value=7.0)  # Higher guidance for In-Context LoRA
            
            steps_icl = gr.Textbox(label="Inference Steps", value="25")  # More steps for better results
            model_icl.change(fn=update_steps_based_on_model, inputs=[model_icl], outputs=[steps_icl])

            with gr.Row():
                seed_icl = gr.Textbox(
                    label="Seed (optional)",
                    value="",
                    scale=9,
                    container=True
                )
                random_seed_icl = gr.Button(
                    "🎲",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )
            random_seed_icl.click(fn=get_random_seed, outputs=[seed_icl])

            # LoRA style selection
            lora_style = gr.Dropdown(
                choices=[
                    "couple", "storyboard", "font", "home", 
                    "illustration", "portrait", "ppt", 
                    "sandstorm", "sparklers", "identity"
                ],
                label="In-Context LoRA Style",
                value="identity"
            )

            with gr.Row():
                lora_files_icl = gr.Dropdown(
                    choices=get_lora_choices(),
                    label="Additional LoRA Files (optional)",
                    multiselect=True,
                    allow_custom_value=True,
                    value=[],
                    interactive=True,
                    scale=9
                )
                refresh_lora_icl = gr.Button(
                    "🔄",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )

            refresh_lora_icl.click(
                fn=refresh_lora_choices,
                inputs=[],
                outputs=[lora_files_icl]
            )

            lora_scales_icl = [
                gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) 
                for _ in range(MAX_LORAS)
            ]
            
            lora_files_icl.change(
                fn=update_lora_scales,
                inputs=[lora_files_icl],
                outputs=lora_scales_icl
            )

            num_images_icl = gr.Number(label="Number of Images", value=1, precision=0)

            metadata_icl = gr.Checkbox(label="Export Metadata as JSON", value=False)
            low_ram_icl = gr.Checkbox(label="Low RAM Mode (reduces memory usage)", value=False)
            
            # MFLUX v0.9.0 Features
            with gr.Accordion("⚡ MFLUX v0.9.0 Features", open=False):
                base_model_icl = gr.Dropdown(
                    choices=["Auto"] + get_base_model_choices(),
                    label="Base Model (for third-party HuggingFace models)",
                    value="Auto"
                )
                prompt_file_icl = gr.Textbox(
                    label="Prompt File Path (--prompt-file)",
                    placeholder="Path to text/JSON file with prompts",
                    value=""
                )
                config_from_metadata_icl = gr.Textbox(
                    label="Config from Metadata (--config-from-metadata)", 
                    placeholder="Path to image file to extract config from",
                    value=""
                )
                stepwise_output_dir_icl = gr.Textbox(
                    label="Stepwise Output Directory (--stepwise-image-output-dir)",
                    placeholder="Directory to save intermediate steps",
                    value=""
                )
                vae_tiling_icl = gr.Checkbox(
                    label="VAE Tiling (--vae-tiling)",
                    value=False,
                    info="Enable tiling for large images to reduce memory usage"
                )
                vae_tiling_split_icl = gr.Dropdown(
                    choices=["horizontal", "vertical"],
                    label="VAE Tiling Split (--vae-tiling-split)",
                    value="horizontal",
                    visible=False
                )
                
                vae_tiling_icl.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[vae_tiling_icl],
                    outputs=[vae_tiling_split_icl]
                )
            
            generate_button_icl = gr.Button("Generate Image", variant='primary')

        with gr.Column():
            output_gallery_icl = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery_in_context_lora",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            output_message_icl = gr.Textbox(label="Saved Image Filenames")
            
            with gr.Accordion("How to use In-Context LoRA", open=True):
                gr.Markdown("""
                ## Tips for using In-Context LoRA
                
                1. **Reference Image**: Upload a clear image that represents the style or content you want to mimic.
                
                2. **Prompt Structure**: In your prompt, describe both the reference image and what you want to generate:
                   - Use markers like `[IMAGE1]`/`[IMAGE2]`, `[LEFT]`/`[RIGHT]`, or `[REFERENCE]`/`[OUTPUT]`
                   - Example: "A set of two product photos. [LEFT] A minimalist white coffee mug on a light gray background; [RIGHT] A minimalist white teapot on a light gray background."
                
                3. **Styles**: Different LoRA styles produce different results. Experiment with:
                   - `identity`: Best for general style transfer
                   - `portrait`: For people and characters
                   - `illustration`: For artistic and stylized images
                   
                4. **Settings**:
                   - Higher guidance (7.0-9.0) gives stronger prompt adherence
                   - More steps (25-30) gives higher quality results
                """)

        def generate_with_loras(*args):
            prompt, reference_image, model, base_model, seed, height, width, steps, guidance, lora_style, lora_files, metadata, low_ram = args[:13]
            prompt_file, config_from_metadata, stepwise_output_dir, vae_tiling, vae_tiling_split = args[13:18]
            lora_scales_and_num = args[18:]
            num_images = lora_scales_and_num[-1]
            lora_scales = lora_scales_and_num[:-1]
            
            # Process LoRA files and get valid paths
            valid_loras = process_lora_files(lora_files) if lora_files else None
            if valid_loras:
                # Only use scales for valid LoRAs
                valid_scales = lora_scales[:len(valid_loras)]
            else:
                valid_scales = []
            
            # Generate images with In-Context LoRA
            return generate_image_in_context_lora_gradio(
                prompt,
                reference_image,
                model,
                base_model if base_model not in ("Auto", "None", "", None) else None,
                seed,
                height,
                width,
                steps,
                guidance,
                lora_style,
                valid_loras,
                metadata,
                prompt_file,
                config_from_metadata,
                stepwise_output_dir,
                vae_tiling,
                vae_tiling_split,
                *valid_scales,
                num_images=num_images,
                low_ram=low_ram
            )

        generate_button_icl.click(
            fn=generate_with_loras,
            inputs=[
                prompt, reference_image, model_icl, base_model_icl, seed_icl, height_icl, width_icl, steps_icl,
                guidance_icl, lora_style, lora_files_icl, metadata_icl, low_ram_icl,
                prompt_file_icl, config_from_metadata_icl, stepwise_output_dir_icl, 
                vae_tiling_icl, vae_tiling_split_icl,
                *lora_scales_icl, num_images_icl
            ],
            outputs=[output_gallery_icl, output_message_icl, prompt]
        )

        return {
            'prompt': prompt,
            'model': model_icl,
            'lora_files': lora_files_icl,
            'output_gallery': output_gallery_icl,
            'output_message': output_message_icl,
            'ollama_components': llm_components,
            'reference_image': reference_image
        } 
