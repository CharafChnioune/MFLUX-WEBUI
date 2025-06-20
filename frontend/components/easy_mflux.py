import gradio as gr
from backend.model_manager import get_updated_models
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    refresh_lora_choices,
    MAX_LORAS,
    process_lora_files
)
from backend.prompts_manager import enhance_prompt
from backend.flux_manager import simple_generate_image
from frontend.components.llmsettings import create_llm_settings

def create_easy_mflux_tab():
    """Create the MFLUX Easy tab interface"""
    with gr.Row():
        with gr.Column():
            prompt_simple = gr.Textbox(label="Prompt", lines=2)
            with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section_simple:
                llm_components_simple = create_llm_settings(tab_name="easy", parent_accordion=llm_section_simple)
            with gr.Row():
                enhance_prompt_btn = gr.Button("Enhance prompt with LLM")
            
            model_simple = gr.Dropdown(
                choices=get_updated_models(),
                label="Model",
                value="schnell-4-bit"
            )
            
            image_format = gr.Dropdown(
                choices=[
                    "Portrait (576x1024)",
                    "Landscape (1024x576)",
                    "Background (1920x1080)",
                    "Square (1024x1024)",
                    "Poster (1080x1920)",
                    "Wide Screen (2560x1440)",
                    "Ultra Wide Screen (3440x1440)",
                    "Banner (728x90)"
                ],
                label="Image Format",
                value="Portrait (576x1024)"
            )
            with gr.Row():
                lora_files_simple = gr.Dropdown(
                    choices=get_lora_choices(),
                    label="Select LoRA Files",
                    multiselect=True,
                    value=[],
                    interactive=True,
                    scale=9
                )
                refresh_lora_simple = gr.Button(
                    "🔄",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )
            refresh_lora_simple.click(
                fn=refresh_lora_choices,
                inputs=[],
                outputs=[lora_files_simple]
            )

            lora_scales_simple = [
                gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) 
                for _ in range(MAX_LORAS)
            ]
            
            lora_files_simple.change(
                fn=update_lora_scales,
                inputs=[lora_files_simple],
                outputs=lora_scales_simple
            )

            with gr.Row():
                num_images_simple = gr.Number(label="Number of Images", value=1, precision=0)
                low_ram_simple = gr.Checkbox(label="Low RAM Mode", value=False)

            generate_button_simple = gr.Button("Generate Image", variant='primary')
        
        with gr.Column():
            output_gallery_simple = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery_simple",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            output_filename_simple = gr.Textbox(label="Saved Image Filenames")

        enhance_prompt_btn.click(
            fn=enhance_prompt,
            inputs=[
                prompt_simple,
                llm_components_simple[0],  # llm_type
                llm_components_simple[1],  # ollama_model
                llm_components_simple[4],  # mlx_model
                llm_components_simple[2]   # system_prompt
            ],
            outputs=prompt_simple
        )
        
        def generate_with_loras(*args):
            prompt, model, image_format, lora_files, llm_type, llm_model, system_prompt, low_ram, *lora_scales_and_num = args
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
            return simple_generate_image(
                prompt,
                model,
                image_format,
                valid_loras,
                llm_model if llm_type == "Ollama" else None,
                system_prompt,
                *valid_scales,
                num_images=num_images,
                low_ram=low_ram
            )
        
        generate_button_simple.click(
            fn=generate_with_loras,
            inputs=[
                prompt_simple,
                model_simple,
                image_format,
                lora_files_simple,
                llm_components_simple[0],  # llm_type
                llm_components_simple[1] if llm_components_simple[0] == "Ollama" else llm_components_simple[4],  # correct model based on type
                llm_components_simple[2],  # system_prompt
                low_ram_simple,
                *lora_scales_simple,
                num_images_simple
            ],
            outputs=[output_gallery_simple, output_filename_simple, prompt_simple]
        )

        return {
            'prompt': prompt_simple,
            'model': model_simple,
            'lora_files': lora_files_simple,
            'output_gallery': output_gallery_simple,
            'output_filename': output_filename_simple,
            'ollama_components': llm_components_simple
        } 