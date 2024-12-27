import gradio as gr
from backend.model_manager import get_updated_models
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    MAX_LORAS,
    process_lora_files
)
from backend.prompts_manager import enhance_prompt
from backend.flux_manager import (
    get_random_seed,
    generate_image_gradio
)
from backend.post_processing import (
    update_height_with_aspect_ratio,
    update_width_with_aspect_ratio
)
from frontend.components.llmsettings import create_llm_settings

def create_advanced_generate_tab():
    """Create the Advanced Generate tab interface"""
    with gr.Row():
        with gr.Column():
            prompt_advanced = gr.Textbox(label="Prompt", lines=2)
            with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section_advanced:
                llm_components_advanced = create_llm_settings(tab_name="advanced", parent_accordion=llm_section_advanced)
            with gr.Row():
                enhance_prompt_btn = gr.Button("Enhance prompt with LLM")
            
            enhance_prompt_btn.click(
                fn=lambda p, t, m1, m2, sp: enhance_prompt(p, t, m1, m2, sp),
                inputs=[
                    prompt_advanced,
                    llm_components_advanced[0],
                    llm_components_advanced[1],
                    llm_components_advanced[4],
                    llm_components_advanced[2]
                ],
                outputs=prompt_advanced
            )

            model = gr.Dropdown(
                choices=get_updated_models(),
                label="Model",
                value="schnell-4-bit"
            )

            with gr.Row():
                width = gr.Number(label="Width", value=576, precision=0)
                height = gr.Number(label="Height", value=1024, precision=0)
            
            def update_steps_based_on_model(model_name):
                if "schnell" in model_name.lower():
                    return "4"
                elif "dev" in model_name.lower():
                    return "20"
                return ""
            
            steps = gr.Textbox(label="Inference Steps (optional)", value="4")
            model.change(fn=update_steps_based_on_model, inputs=[model], outputs=[steps])
            
            with gr.Row():
                seed = gr.Textbox(
                    label="Seed (optional)", 
                    value="",
                    scale=9,
                    container=True
                )
                random_seed = gr.Button(
                    "🎲",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )
            random_seed.click(fn=get_random_seed, outputs=[seed])

            with gr.Row():
                lora_files = gr.Dropdown(
                    choices=get_lora_choices(),
                    label="Select LoRA Files",
                    multiselect=True,
                    allow_custom_value=True,
                    value=[],
                    interactive=True,
                    scale=9
                )
                refresh_lora = gr.Button(
                    "🔄",
                    variant='tool',
                    size='sm',
                    scale=1,
                    min_width=30,
                    elem_classes='refresh-button'
                )

            lora_scales = [
                gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) 
                for _ in range(MAX_LORAS)
            ]
            
            lora_files.change(
                fn=update_lora_scales,
                inputs=[lora_files],
                outputs=lora_scales
            )

            num_images = gr.Number(label="Number of Images", value=1, precision=0)
            
            guidance = gr.Number(label="Guidance Scale", value=3.5, visible=False)
            metadata = gr.Checkbox(label="Export Metadata as JSON", value=False)
            generate_button = gr.Button("Generate Image", variant='primary')

        with gr.Column():
            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery_advanced",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            output_filename = gr.Textbox(label="Saved Image Filenames")

        def generate_with_loras(*args):
            prompt, model, seed, width, height, steps, guidance, lora_files, metadata, llm_type, llm_model, *lora_scales_and_num = args
            num_images = lora_scales_and_num[-1]
            lora_scales = lora_scales_and_num[:-1]
            
            valid_loras = process_lora_files(lora_files) if lora_files else None
            if valid_loras:
                valid_scales = lora_scales[:len(valid_loras)]
            else:
                valid_scales = []
            
            return generate_image_gradio(
                prompt,
                model,
                seed,
                width,
                height,
                steps,
                guidance,
                valid_loras,
                metadata,
                llm_model if llm_type == "Ollama" else None,
                None,
                *valid_scales,
                num_images=num_images
            )

        generate_button.click(
            fn=generate_with_loras,
            inputs=[
                prompt_advanced, model, seed, width, height, steps, guidance, lora_files,
                metadata, llm_components_advanced[0], llm_components_advanced[1],
                *lora_scales, num_images
            ],
            outputs=[output_gallery, output_filename, prompt_advanced]
        )

        return {
            'prompt': prompt_advanced,
            'model': model,
            'lora_files': lora_files,
            'output_gallery': output_gallery,
            'output_filename': output_filename,
            'llm_components': llm_components_advanced
        } 