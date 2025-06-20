import gradio as gr
from backend.model_manager import get_updated_models, update_guidance_visibility
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
            
            # Stel de standaard waardes in op basis van het standaard model
            default_model = "schnell-4-bit"
            steps = gr.Textbox(
                label="Inference Steps (optional)", 
                value=update_steps_based_on_model(default_model)
            )
            
            # Voeg guidance toe net onder steps
            guidance = gr.Number(
                label=update_guidance_visibility(default_model)["label"],
                value=3.5, 
                visible=True
            )
            
            # Update beide steps en guidance op basis van het model
            model.change(
                fn=lambda model_name: [
                    update_steps_based_on_model(model_name),
                    update_guidance_visibility(model_name)
                ],
                inputs=[model],
                outputs=[steps, guidance]
            )

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
            
            # Add click handler for the refresh button
            refresh_lora.click(
                fn=lambda: gr.update(choices=get_lora_choices()),
                outputs=[lora_files]
            )

            # Options for generating multiple images
            with gr.Row():
                num_images = gr.Number(label="Number of Images", value=1, precision=0)
                auto_seeds = gr.Checkbox(label="Auto-generate random seeds", value=False)
            
            # Additional options
            with gr.Accordion("Additional Options", open=False):
                with gr.Row():
                    metadata = gr.Checkbox(label="Export Metadata as JSON", value=False)
                    low_ram = gr.Checkbox(label="Low RAM Mode (reduces memory usage)", value=False)
                    
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
            prompt, model, seed, width, height, steps, guidance, lora_files, metadata, low_ram, auto_seeds, llm_type, llm_model, *lora_scales_and_num = args
            num_images = lora_scales_and_num[-1]
            lora_scales = lora_scales_and_num[:-1]
            
            valid_loras = process_lora_files(lora_files) if lora_files else None
            if valid_loras:
                valid_scales = lora_scales[:len(valid_loras)]
            else:
                valid_scales = []
            
            # Convert auto_seeds to number or None
            auto_seeds_value = 8 if auto_seeds else None
            
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
                num_images=num_images,
                low_ram=low_ram,
                auto_seeds=auto_seeds_value
            )

        generate_button.click(
            fn=generate_with_loras,
            inputs=[
                prompt_advanced, model, seed, width, height, steps, guidance, lora_files,
                metadata, low_ram, auto_seeds, llm_components_advanced[0], llm_components_advanced[1],
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