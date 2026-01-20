import gradio as gr

from backend.flux2_manager import generate_flux2_edit_gradio
from backend.lora_manager import get_lora_choices, update_lora_scales, MAX_LORAS
from backend.model_manager import get_flux2_models


def create_flux2_edit_tab():
    with gr.TabItem("Flux2 Klein Edit"):
        gr.Markdown(
            "### ✏️ Flux2 Klein Edit\n"
            "Edit using one or more reference images (guidance fixed at 1.0)."
        )

        prompt = gr.Textbox(label="Prompt", lines=3)
        image_inputs = gr.Files(label="Reference Images", file_types=["image"], type="filepath")

        flux2_choices = get_flux2_models()
        default_model = flux2_choices[0] if flux2_choices else "flux2-klein-4b"
        model = gr.Dropdown(
            choices=flux2_choices,
            label="Model",
            value=default_model,
            allow_custom_value=True,
        )

        with gr.Row():
            width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
            height = gr.Slider(512, 2048, value=1024, step=64, label="Height")

        steps = gr.Slider(1, 12, value=4, step=1, label="Inference Steps")
        guidance = gr.Number(label="Guidance Scale", value=1.0, interactive=False)
        seed = gr.Textbox(label="Seed", placeholder="Leave blank for random")
        num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")

        with gr.Accordion("LoRA Settings", open=False):
            lora_files = gr.Dropdown(
                choices=get_lora_choices(),
                multiselect=True,
                label="LoRA Files",
            )
            lora_scale_sliders = [
                gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    label=f"LoRA Weight {idx + 1}",
                    visible=False,
                )
                for idx in range(MAX_LORAS)
            ]
            lora_files.change(
                fn=update_lora_scales,
                inputs=[lora_files],
                outputs=lora_scale_sliders,
            )

        metadata = gr.Checkbox(label="Export Metadata", value=True)
        run_button = gr.Button("Run Flux2 Edit", variant="primary")

        output_gallery = gr.Gallery(label="Edited Images", columns=[2], height=400)
        status = gr.Textbox(label="Status", interactive=False)

        def _files_to_paths(files):
            if not files:
                return []
            return [f.name if hasattr(f, "name") else f for f in files]

        def _run_flux2_edit(
            prompt_val,
            image_inputs_val,
            model_val,
            seed_val,
            width_val,
            height_val,
            steps_val,
            lora_files_val,
            *scale_meta_and_num,
        ):
            if len(scale_meta_and_num) < 2:
                return [], "Internal error: missing metadata", prompt_val, []

            metadata_val = scale_meta_and_num[-2]
            num_images_val = scale_meta_and_num[-1]
            scale_vals = scale_meta_and_num[:-2]

            image_paths = _files_to_paths(image_inputs_val)
            images, status_text, updated_prompt = generate_flux2_edit_gradio(
                prompt_val,
                image_paths,
                model_val,
                seed_val,
                int(width_val),
                int(height_val),
                int(steps_val),
                lora_files_val,
                scale_vals,
                metadata_val,
                num_images=int(num_images_val),
            )
            return images, status_text, updated_prompt

        run_button.click(
            fn=_run_flux2_edit,
            inputs=[
                prompt,
                image_inputs,
                model,
                seed,
                width,
                height,
                steps,
                lora_files,
                *lora_scale_sliders,
                metadata,
                num_images,
            ],
            outputs=[output_gallery, status, prompt],
            concurrency_id="flux2_edit_queue",
        )

    return {"prompt": prompt, "gallery": output_gallery, "status": status}
