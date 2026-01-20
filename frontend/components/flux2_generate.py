import gradio as gr

from backend.flux2_manager import generate_flux2_image_gradio
from backend.lora_manager import get_lora_choices, update_lora_scales, MAX_LORAS
from backend.model_manager import get_flux2_models


def create_flux2_generate_tab():
    with gr.TabItem("Flux2 Klein"):
        gr.Markdown(
            "### ⚡ Flux2 Klein (Text-to-Image)\n"
            "FLUX.2 uses guidance=1.0 (fixed) and is optimized for fast 4-step generation."
        )

        prompt = gr.Textbox(label="Prompt", lines=3)

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
        run_button = gr.Button("Generate Flux2 Image", variant="primary")

        output_gallery = gr.Gallery(label="Generated Images", columns=[2], height=400)
        status = gr.Textbox(label="Status", interactive=False)

        def _run_flux2(
            prompt_val,
            model_val,
            seed_val,
            width_val,
            height_val,
            steps_val,
            lora_files_val,
            *scale_and_meta,
        ):
            if len(scale_and_meta) < 2:
                return [], "Internal error: missing metadata", prompt_val

            metadata_val = scale_and_meta[-2]
            num_images_val = scale_and_meta[-1]
            scale_vals = scale_and_meta[:-2]

            images, status_text, updated_prompt = generate_flux2_image_gradio(
                prompt_val,
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
            fn=_run_flux2,
            inputs=[
                prompt,
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
            concurrency_id="flux2_generate_queue",
        )

    return {"prompt": prompt, "gallery": output_gallery, "status": status}
