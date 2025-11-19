import gradio as gr

from backend.qwen_manager import generate_qwen_image_gradio
from backend.lora_manager import get_lora_choices, update_lora_scales, MAX_LORAS
from backend.model_manager import get_base_model_choices

# Layout doc references:
# - gradiodocs/guides-controlling-layout/controlling_layout.md
# - gradiodocs/docs-image/image.md


def create_qwen_image_tab():
    with gr.TabItem("Qwen Image"):
        gr.Markdown(
            "### 🖼️ Qwen Image Generation\n"
            "Wraps `mflux-generate-qwen` with full prompt, multilingual, and LoRA controls."
        )

        prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image you want")
        negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Optional content to avoid")

        model = gr.Dropdown(
            choices=["qwen-image"],
            value="qwen-image",
            label="Model",
            allow_custom_value=True,
        )
        base_model = gr.Dropdown(
            choices=["Auto"] + get_base_model_choices(),
            value="Auto",
            label="Base Model (--base-model)",
            info="Needed when targeting third-party Hugging Face repos.",
        )
        quantize = gr.Dropdown(
            choices=["None", "16", "8", "4"],
            value="None",
            label="Quantize Bits",
        )

        with gr.Row():
            width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
            height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
        steps = gr.Slider(4, 50, value=20, step=1, label="Inference Steps")
        guidance = gr.Slider(0.0, 10.0, value=4.5, step=0.1, label="Guidance")
        seed = gr.Textbox(label="Seed", placeholder="Leave blank for random")
        num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images", interactive=True)

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

        init_image = gr.Image(label="Optional Init Image", type="filepath", interactive=True)

        with gr.Row():
            metadata = gr.Checkbox(label="Export Metadata", value=True)
            run_button = gr.Button("Generate with Qwen", variant="primary")

        output_gallery = gr.Gallery(
            label="Generated Images",
            columns=[2],
            height=400,
        )
        status = gr.Textbox(label="Status", interactive=False)

        def _run_qwen_image(
            prompt_val,
            negative_prompt_val,
            model_val,
            base_model_val,
            seed_val,
            width_val,
            height_val,
            steps_val,
            guidance_val,
            quantize_val,
            lora_files_val,
            *scale_metadata_and_inputs,
        ):
            metadata_val = scale_metadata_and_inputs[-3]
            init_image_val = scale_metadata_and_inputs[-2]
            num_images_val = scale_metadata_and_inputs[-1]
            scale_vals = scale_metadata_and_inputs[:-3]
            return generate_qwen_image_gradio(
                prompt_val,
                negative_prompt_val,
                model_val,
                base_model_val,
                seed_val,
                width_val,
                height_val,
                steps_val,
                guidance_val,
                quantize_val,
                lora_files_val,
                scale_vals,
                metadata_val,
                init_image_val,
                num_images_val,
            )

        run_button.click(
            fn=_run_qwen_image,
            inputs=[
                prompt,
                negative_prompt,
                model,
                base_model,
                seed,
                width,
                height,
                steps,
                guidance,
                quantize,
                lora_files,
                *lora_scale_sliders,
                metadata,
                init_image,
                num_images,
            ],
            outputs=[output_gallery, status, prompt],
            concurrency_id="qwen_queue",
        )

        return {
            "prompt": prompt,
            "gallery": output_gallery,
            "status": status,
        }
