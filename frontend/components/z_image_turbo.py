import gradio as gr

from backend.z_image_manager import generate_z_image_gradio
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    MAX_LORAS,
    refresh_lora_choices,
)


def create_z_image_turbo_tab():
    with gr.TabItem("Z-Image Turbo"):
        gr.Markdown(
            """
            # Z-Image Turbo
            Fast text-to-image generation with the Z-Image Turbo model. Guidance is fixed to 0.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the scene...",
                    lines=3,
                )
                init_image = gr.Image(
                    label="Init Image (optional)",
                    type="filepath",
                    height=240,
                )
                image_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Image Strength",
                )

                model_path = gr.Textbox(
                    label="Model Path or HF Repo (optional)",
                    placeholder="e.g. filipstrand/Z-Image-Turbo-mflux-4bit",
                )
                quantize_bits = gr.Dropdown(
                    label="Quantization",
                    choices=["None", "8", "6", "5", "4", "3"],
                    value="None",
                )

                with gr.Row():
                    width = gr.Number(label="Width", value=1024, precision=0)
                    height = gr.Number(label="Height", value=1024, precision=0)

                steps = gr.Slider(1, 20, value=9, step=1, label="Steps")

                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=["linear", "flow_match_euler_discrete"],
                    value="linear",
                    allow_custom_value=True,
                )

                with gr.Row():
                    seed = gr.Textbox(label="Seed", placeholder="Leave blank for random")
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Variations")

                with gr.Group():
                    with gr.Row():
                        lora_files = gr.Dropdown(
                            choices=get_lora_choices(),
                            label="LoRA Files (optional)",
                            multiselect=True,
                            allow_custom_value=True,
                            value=[],
                            interactive=True,
                            scale=9,
                        )
                        refresh_lora_btn = gr.Button(
                            "Refresh",
                            variant="secondary",
                            size="sm",
                            scale=1,
                        )

                    refresh_lora_btn.click(
                        fn=refresh_lora_choices,
                        inputs=[],
                        outputs=[lora_files],
                    )

                    lora_scales = [
                        gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0)
                        for _ in range(MAX_LORAS)
                    ]
                    lora_files.change(
                        fn=update_lora_scales,
                        inputs=[lora_files],
                        outputs=lora_scales,
                    )

                metadata = gr.Checkbox(label="Save Metadata", value=True)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=2,
                    object_fit="contain",
                    height=520,
                )
                status = gr.Textbox(label="Status", interactive=False)

        def _run_z_image(*args):
            (
                prompt_val,
                model_path_val,
                quantize_val,
                seed_val,
                width_val,
                height_val,
                steps_val,
                scheduler_val,
                lora_files_val,
                metadata_val,
                init_image_val,
                image_strength_val,
                *lora_scales_and_num,
            ) = args

            num_images_val = lora_scales_and_num[-1]
            lora_scales_val = lora_scales_and_num[:-1]

            return generate_z_image_gradio(
                prompt=prompt_val,
                model_path=model_path_val,
                quantize_bits=quantize_val,
                seed=seed_val,
                width=width_val,
                height=height_val,
                steps=steps_val,
                scheduler=scheduler_val,
                lora_files=lora_files_val,
                lora_scales=lora_scales_val,
                metadata=metadata_val,
                init_image=init_image_val,
                image_strength=image_strength_val,
                num_images=int(num_images_val) if num_images_val else 1,
            )

        generate_btn.click(
            fn=_run_z_image,
            inputs=[
                prompt,
                model_path,
                quantize_bits,
                seed,
                width,
                height,
                steps,
                scheduler,
                lora_files,
                metadata,
                init_image,
                image_strength,
                *lora_scales,
                num_images,
            ],
            outputs=[output_gallery, status, prompt],
        )

    return {}
