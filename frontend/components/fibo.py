import gradio as gr

from backend.fibo_manager import generate_fibo_gradio
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    MAX_LORAS,
    refresh_lora_choices,
)


def create_fibo_tab():
    with gr.TabItem("FIBO"):
        gr.Markdown(
            """
            # FIBO
            Generate images with the FIBO model. Provide a JSON prompt or let the FIBO VLM expand plain text.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_mode = gr.Radio(
                    label="Prompt Mode",
                    choices=["Text", "JSON"],
                    value="Text",
                )
                prompt_text = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the scene...",
                    lines=3,
                )
                json_prompt = gr.Textbox(
                    label="JSON Prompt",
                    placeholder="Paste structured JSON prompt...",
                    lines=8,
                    visible=False,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Optional negative prompt...",
                    lines=2,
                )

                with gr.Row():
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
                    placeholder="e.g. briaai/Fibo-mlx-4bit or /path/to/fibo",
                )
                quantize_bits = gr.Dropdown(
                    label="Quantization",
                    choices=["None", "8", "6", "5", "4", "3"],
                    value="None",
                )

                with gr.Row():
                    width = gr.Number(label="Width", value=1024, precision=0)
                    height = gr.Number(label="Height", value=1024, precision=0)

                with gr.Row():
                    steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="Guidance")

                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=["flow_match_euler_discrete", "linear"],
                    value="flow_match_euler_discrete",
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

                with gr.Accordion("FIBO VLM Settings", open=False):
                    use_vlm = gr.Checkbox(
                        label="Use FIBO VLM to expand text prompts",
                        value=True,
                    )
                    vlm_quantize = gr.Dropdown(
                        label="VLM Quantization",
                        choices=["None", "8", "6", "5", "4", "3"],
                        value="None",
                    )
                    vlm_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    vlm_temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                    vlm_max_tokens = gr.Number(label="Max Tokens", value=4096, precision=0)
                    vlm_seed = gr.Number(label="VLM Seed", value=-1, precision=0)

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
                used_json_prompt = gr.Textbox(
                    label="JSON Prompt Used",
                    lines=8,
                    interactive=False,
                )

        def _toggle_prompt_mode(mode):
            return gr.update(visible=mode == "Text"), gr.update(visible=mode == "JSON")

        prompt_mode.change(
            fn=_toggle_prompt_mode,
            inputs=[prompt_mode],
            outputs=[prompt_text, json_prompt],
        )

        def _run_fibo(*args):
            (
                prompt_val,
                json_prompt_val,
                mode_val,
                negative_val,
                model_path_val,
                quantize_val,
                seed_val,
                width_val,
                height_val,
                steps_val,
                guidance_val,
                scheduler_val,
                lora_files_val,
                metadata_val,
                init_image_val,
                image_strength_val,
                use_vlm_val,
                vlm_quant_val,
                vlm_top_p_val,
                vlm_temp_val,
                vlm_max_tokens_val,
                vlm_seed_val,
                *lora_scales_and_num,
            ) = args

            num_images_val = lora_scales_and_num[-1]
            lora_scales_val = lora_scales_and_num[:-1]

            vlm_seed_final = None
            if isinstance(vlm_seed_val, (int, float)) and vlm_seed_val >= 0:
                vlm_seed_final = int(vlm_seed_val)

            return generate_fibo_gradio(
                prompt_text=prompt_val,
                json_prompt_text=json_prompt_val,
                prompt_mode=mode_val,
                negative_prompt=negative_val,
                model_path=model_path_val,
                quantize_bits=quantize_val,
                seed=seed_val,
                width=width_val,
                height=height_val,
                steps=steps_val,
                guidance=guidance_val,
                scheduler=scheduler_val,
                lora_files=lora_files_val,
                lora_scales=lora_scales_val,
                metadata=metadata_val,
                init_image=init_image_val,
                image_strength=image_strength_val,
                use_vlm=use_vlm_val,
                vlm_quantize=vlm_quant_val,
                vlm_top_p=vlm_top_p_val,
                vlm_temperature=vlm_temp_val,
                vlm_max_tokens=vlm_max_tokens_val,
                vlm_seed=vlm_seed_final,
                num_images=int(num_images_val) if num_images_val else 1,
            )

        generate_btn.click(
            fn=_run_fibo,
            inputs=[
                prompt_text,
                json_prompt,
                prompt_mode,
                negative_prompt,
                model_path,
                quantize_bits,
                seed,
                width,
                height,
                steps,
                guidance,
                scheduler,
                lora_files,
                metadata,
                init_image,
                image_strength,
                use_vlm,
                vlm_quantize,
                vlm_top_p,
                vlm_temperature,
                vlm_max_tokens,
                vlm_seed,
                *lora_scales,
                num_images,
            ],
            outputs=[output_gallery, status, used_json_prompt],
        )

    return {}
