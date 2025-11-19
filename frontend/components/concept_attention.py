import gradio as gr

from backend.concept_attention_manager import (
    generate_text_concept_heatmap,
    generate_image_concept_heatmap,
)
from backend.lora_manager import get_lora_choices, update_lora_scales, MAX_LORAS
from backend.model_manager import get_base_model_choices, get_updated_models

# Layout doc reference: gradiodocs/docs-tab, gradiodocs/docs-row


def _lora_controls():
    lora_files = gr.Dropdown(
        choices=get_lora_choices(),
        multiselect=True,
        label="LoRA Files",
    )
    lora_scales = [
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
        outputs=lora_scales,
    )
    return lora_files, lora_scales


def create_concept_attention_tab():
    with gr.TabItem("Concept Attention"):
        gr.Markdown(
            "### 🧠 Concept Attention\n"
            "Visualize how prompts influence spatial attention using Flux concept analyzers."
        )

        with gr.Tabs():
            with gr.TabItem("Text Concept Heatmap"):
                prompt = gr.Textbox(label="Prompt", lines=3)
                concept = gr.Textbox(label="Concept Text", placeholder="e.g. luminous aura")
                model = gr.Dropdown(
                    choices=get_updated_models(),
                    value="dev",
                    label="Model",
                )
                base_model = gr.Dropdown(
                    choices=["Auto"] + get_base_model_choices(),
                    value="Auto",
                    label="Base Model (--base-model)",
                )
                seed = gr.Number(label="Seed", value=42, precision=0)
                with gr.Row():
                    width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                    height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                steps = gr.Slider(4, 50, value=20, step=1, label="Steps")
                guidance = gr.Slider(0.0, 10.0, value=4.0, step=0.1, label="Guidance")
                quantize = gr.Dropdown(
                    choices=["None", "8", "4", "3"],
                    value="None",
                    label="Quantize Bits",
                )
                heatmap_layers = gr.Textbox(
                    label="Heatmap Layer Indices",
                    placeholder="e.g. 4,8,12",
                    info="Leave blank for defaults.",
                )
                heatmap_timesteps = gr.Textbox(
                    label="Heatmap Timesteps",
                    placeholder="e.g. 5,10",
                )
                metadata = gr.Checkbox(label="Export Metadata", value=True)

                lora_files, lora_scales = _lora_controls()

                def _run_text_concept(
                    prompt_val,
                    concept_val,
                    model_val,
                    base_model_val,
                    seed_val,
                    width_val,
                    height_val,
                    steps_val,
                    guidance_val,
                    quantize_val,
                    heatmap_layers_val,
                    heatmap_timesteps_val,
                    lora_files_val,
                    *scale_and_metadata,
                ):
                    metadata_val = scale_and_metadata[-1]
                    scale_vals = scale_and_metadata[:-1]
                    return generate_text_concept_heatmap(
                        prompt_val,
                        concept_val,
                        model_val,
                        base_model_val,
                        seed_val,
                        width_val,
                        height_val,
                        steps_val,
                        guidance_val,
                        quantize_val,
                        heatmap_layers_val,
                        heatmap_timesteps_val,
                        lora_files_val,
                        scale_vals,
                        metadata_val,
                    )

                render_gallery = gr.Gallery(label="Render", columns=[2], height=300)
                heatmap_gallery = gr.Gallery(label="Heatmap", columns=[2], height=300)
                status = gr.Textbox(label="Status", interactive=False)
                run_btn = gr.Button("Run Concept Analysis", variant="primary")

                run_btn.click(
                    fn=_run_text_concept,
                    inputs=[
                        prompt,
                        concept,
                        model,
                        base_model,
                        seed,
                        width,
                        height,
                        steps,
                        guidance,
                        quantize,
                        heatmap_layers,
                        heatmap_timesteps,
                        lora_files,
                        *lora_scales,
                        metadata,
                    ],
                    outputs=[render_gallery, heatmap_gallery, status],
                )

            with gr.TabItem("Image-Guided Heatmap"):
                prompt_img = gr.Textbox(label="Prompt", lines=3)
                concept_img = gr.Textbox(label="Concept Text")
                ref_image = gr.Image(label="Reference Image", type="filepath")
                model_img = gr.Dropdown(
                    choices=get_updated_models(),
                    value="dev",
                    label="Model",
                )
                base_model_img = gr.Dropdown(
                    choices=["Auto"] + get_base_model_choices(),
                    value="Auto",
                    label="Base Model (--base-model)",
                )
                seed_img = gr.Number(label="Seed", value=101, precision=0)
                with gr.Row():
                    width_img = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                    height_img = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                steps_img = gr.Slider(4, 50, value=20, step=1, label="Steps")
                guidance_img = gr.Slider(0.0, 10.0, value=4.0, step=0.1, label="Guidance")
                quantize_img = gr.Dropdown(
                    choices=["None", "8", "4", "3"],
                    value="None",
                    label="Quantize Bits",
                )
                heatmap_layers_img = gr.Textbox(label="Heatmap Layer Indices")
                heatmap_timesteps_img = gr.Textbox(label="Heatmap Timesteps")
                metadata_img = gr.Checkbox(label="Export Metadata", value=True)

                lora_files_img, lora_scales_img = _lora_controls()

                def _run_image_concept(
                    prompt_val,
                    concept_val,
                    ref_image_val,
                    model_val,
                    base_model_val,
                    seed_val,
                    width_val,
                    height_val,
                    steps_val,
                    guidance_val,
                    quantize_val,
                    heatmap_layers_val,
                    heatmap_timesteps_val,
                    lora_files_val,
                    *scale_and_metadata,
                ):
                    metadata_val = scale_and_metadata[-1]
                    scale_vals = scale_and_metadata[:-1]
                    return generate_image_concept_heatmap(
                        prompt_val,
                        concept_val,
                        ref_image_val,
                        model_val,
                        base_model_val,
                        seed_val,
                        width_val,
                        height_val,
                        steps_val,
                        guidance_val,
                        quantize_val,
                        heatmap_layers_val,
                        heatmap_timesteps_val,
                        lora_files_val,
                        scale_vals,
                        metadata_val,
                    )

                render_gallery_img = gr.Gallery(label="Render", columns=[2], height=300)
                heatmap_gallery_img = gr.Gallery(label="Heatmap", columns=[2], height=300)
                status_img = gr.Textbox(label="Status", interactive=False)
                run_btn_img = gr.Button("Run Image-Guided Analysis", variant="primary")

                run_btn_img.click(
                    fn=_run_image_concept,
                    inputs=[
                        prompt_img,
                        concept_img,
                        ref_image,
                        model_img,
                        base_model_img,
                        seed_img,
                        width_img,
                        height_img,
                        steps_img,
                        guidance_img,
                        quantize_img,
                        heatmap_layers_img,
                        heatmap_timesteps_img,
                        lora_files_img,
                        *lora_scales_img,
                        metadata_img,
                    ],
                    outputs=[render_gallery_img, heatmap_gallery_img, status_img],
                )

    return {}
