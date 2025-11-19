import os
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from backend.qwen_manager import generate_qwen_edit_gradio
from backend.lora_manager import get_lora_choices, update_lora_scales, MAX_LORAS


def create_qwen_edit_tab():
    with gr.TabItem("Qwen Image Edit"):
        gr.Markdown(
            "### ✏️ Qwen Image Edit\n"
            "Bridge multiple init images into a single edited output."
        )

        prompt = gr.Textbox(label="Prompt", lines=3)
        negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)

        image_inputs = gr.Files(label="Reference Images", file_types=["image"], type="filepath")

        preview_image = gr.Image(
            label="Preview / Crop (optional)",
            type="pil",
            tool="select",
        )

        with gr.Row():
            width = gr.Slider(512, 2048, step=64, value=1024, label="Width")
            height = gr.Slider(512, 2048, step=64, value=1024, label="Height")
        steps = gr.Slider(4, 40, value=25, step=1, label="Inference Steps")
        guidance = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="Guidance")
        seed = gr.Textbox(label="Seed", placeholder="Leave blank for random")
        quantize = gr.Dropdown(
            choices=["None", "16", "8", "4"],
            value="None",
            label="Quantize Bits",
        )

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
        with gr.Row():
            run_button = gr.Button("Run Qwen Edit", variant="primary")
            use_last_as_input_btn = gr.Button("Use last output as input", variant="secondary")

        output_gallery = gr.Gallery(label="Edited Images", columns=[2], height=400)
        status = gr.Textbox(label="Status", interactive=False)

        # Keep track of last generated output file paths for chaining
        last_output_paths = gr.State([])

        def _files_to_paths(files):
            if not files:
                return []
            return [f.name if hasattr(f, "name") else f for f in files]

        def _first_image(files):
            paths = _files_to_paths(files)
            if not paths:
                return None
            try:
                return Image.open(paths[0]).convert("RGB")
            except Exception:
                return None

        def _run_qwen_edit(
            prompt_val,
            negative_prompt_val,
            image_inputs_val,
            seed_val,
            width_val,
            height_val,
            steps_val,
            guidance_val,
            quantize_val,
            lora_files_val,
            *scale_metadata_and_state,
        ):
            # Last three entries: metadata, crop image, last_paths (state)
            if len(scale_metadata_and_state) < 3:
                return [], "Internal error: missing metadata/state", prompt_val, []

            metadata_val = scale_metadata_and_state[-3]
            crop_image_val = scale_metadata_and_state[-2]
            last_paths_val = scale_metadata_and_state[-1]
            scale_vals = scale_metadata_and_state[:-3]

            # Resolve input image paths
            paths = _files_to_paths(image_inputs_val)

            # If a crop image is provided, save it and use it as the sole reference
            if crop_image_val is not None:
                try:
                    output_dir = Path(__file__).resolve().parent.parent / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    crop_filename = f"qwen_edit_crop_{int(time.time())}.png"
                    crop_path = output_dir / crop_filename
                    crop_image_val.save(crop_path)
                    paths = [str(crop_path)]
                except Exception as e:
                    print(f"Error saving crop image for Qwen Edit: {e}")

            images, status_text, new_prompt, output_paths = generate_qwen_edit_gradio(
                prompt_val,
                negative_prompt_val,
                paths,
                seed_val,
                width_val,
                height_val,
                steps_val,
                guidance_val,
                quantize_val,
                lora_files_val,
                scale_vals,
                metadata_val,
            )
            return images, status_text, new_prompt, output_paths

        run_button.click(
            fn=_run_qwen_edit,
            inputs=[
                prompt,
                negative_prompt,
                image_inputs,
                seed,
                width,
                height,
                steps,
                guidance,
                quantize,
                lora_files,
                *lora_scale_sliders,
                metadata,
                preview_image,
                last_output_paths,
            ],
            outputs=[output_gallery, status, prompt, last_output_paths],
            concurrency_id="qwen_queue",
        )

        image_inputs.change(
            fn=_first_image,
            inputs=[image_inputs],
            outputs=[preview_image],
        )

        use_last_as_input_btn.click(
            fn=lambda paths: gr.update(value=paths),
            inputs=[last_output_paths],
            outputs=[image_inputs],
        )

        return {
            "prompt": prompt,
            "gallery": output_gallery,
            "status": status,
        }
