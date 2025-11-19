import gradio as gr
from PIL import Image

from backend.fill_manager import generate_fill_gradio
from backend.model_manager import get_base_model_choices
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

# Layout references:
# - gradiodocs/docs-image/image.md (sketch tool & accessibility notes)
# - gradiodocs/guides-controlling-layout (Row/Column usage)


def _blank_mask(size: int = 1024) -> Image.Image:
    return Image.new("L", (size, size), 0)


def create_fill_tab():
    with gr.TabItem("Fill"):
        gr.Markdown(
            "### 🎨 Fill (Inpaint & Outpaint)\n"
            "Upload an image, paint a mask, and let Flux Fill regenerate the masked region."
        )

        with gr.Row():
            with gr.Column(scale=1):
                base_model = gr.Dropdown(
                    label="Base Model (--base-model)",
                    choices=["Auto"] + get_base_model_choices(),
                    value="Auto",
                )
                original_image = gr.Image(
                    label="Original Image",
                    type="filepath",
                    height=320,
                )
                mask_image = gr.Image(
                    label="Mask (white = fill area)",
                    type="pil",
                    image_mode="L",
                    height=320,
                    value=_blank_mask(),
                )
                reset_mask_btn = gr.Button("Reset Mask", variant="secondary", size="sm")

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what should appear where the mask is painted…",
                    lines=3,
                )

                with gr.Accordion("⚙️ Prompt Helper", open=False) as llm_acc:
                    llm_components = create_llm_settings("fill", llm_acc)
                enhance_btn = gr.Button("🔮 Enhance Prompt")

                seed = gr.Textbox(label="Seed", placeholder="Leave blank for random")
                width = gr.Slider(256, 1536, value=1024, step=64, label="Width")
                height = gr.Slider(256, 1536, value=1024, step=64, label="Height")
                steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                guidance = gr.Slider(5, 40, value=30, step=0.5, label="Guidance")
                num_images = gr.Slider(1, 4, value=1, step=1, label="Variations")

                with gr.Row():
                    metadata = gr.Checkbox(label="Save Metadata", value=True)
                    low_ram = gr.Checkbox(label="Low RAM Mode (8-bit)", value=False)

                generate_btn = gr.Button("Generate Fill", variant="primary")

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Results",
                    columns=2,
                    height=420,
                )
                status = gr.Textbox(label="Status", interactive=False)
                used_prompt = gr.Textbox(label="Used Prompt", interactive=False)

        with gr.Accordion("ℹ️ Tips", open=False):
            gr.Markdown(
                "- Use the mask canvas to paint white over regions you want regenerated.\n"
                "- For outpainting, paint beyond the original borders.\n"
                "- High guidance (20–35) keeps the prompt tightly enforced."
            )

        def _sync_dimensions(image_path):
            try:
                if not image_path:
                    return gr.update(), gr.update()
                img = Image.open(image_path)
                return gr.update(value=img.width), gr.update(value=img.height)
            except Exception:
                return gr.update(), gr.update()

        def _reset_mask():
            return gr.update(value=_blank_mask())

        def _run_fill(
            image_val,
            mask_val,
            prompt_val,
            base_model_val,
            seed_val,
            width_val,
            height_val,
            steps_val,
            guidance_val,
            num_images_val,
            low_ram_val,
            metadata_val,
        ):
            if not image_val or not mask_val:
                return [], "Please provide both image and mask.", prompt_val
            seed_text = ""
            if isinstance(seed_val, str) and seed_val.strip():
                seed_text = seed_val.strip()
            elif isinstance(seed_val, (int, float)) and seed_val >= 0:
                seed_text = str(int(seed_val))

            images, message, prompt_used = generate_fill_gradio(
                prompt_val,
                image_val,
                mask_val,
                base_model_val,
                seed_text,
                int(height_val),
                int(width_val),
                int(steps_val),
                guidance_val,
                metadata_val,
                num_images=int(num_images_val),
                low_ram=low_ram_val,
            )
            return images, message, prompt_used

        original_image.change(
            fn=_sync_dimensions,
            inputs=[original_image],
            outputs=[width, height],
        )
        reset_mask_btn.click(_reset_mask, outputs=[mask_image])
        enhance_btn.click(
            lambda p, *args: enhance_prompt(p, *args, tab_name="fill"),
            inputs=[prompt, *llm_components[:5]],
            outputs=prompt,
        )

        generate_btn.click(
            fn=_run_fill,
            inputs=[
                original_image,
                mask_image,
                prompt,
                base_model,
                seed,
                width,
                height,
                steps,
                guidance,
                num_images,
                low_ram,
                metadata,
            ],
            outputs=[output_gallery, status, used_prompt],
            concurrency_id="fill_queue",
        )

    return {}
