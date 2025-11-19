import gradio as gr
from backend.catvton_manager import generate_catvton_gradio
from backend.model_manager import get_updated_models, get_base_model_choices
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

# Layout references: gradiodocs/docs-blocks, gradiodocs/guides-controlling-layout

STYLE_CHOICES = [
    "Professional",
    "Casual",
    "Fashion",
    "Outdoor",
    "Custom",
]


def create_catvton_tab():
    """Create the CatVTON virtual try-on tab following the Gradio layout docs."""

    with gr.TabItem("CatVTON"):
        gr.Markdown(
            """
            ## 👕 CatVTON (Virtual Try-On)
            Upload a person photo plus a garment image to generate a realistic try-on preview.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Inputs** · clear, front-facing people + isolated garments yield the best results")
                person_image = gr.Image(
                    label="Person Image",
                    type="filepath",
                    height=300,
                )
                garment_image = gr.Image(
                    label="Garment Image",
                    type="filepath",
                    height=300,
                )

                style_dropdown = gr.Dropdown(
                    label="Prompt Style",
                    choices=STYLE_CHOICES,
                    value="Professional",
                    info="Choose an automatic styling template or switch to Custom.",
                )
                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Describe the try-on scene...",
                    visible=False,
                    lines=3,
                )

                def _toggle_custom(style):
                    return gr.update(visible=style == "Custom")

                style_dropdown.change(
                    fn=_toggle_custom,
                    inputs=style_dropdown,
                    outputs=custom_prompt,
                )

                with gr.Accordion("⚙️ LLM Prompt Helper", open=False) as llm_acc:
                    llm_components = create_llm_settings(tab_name="catvton", parent_accordion=llm_acc)
                    enhance_btn = gr.Button("🔮 Enhance Prompt")

                def _enhance(p, *llm_args):
                    try:
                        return enhance_prompt(p, *llm_args, tab_name="catvton")
                    except Exception as exc:
                        print(f"CatVTON prompt enhancement failed: {exc}")
                        return p

                enhance_btn.click(
                    _enhance,
                    inputs=[custom_prompt, *llm_components[:5]],
                    outputs=custom_prompt,
                )

                with gr.Accordion("Model Settings", open=False):
                    model = gr.Dropdown(
                        label="Model",
                        choices=get_updated_models(),
                        value="dev",
                        allow_custom_value=True,
                    )
                    base_model = gr.Dropdown(
                        label="Base Model (--base-model)",
                        choices=["Auto"] + get_base_model_choices(),
                        value="Auto",
                        info="Required when using custom Hugging Face checkpoints.",
                    )
                    num_images = gr.Slider(
                        label="Variations",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                    )

                with gr.Accordion("Generation Settings", open=True):
                    seed = gr.Textbox(label="Seed", placeholder="Leave blank for random")
                    width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                    height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                    steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                    guidance = gr.Slider(5.0, 40.0, value=30.0, step=0.5, label="Guidance")
                    low_ram = gr.Checkbox(label="Low RAM Mode", value=False)
                    metadata = gr.Checkbox(label="Save Metadata", value=True)

                generate_btn = gr.Button("Generate Try-On", variant="primary")

            with gr.Column(scale=1):
                result_image = gr.Image(label="Try-On Preview", type="filepath", height=512)
                status = gr.Textbox(label="Status", interactive=False)
                info_box = gr.Textbox(label="Generation Info", lines=4, interactive=False)

        with gr.Accordion("📘 Tips", open=False):
            gr.Markdown(
                """
                - Use high-resolution, front-facing person photos.
                - The garment works best when photographed straight on with minimal wrinkles.
                - Adjust the prompt style to match the desired shoot (professional, casual, etc.).
                """
            )

        def _run_tryon(
            person_path,
            garment_path,
            model_val,
            base_model_val,
            style_choice,
            custom_prompt_val,
            seed_val,
            width_val,
            height_val,
            steps_val,
            guidance_val,
            num_images_val,
            low_ram_val,
            metadata_val,
        ):
            if not person_path or not garment_path:
                return None, "Please provide both person and garment images.", ""

            if style_choice == "Custom" and custom_prompt_val:
                prompt_style = custom_prompt_val
            else:
                prompt_style = style_choice

            seed_text = ""
            if isinstance(seed_val, str) and seed_val.strip():
                seed_text = seed_val.strip()
            elif isinstance(seed_val, (int, float)) and seed_val >= 0:
                seed_text = str(int(seed_val))

            images, message = generate_catvton_gradio(
                person_image=person_path,
                clothing_image=garment_path,
                model=model_val,
                base_model=base_model_val,
                seed=seed_text,
                height=height_val,
                width=width_val,
                steps=steps_val,
                guidance=guidance_val,
                prompt_style=prompt_style,
                metadata=metadata_val,
                num_images=int(num_images_val),
                low_ram=low_ram_val,
            )

            image = images[0] if images else None
            info = f"Model: {model_val}\nBase Model: {base_model_val}\nGuidance: {guidance_val}\nResolution: {width_val}x{height_val}"
            return image, message, info

        generate_btn.click(
            fn=_run_tryon,
            inputs=[
                person_image,
                garment_image,
                model,
                base_model,
                style_dropdown,
                custom_prompt,
                seed,
                width,
                height,
                steps,
                guidance,
                num_images,
                low_ram,
                metadata,
            ],
            outputs=[result_image, status, info_box],
            concurrency_id="catvton_queue",
        )

    return {}
