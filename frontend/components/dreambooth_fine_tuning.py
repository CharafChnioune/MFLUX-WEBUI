import gradio as gr
from backend.model_manager import get_updated_models
from backend.captions import (
    show_uploaded_images,
    fill_captions
)
from backend.training_manager import run_dreambooth_from_ui_no_explicit_quantize
import os

def create_dreambooth_fine_tuning_tab():
    """Create the Dreambooth Fine-Tuning tab interface"""
    gr.Markdown(
        """
        ### Dreambooth Fine-Tuning
        Train your own LoRA model by uploading images and setting parameters.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            uploaded_images = gr.Files(
                label="Training Images",
                file_types=["image"],
                file_count="multiple"
            )
            
            base_model_dd = gr.Dropdown(
                choices=get_updated_models(),
                value="schnell-4-bit",
                label="Base Model",
                allow_custom_value=True
            )
            
            prompt_for_images = gr.Textbox(
                label="Training Prompt (trigger word)",
                value="a photo of sks",
                lines=1
            )

            # Hidden state for MLX-VLM model
            mlx_vlm_model = gr.State(value="mlx-community/Florence-2-large-ft-bf16")
            
            with gr.Row():
                create_captions = gr.Button("Create captions with MLX-VLM", variant="primary")
                
            image_caption_pairs = []
            with gr.Group():
                with gr.Row():
                    for i in range(4):
                        with gr.Column():
                            img = gr.Image(
                                show_label=False,
                                visible=False,
                                height=200,
                                container=False
                            )
                            caption = gr.Textbox(
                                label=f"Caption {i+1}",
                                show_label=True,
                                visible=False,
                                lines=3,
                                interactive=True
                            )
                            image_caption_pairs.extend([img, caption])
                with gr.Row():
                    for i in range(4, 8):
                        with gr.Column():
                            img = gr.Image(
                                show_label=False,
                                visible=False,
                                height=200,
                                container=False
                            )
                            caption = gr.Textbox(
                                label=f"Caption {i+1}",
                                show_label=True,
                                visible=False,
                                lines=3,
                                interactive=True
                            )
                            image_caption_pairs.extend([img, caption])
                with gr.Row():
                    for i in range(8, 12):
                        with gr.Column():
                            img = gr.Image(
                                show_label=False,
                                visible=False,
                                height=200,
                                container=False
                            )
                            caption = gr.Textbox(
                                label=f"Caption {i+1}",
                                show_label=True,
                                visible=False,
                                lines=3,
                                interactive=True
                            )
                            image_caption_pairs.extend([img, caption])
                with gr.Row():
                    for i in range(12, 16):
                        with gr.Column():
                            img = gr.Image(
                                show_label=False,
                                visible=False,
                                height=200,
                                container=False
                            )
                            caption = gr.Textbox(
                                label=f"Caption {i+1}",
                                show_label=True,
                                visible=False,
                                lines=3,
                                interactive=True
                            )
                            image_caption_pairs.extend([img, caption])
                with gr.Row():
                    for i in range(16, 20):
                        with gr.Column():
                            img = gr.Image(
                                show_label=False,
                                visible=False,
                                height=200,
                                container=False
                            )
                            caption = gr.Textbox(
                                label=f"Caption {i+1}",
                                show_label=True,
                                visible=False,
                                lines=3,
                                interactive=True
                            )
                            image_caption_pairs.extend([img, caption])

            # Event handlers
            def show_images(files):
                """Show uploaded images immediately"""
                if not files:
                    return [gr.update(value=None, visible=False)] * 40
                updates = []
                for i, f in enumerate(files[:20]):
                    updates.extend([
                        gr.update(value=f.name, visible=True),
                        gr.update(value="", visible=True) 
                    ])
                remaining = 20 - len(files[:20])
                for _ in range(remaining):
                    updates.extend([
                        gr.update(value=None, visible=False),
                        gr.update(value="", visible=False)
                    ])
                return updates

            uploaded_images.change(
                fn=show_images,
                inputs=[uploaded_images],
                outputs=image_caption_pairs
            )

            create_captions.click(
                fn=fill_captions,
                inputs=[uploaded_images, mlx_vlm_model, prompt_for_images],
                outputs=image_caption_pairs
            )

        with gr.Column(scale=1):
            gr.Markdown("### Training Parameters")
            
            with gr.Group():
                image_size = gr.Dropdown(
                    choices=[
                        "256x256",
                        "512x512",
                        "768x768",
                        "1024x1024"
                    ],
                    label="Training Image Size",
                    value="512x512",
                    info="Images will be resized to this size while maintaining aspect ratio"
                )

                epochs_txt = gr.Number(
                    label="Epochs (number of training iterations)",
                    value=10,
                    precision=0
                )
                
                batch_size_txt = gr.Number(
                    label="Batch Size (images processed at once)",
                    value=1,
                    precision=0
                )
                
                lora_rank_txt = gr.Number(
                    label="LoRA Rank (higher = stronger effect but larger file)",
                    value=4,
                    precision=0
                )
                
                with gr.Row():
                    output_dir_txt = gr.Textbox(
                        label="Output Directory",
                        value=os.path.expanduser("~/Desktop/mflux_training"),
                        lines=1,
                        interactive=True
                    )

                resume_chkpt_txt = gr.Textbox(
                    label="Resume from Checkpoint (optional)",
                    value="",
                    placeholder="Path to checkpoint.zip",
                    lines=1
                )

            gr.Markdown("""
            ### Advanced Training Options
            Control which parts of the model to train. More layers = better results but slower training.
            """)

            with gr.Accordion("🔄 Transformer Blocks (Early Layers)", open=False):
                gr.Markdown("""
                Early network layers (0-19) that have more impact but are slower to train.
                Training these requires more memory but can give better results.
                Enable only if you have enough memory and want better results.
                """)
                transformer_blocks_enabled = gr.Checkbox(
                    label="Enable Transformer Blocks",
                    value=True
                )
                transformer_start = gr.Slider(
                    label="Start Block",
                    minimum=0,
                    maximum=19,
                    value=0,
                    step=1
                )
                transformer_end = gr.Slider(
                    label="End Block",
                    minimum=0,
                    maximum=19,
                    value=19,
                    step=1
                )

            with gr.Accordion("🔄 Single Transformer Blocks (Late Layers)", open=False):
                gr.Markdown("""
                Later network layers (0-38) that are faster to train and use less memory.
                Recommended for most training runs.
                """)
                single_blocks_enabled = gr.Checkbox(
                    label="Enable Single Transformer Blocks",
                    value=True
                )
                single_start = gr.Slider(
                    label="Start Block",
                    minimum=0,
                    maximum=38,
                    value=0,
                    step=1
                )
                single_end = gr.Slider(
                    label="End Block",
                    minimum=0,
                    maximum=38,
                    value=38,
                    step=1
                )

    with gr.Row():
        with gr.Column(scale=1):
            start_train_btn_v2 = gr.Button("Start Training", variant="primary", size="lg")
    
    training_progress = gr.Textbox(
        label="Training Progress",
        interactive=False,
        lines=10
    )

    debug_dir = gr.Textbox(
        label="Debug Config Location", 
        value="",
        interactive=False
    )

    def on_start_training(*args):
        try:
            result = run_dreambooth_from_ui_no_explicit_quantize(*args)
            
            debug_path = os.path.join(args[6], "debug_config.json")
            return {
                training_progress: result,
                debug_dir: f"Debug config saved to: {debug_path}"
            }
        except Exception as e:
            return {
                training_progress: f"Error: {str(e)}",
                debug_dir: "Failed to save debug config"
            }

    start_train_btn_v2.click(
        fn=on_start_training,
        inputs=[
            base_model_dd,
            uploaded_images,
            prompt_for_images,
            epochs_txt,
            batch_size_txt,
            lora_rank_txt,
            output_dir_txt,
            resume_chkpt_txt,
            mlx_vlm_model,
            transformer_blocks_enabled,
            transformer_start,
            transformer_end,
            single_blocks_enabled,
            single_start,
            single_end,
            image_size,
            *[pair[1] for pair in zip(image_caption_pairs[::2], image_caption_pairs[1::2])]
        ],
        outputs=[training_progress, debug_dir]
    )

    return {
        'base_model': base_model_dd,
        'uploaded_images': uploaded_images,
        'training_progress': training_progress,
        'mlx_vlm_model': mlx_vlm_model
    } 