import gradio as gr
import os
from pathlib import Path
from backend.flux_manager import generate_image_kontext_gradio, get_random_seed

def create_kontext_tab():
    """Create the FLUX.1 Kontext tab"""
    
    with gr.Column():
        gr.Markdown("# 🎭 FLUX.1 Kontext\nAdvanced image editing with text instructions using a reference image")
        gr.Markdown("*Kontext uses the dev-kontext model automatically and requires a reference image*")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                with gr.Group():
                    gr.Markdown("### Prompt & Reference Image")
                    prompt = gr.Textbox(
                        label="Text Instructions",
                        placeholder="Describe the changes you want to make to the reference image...",
                        lines=3,
                        value=""
                    )
                    
                    reference_image = gr.Image(
                        label="Reference Image (Required)",
                        type="filepath",
                        height=300
                    )
                
                with gr.Group():
                    gr.Markdown("### Generation Settings")
                    
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0
                        )
                        random_seed_btn = gr.Button("🎲", elem_classes=["refresh-button"])
                    
                    with gr.Row():
                        width = gr.Number(
                            label="Width",
                            value=1024,
                            precision=0
                        )
                        height = gr.Number(
                            label="Height", 
                            value=1024,
                            precision=0
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=25,
                            step=1,
                            label="Steps"
                        )
                        guidance = gr.Slider(
                            minimum=1.0,
                            maximum=6.0,
                            value=3.0,
                            step=0.1,
                            label="Guidance (Recommended: 2.0-4.0)"
                        )
                    
                    with gr.Row():
                        num_images = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            label="Number of Images"
                        )
                        low_ram = gr.Checkbox(
                            label="Low RAM Mode",
                            value=False
                        )
                
                # MFLUX v0.9.0 Advanced Options
                with gr.Accordion("Advanced Options (MFLUX v0.9.0)", open=False):
                    prompt_file = gr.File(
                        label="Prompt File",
                        file_types=[".txt"],
                        file_count="single"
                    )
                    
                    config_from_metadata = gr.File(
                        label="Config from Metadata",
                        file_types=[".json", ".png", ".jpg", ".jpeg"],
                        file_count="single"
                    )
                    
                    stepwise_output_dir = gr.Textbox(
                        label="Stepwise Output Directory",
                        placeholder="/path/to/stepwise/output"
                    )
                    
                    with gr.Row():
                        vae_tiling = gr.Checkbox(
                            label="VAE Tiling",
                            value=False
                        )
                        vae_tiling_split = gr.Dropdown(
                            choices=["horizontal", "vertical"],
                            label="VAE Tiling Split",
                            value="horizontal"
                        )
                
                with gr.Group():
                    gr.Markdown("### Output Settings")
                    metadata = gr.Checkbox(
                        label="Save Metadata",
                        value=True
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎭 Generate with Kontext",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Output
                with gr.Group():
                    gr.Markdown("### Generated Images")
                    output_images = gr.Gallery(
                        label="Generated Images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        object_fit="contain",
                        height="auto"
                    )
                    
                    output_info = gr.Textbox(
                        label="Generation Info",
                        lines=3,
                        interactive=False
                    )
    
    # Event handlers
    def update_seed():
        return get_random_seed()
    
    # Bind events
    random_seed_btn.click(update_seed, outputs=seed)
    
    generate_btn.click(
        generate_image_kontext_gradio,
        inputs=[
            prompt, reference_image, seed, height, width, steps, guidance,
            metadata, prompt_file, config_from_metadata, stepwise_output_dir,
            vae_tiling, vae_tiling_split, num_images, low_ram
        ],
        outputs=[output_images, output_info, prompt]
    )
    
    return {
        'prompt': prompt,
        'reference_image': reference_image,
        'output_images': output_images,
        'output_info': output_info
    }
