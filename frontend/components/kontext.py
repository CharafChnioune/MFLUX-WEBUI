import gradio as gr
import os
from pathlib import Path
from backend.flux_manager import generate_image_kontext_gradio, get_random_seed
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

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
                    
                    # LLM Enhancement section
                    with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                        llm_components = create_llm_settings(tab_name="kontext", parent_accordion=llm_section)
                    
                    with gr.Row():
                        enhance_prompt_btn = gr.Button("🔮 Enhance prompt with LLM")
                    
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
    
    # Enhance prompt with LLM for kontext
    def kontext_enhance_prompt(p, t, m1, m2, sp, ref_img):
        """Enhanced prompt function for Kontext with reference image context"""
        try:
            return enhance_prompt(p, t, m1, m2, sp, ref_img, tab_name="kontext")
        except Exception as e:
            print(f"Error enhancing prompt in kontext: {str(e)}")
            return p
    
    # Bind events
    random_seed_btn.click(update_seed, outputs=seed)
    
    enhance_prompt_btn.click(
        kontext_enhance_prompt,
        inputs=[
            prompt,
            llm_components[0],  # LLM type
            llm_components[1],  # Ollama model
            llm_components[4],  # MLX model
            llm_components[2],  # System prompt
            reference_image     # Reference image for context
        ],
        outputs=prompt
    )
    
    def run_kontext_generation(
        prompt_val,
        reference_image_val,
        seed_val,
        width_val,
        height_val,
        steps_val,
        guidance_val,
        num_images_val,
        low_ram_val,
        metadata_val,
    ):
        if not reference_image_val:
            return [], "Error: Reference image is required", prompt_val
        parsed_seed = None
        if isinstance(seed_val, (int, float)) and seed_val >= 0:
            parsed_seed = int(seed_val)
        return generate_image_kontext_gradio(
            prompt_val,
            reference_image_val,
            "dev",
            parsed_seed,
            width_val,
            height_val,
            steps_val,
            guidance_val,
            None,
            metadata_val,
            num_images=num_images_val,
            low_ram=low_ram_val,
        )

    generate_btn.click(
        fn=run_kontext_generation,
        inputs=[
            prompt,
            reference_image,
            seed,
            width,
            height,
            steps,
            guidance,
            num_images,
            low_ram,
            metadata,
        ],
        outputs=[output_images, output_info, prompt],
    )
    
    return {
        'prompt': prompt,
        'reference_image': reference_image,
        'output_images': output_images,
        'output_info': output_info
    }
