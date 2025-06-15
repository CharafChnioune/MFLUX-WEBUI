import gradio as gr
from backend.depth_manager import depth_manager

def create_depth_tab():
    """Create the Depth tool tab for generating images from depth maps"""
    
    with gr.TabItem("Depth"):
        gr.Markdown("""
        ## 🔍 Depth Tool
        Generate images based on depth maps using your own prompts. The tool extracts depth information from 
        reference images and uses it to guide the generation of new images with similar spatial structure.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                reference_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=512
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["schnell", "dev"],
                    value="schnell"
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            minimum=-1,
                            maximum=2**32-1
                        )
                        random_seed_btn = gr.Button("🎲", scale=0, min_width=40)
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                    
                    guidance = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=3.5
                    )
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=4 if model_name.value == "schnell" else 20
                    )
                    
                    num_images = gr.Slider(
                        label="Number of Images",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1
                    )
                    
                    controlnet_strength = gr.Slider(
                        label="ControlNet Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.9,
                        info="How strongly the depth map influences generation"
                    )
                    
                    save_metadata = gr.Checkbox(
                        label="Save Metadata",
                        value=True,
                        info="Save generation parameters with the image"
                    )
                    
                    low_ram_mode = gr.Checkbox(
                        label="Low RAM Mode",
                        value=False,
                        info="Reduce memory usage (slower)"
                    )
                    
                    quantize = gr.Dropdown(
                        label="Quantization",
                        choices=[("None", 0), ("8-bit", 8), ("6-bit", 6), ("4-bit", 4), ("3-bit", 3)],
                        value=0,
                        info="Use quantized model for reduced memory usage"
                    )
                    
                    save_canny = gr.Checkbox(
                        label="Save Depth Map",
                        value=False,
                        info="Save the extracted depth map as a separate image"
                    )
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1):
                # Output section
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="depth_gallery",
                    columns=2,
                    rows=2,
                    height=600
                )
                
                status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Accordion("Generation Info", open=False):
                    generation_info = gr.Textbox(
                        label="Last Generation",
                        interactive=False,
                        lines=5
                    )
        
        # Help section
        with gr.Accordion("📖 How to use", open=False):
            gr.Markdown("""
            ### Depth Tool Guide
            
            1. **Upload a reference image** - The tool will extract depth information from this image
            2. **Enter your prompt** - Describe what you want to generate while keeping the same depth structure
            3. **Adjust settings** - Fine-tune the generation parameters
            4. **Generate** - Create new images with the same spatial structure as your reference
            
            **Tips:**
            - The depth tool works best with images that have clear foreground/background separation
            - Higher ControlNet strength (0.8-1.0) preserves the depth structure more strictly
            - Lower strength (0.3-0.6) allows more creative freedom while still following the general structure
            - Use the 'Save Depth Map' option to see the extracted depth information
            
            **Examples:**
            - Transform a portrait into a different character while keeping the same pose
            - Change the style of a landscape while preserving the composition
            - Generate variations of objects with the same spatial arrangement
            """)
        
        # Event handlers
        def update_steps(model):
            return 4 if model == "schnell" else 20
        
        def random_seed():
            import random
            return random.randint(0, 2**32-1)
        
        # Wire up events
        model_name.change(fn=update_steps, inputs=[model_name], outputs=[steps])
        random_seed_btn.click(fn=random_seed, outputs=[seed])
        
        # Generation function wrapper
        def generate_depth_images(reference_image, prompt, model_name, seed, width, height, 
                                 guidance, steps, num_images, controlnet_strength, 
                                 save_metadata, low_ram_mode, quantize, save_canny):
            if reference_image is None:
                return None, "Please upload a reference image", ""
            
            if not prompt:
                return None, "Please enter a prompt", ""
            
            try:
                status_msg = f"Generating {num_images} image(s) with depth control..."
                yield None, status_msg, ""
                
                images, metadata = depth_manager.generate_depth_gradio(
                    reference_image=reference_image,
                    prompt=prompt,
                    model_name=model_name,
                    seed=seed,
                    width=width,
                    height=height,
                    guidance=guidance,
                    steps=steps,
                    num_images=num_images,
                    controlnet_strength=controlnet_strength,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize,
                    save_canny=save_canny
                )
                
                info = f"Generated {len(images)} image(s)\n"
                info += f"Model: {model_name}\n"
                info += f"Prompt: {prompt}\n"
                info += f"Seed: {metadata.get('seed', 'N/A')}\n"
                info += f"ControlNet Strength: {controlnet_strength}"
                
                yield images, "Generation complete!", info
                
            except Exception as e:
                yield None, f"Error: {str(e)}", ""
        
        # Connect generation button
        generate_btn.click(
            fn=generate_depth_images,
            inputs=[
                reference_image, prompt, model_name, seed, width, height,
                guidance, steps, num_images, controlnet_strength,
                save_metadata, low_ram_mode, quantize, save_canny
            ],
            outputs=[output_gallery, status, generation_info]
        )
    
    return reference_image, prompt, generate_btn
