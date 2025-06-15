import gradio as gr
from backend.redux_manager import redux_manager

def create_redux_tab():
    """Create the Redux tool tab for generating image variations"""
    
    with gr.TabItem("Redux"):
        gr.Markdown("""
        ## 🔄 Redux Tool
        Generate variations of existing images while maintaining their core characteristics. 
        Redux allows you to create multiple versions of an image with controlled variations.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=512
                )
                
                prompt = gr.Textbox(
                    label="Prompt (Optional)",
                    placeholder="Optionally describe desired variations...",
                    lines=2,
                    info="Leave empty to use automatic variation"
                )
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["schnell", "dev"],
                    value="schnell"
                )
                
                # Main controls
                with gr.Row():
                    redux_strength = gr.Slider(
                        label="Redux Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.75,
                        info="Higher values create more variation"
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
                        label="Number of Variations",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=4
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
                
                generate_btn = gr.Button("Generate Variations", variant="primary")
            
            with gr.Column(scale=1):
                # Output section
                output_gallery = gr.Gallery(
                    label="Generated Variations",
                    show_label=True,
                    elem_id="redux_gallery",
                    columns=2,
                    rows=4,
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
            ### Redux Tool Guide
            
            The Redux tool creates variations of your input image, perfect for:
            - Exploring different interpretations of the same concept
            - Creating multiple versions for A/B testing
            - Generating variations for creative projects
            - Finding the perfect variation of an image
            
            **How it works:**
            1. **Upload an image** - This will be the base for all variations
            2. **Set Redux Strength** - Controls how much variation to introduce:
               - 0.0-0.3: Very subtle variations
               - 0.4-0.6: Moderate variations
               - 0.7-0.9: Strong variations
               - 1.0: Maximum variation
            3. **Optional prompt** - Guide the variations in a specific direction
            4. **Generate** - Create multiple variations at once
            
            **Tips:**
            - Start with Redux Strength around 0.75 for balanced variations
            - Use the same seed with different strengths to see progression
            - Generate multiple variations to explore possibilities
            - Add a prompt to guide variations in a specific direction
            - Without a prompt, Redux creates general variations
            
            **Examples:**
            - Product photography: Create variations of product shots
            - Character design: Explore different poses or expressions
            - Landscapes: Generate variations with different moods
            - Abstract art: Create a series of related artworks
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
        def generate_redux_variations(input_image, prompt, model_name, redux_strength, seed, 
                                    width, height, guidance, steps, num_images, 
                                    save_metadata, low_ram_mode, quantize):
            if input_image is None:
                return None, "Please upload an input image", ""
            
            try:
                status_msg = f"Generating {num_images} variation(s) with Redux strength {redux_strength}..."
                yield None, status_msg, ""
                
                images, metadata = redux_manager.generate_redux_gradio(
                    input_image=input_image,
                    prompt=prompt if prompt else None,
                    model_name=model_name,
                    redux_strength=redux_strength,
                    seed=seed,
                    width=width,
                    height=height,
                    guidance=guidance,
                    steps=steps,
                    num_images=num_images,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize
                )
                
                info = f"Generated {len(images)} variation(s)\n"
                info += f"Model: {model_name}\n"
                info += f"Redux Strength: {redux_strength}\n"
                if prompt:
                    info += f"Prompt: {prompt}\n"
                info += f"Seed: {metadata.get('seed', 'N/A')}"
                
                yield images, "Variations generated successfully!", info
                
            except Exception as e:
                yield None, f"Error: {str(e)}", ""
        
        # Connect generation button
        generate_btn.click(
            fn=generate_redux_variations,
            inputs=[
                input_image, prompt, model_name, redux_strength, seed, 
                width, height, guidance, steps, num_images,
                save_metadata, low_ram_mode, quantize
            ],
            outputs=[output_gallery, status, generation_info]
        )
    
    return input_image, prompt, generate_btn
