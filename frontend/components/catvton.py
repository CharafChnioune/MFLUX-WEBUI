import gradio as gr
from backend.catvton_manager import catvton_manager

def create_catvton_tab():
    """Create the CatVTON tab for virtual try-on"""
    
    with gr.TabItem("CatVTON"):
        gr.Markdown("""
        ## 👕 CatVTON - Virtual Try-On
        
        ⚠️ **Experimental Feature**: Generate realistic virtual try-on images by combining person photos with garment images.
        
        This tool uses AI to virtually "dress" a person in different garments while maintaining realistic lighting, shadows, and fit.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input images
                with gr.Row():
                    person_image = gr.Image(
                        label="Person Image",
                        type="pil",
                        height=300
                    )
                    person_mask = gr.Image(
                        label="Person Mask",
                        type="pil",
                        height=300
                    )
                
                garment_image = gr.Image(
                    label="Garment Image",
                    type="pil",
                    height=300
                )
                
                # Quick mask tools
                with gr.Row():
                    auto_mask_btn = gr.Button("🎯 Auto Generate Mask", size="sm")
                    mask_tips_btn = gr.Button("💡 Mask Tips", size="sm")
                
                prompt = gr.Textbox(
                    label="Prompt (Optional)",
                    placeholder="Leave empty for automatic prompt, or describe the try-on scenario...",
                    lines=3,
                    info="Automatic prompts are optimized for virtual try-on"
                )
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["dev"],
                    value="dev",
                    info="CatVTON requires the dev model"
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=6269363,
                            precision=0,
                            minimum=-1,
                            maximum=2**32-1,
                            info="Default seed optimized for try-on"
                        )
                        random_seed_btn = gr.Button("🎲", scale=0, min_width=40)
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=891,
                            info="Adjust to match person image aspect ratio"
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=1024
                        )
                    
                    guidance = gr.Slider(
                        label="Guidance Scale",
                        minimum=15.0,
                        maximum=40.0,
                        step=0.5,
                        value=30.0,
                        info="Higher values = stronger adherence to garment"
                    )
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=30,
                        step=1,
                        value=20
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
                        value=8,
                        info="8-bit recommended for CatVTON"
                    )
                
                generate_btn = gr.Button("Generate Try-On", variant="primary")
            
            with gr.Column(scale=1):
                # Output section
                output_image = gr.Image(
                    label="Virtual Try-On Result",
                    type="pil",
                    height=600
                )
                
                status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Accordion("Generation Info", open=False):
                    generation_info = gr.Textbox(
                        label="Last Generation",
                        interactive=False,
                        lines=5
                    )
                
                # Preview section
                with gr.Accordion("Side-by-Side Preview", open=False):
                    preview_image = gr.Image(
                        label="Full Diptych (Garment | Person)",
                        type="pil",
                        height=400
                    )
        
        # Help section
        with gr.Accordion("📖 How to use CatVTON", open=False):
            gr.Markdown("""
            ### Virtual Try-On Guide
            
            **Required Inputs:**
            1. **Person Image** - Photo of the person who will "wear" the garment
            2. **Person Mask** - Binary mask showing where the garment should appear (white = garment area)
            3. **Garment Image** - The clothing item to virtually try on
            
            **Creating the Mask:**
            - White pixels = where the garment will be placed
            - Black pixels = areas to keep from original person image
            - Use image editing software or the auto-mask feature
            - Common mask areas: upper body for shirts, lower body for pants/skirts
            
            **Tips for Best Results:**
            - **High-Quality Images**: Use clear, well-lit photos
            - **Consistent Poses**: Front-facing poses work best
            - **Accurate Masks**: Precise masks produce better results
            - **Garment Types**: Works best with:
              - Shirts, blouses, t-shirts
              - Dresses and gowns
              - Jackets and coats
              - Simple patterns and solid colors
            
            **Limitations:**
            - Complex patterns may not transfer perfectly
            - Very fitted clothing may have sizing issues
            - Accessories and small details might be lost
            - Multiple garments in one try-on not supported
            
            **Recommended Settings:**
            - Guidance: 30 (higher for stronger garment adherence)
            - Steps: 20 (balance between quality and speed)
            - Seed: Try multiple seeds for best results
            - Resolution: Match person image aspect ratio
            
            **Automatic Prompting:**
            If you leave the prompt empty, CatVTON uses an optimized prompt template designed specifically for virtual try-on scenarios.
            """)
        
        # Mask tips modal
        with gr.Row(visible=False) as mask_tips_modal:
            gr.Markdown("""
            ### 🎭 Creating Effective Masks
            
            **Quick Tips:**
            - Use any image editor (Photoshop, GIMP, Paint)
            - Paint white where you want the garment
            - Keep black where you want the original person
            - Save as PNG for best quality
            
            **Common Mask Templates:**
            - **Shirt/Top**: Cover torso and arms
            - **Dress**: Cover from shoulders to desired length
            - **Pants**: Cover from waist to ankles
            - **Full Outfit**: Cover entire body except head
            
            **Pro Tips:**
            - Feather edges slightly for natural blending
            - Include a bit of neck area for tops
            - Extend mask slightly beyond garment boundaries
            """)
        
        # Event handlers
        def toggle_mask_tips():
            return gr.update(visible=True)
        
        def random_seed():
            import random
            return random.randint(0, 2**32-1)
        
        def auto_generate_mask(person_image):
            # Placeholder for auto-mask generation
            # In a real implementation, this would use segmentation models
            return None, "Auto-mask generation not yet implemented. Please create mask manually."
        
        # Wire up events
        mask_tips_btn.click(fn=toggle_mask_tips, outputs=[mask_tips_modal])
        random_seed_btn.click(fn=random_seed, outputs=[seed])
        auto_mask_btn.click(
            fn=auto_generate_mask,
            inputs=[person_image],
            outputs=[person_mask, status]
        )
        
        # Generation function
        def generate_tryon(person_image, person_mask, garment_image, prompt, model_name,
                          seed, width, height, guidance, steps, save_metadata,
                          low_ram_mode, quantize):
            if person_image is None:
                return None, None, "Please upload a person image", ""
            if person_mask is None:
                return None, None, "Please upload or create a person mask", ""
            if garment_image is None:
                return None, None, "Please upload a garment image", ""
            
            try:
                status_msg = "Generating virtual try-on..."
                yield None, None, status_msg, ""
                
                # Use automatic prompt if none provided
                if not prompt:
                    prompt = None  # Manager will use default
                
                result_image, metadata = catvton_manager.generate_catvton_gradio(
                    person_image=person_image,
                    person_mask=person_mask,
                    garment_image=garment_image,
                    prompt=prompt,
                    model_name=model_name,
                    seed=seed,
                    width=width,
                    height=height,
                    guidance=guidance,
                    steps=steps,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize
                )
                
                # Extract the right half (person with garment) for main display
                full_width = result_image.width
                right_half = result_image.crop((full_width // 2, 0, full_width, result_image.height))
                
                info = f"Virtual try-on complete!\n"
                info += f"Model: {model_name}\n"
                info += f"Seed: {metadata.get('seed', 'N/A')}\n"
                info += f"Guidance: {guidance}\n"
                info += f"Resolution: {width}x{height}"
                
                yield right_half, result_image, "Try-on generated successfully!", info
                
            except Exception as e:
                yield None, None, f"Error: {str(e)}", ""
        
        # Connect generation button
        generate_btn.click(
            fn=generate_tryon,
            inputs=[
                person_image, person_mask, garment_image, prompt, model_name,
                seed, width, height, guidance, steps, save_metadata,
                low_ram_mode, quantize
            ],
            outputs=[output_image, preview_image, status, generation_info]
        )
    
    return person_image, garment_image, generate_btn
