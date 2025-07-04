import gradio as gr
from backend.flux_manager import get_random_seed
from backend.fill_manager import generate_fill_gradio
from backend.model_manager import get_updated_models
from backend.post_processing import update_guidance_visibility
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

def create_fill_tab():
    """
    Create the Fill (Inpaint/Outpaint) tab interface
    """
    with gr.Column():
        with gr.Group():
            gr.Markdown("# 🎨 Fill Tool (Inpaint/Outpaint)")
            gr.Markdown("Use the Fill tool to inpaint (fill in) or outpaint (extend) images. Upload an image and a mask to specify which areas to fill.")
            
        with gr.Row():
            with gr.Column(scale=1):
                # Input images
                with gr.Group():
                    gr.Markdown("### Input Images")
                    input_image = gr.Image(
                        label="Original Image",
                        type="filepath",
                        height=300
                    )
                    mask_image = gr.Image(
                        label="Mask (White = Fill Area)",
                        type="filepath",
                        height=300,
                        image_mode="L",  # Grayscale
                        sources=["upload", "clipboard"]
                    )
                    
                    # Interactive mask tools
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎨 Mask Drawing Tools")
                            mask_tool = gr.Radio(
                                choices=["draw", "rectangle", "ellipse", "eraser"],
                                value="draw",
                                label="Tool",
                                info="Select drawing tool"
                            )
                            
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎛️ Mask Controls")
                            with gr.Row():
                                clear_mask_btn = gr.Button("🗑️ Clear Mask", size="sm")
                                undo_mask_btn = gr.Button("↶ Undo", size="sm")
                                redo_mask_btn = gr.Button("↷ Redo", size="sm")
                    
                    gr.Markdown("💡 **Tips**: "
                              "• White areas will be filled, black areas preserved\n"
                              "• Use rectangle/ellipse for precise shapes\n"
                              "• Eraser mode for corrections\n"
                              "• Undo/Redo for non-destructive editing")
                    
                # Generation settings
                with gr.Group():
                    gr.Markdown("### Generation Settings")
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe what should appear in the filled area...",
                        lines=3
                    )
                    
                    # LLM Enhancement section
                    with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                        llm_components = create_llm_settings(tab_name="fill", parent_accordion=llm_section)
                    
                    with gr.Row():
                        enhance_prompt_btn = gr.Button("🔮 Enhance prompt with LLM")
                    
                    model = gr.Dropdown(
                        choices=get_updated_models(),
                        value="dev-8-bit",
                        label="Model",
                        interactive=True
                    )
                    
                    with gr.Row():
                        seed = gr.Textbox(
                            label="Seed", 
                            value="random",
                            placeholder="Enter seed or 'random'"
                        )
                        random_seed_btn = gr.Button("🎲", elem_classes=["refresh-button"])
                        
                    with gr.Row():
                        width = gr.Slider(
                            minimum=256,
                            maximum=1536,
                            value=512,
                            step=64,
                            label="Width",
                            info="Output width (must be multiple of 64)"
                        )
                        height = gr.Slider(
                            minimum=256,
                            maximum=1536,
                            value=512,
                            step=64,
                            label="Height",
                            info="Output height (must be multiple of 64)"
                        )
                        
                    steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Steps",
                        info="More steps = better quality but slower"
                    )
                    
                    guidance = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=30,
                        step=0.5,
                        label="Guidance Scale",
                        info="Fill works best with high guidance (20-40)"
                    )
                    
                    num_images = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                        label="Number of Images"
                    )
                    
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    metadata = gr.Checkbox(
                        label="Save Metadata",
                        value=True,
                        info="Save generation parameters as JSON"
                    )
                    
                    low_ram = gr.Checkbox(
                        label="Low RAM Mode (8-bit Quantization)",
                        value=False,
                        info="Use less memory but may reduce quality"
                    )
                    
                generate_btn = gr.Button("🎨 Generate Fill", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Output section
                with gr.Group():
                    gr.Markdown("### Generated Images")
                    output_gallery = gr.Gallery(
                        label="Output",
                        show_label=False,
                        elem_id="fill_gallery",
                        columns=2,
                        rows=2,
                        height="auto"
                    )
                    
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        show_label=True
                    )
                    
                    enhanced_prompt_display = gr.Textbox(
                        label="Used Prompt",
                        interactive=False,
                        visible=True
                    )
                    
        # Info section
        with gr.Accordion("ℹ️ How to Use Fill Tool", open=False):
            gr.Markdown("""
            ### Fill Tool Guide
            
            The Fill tool allows you to:
            - **Inpaint**: Remove unwanted objects or fill in missing parts
            - **Outpaint**: Extend images beyond their original boundaries
            
            **Steps:**
            1. Upload your original image
            2. Create or upload a mask image where:
                - White pixels = areas to fill/generate
                - Black pixels = areas to preserve
            3. Write a prompt describing what should appear in the filled areas
            4. Adjust settings and generate
            
            **Tips:**
            - Use high guidance values (20-40) for best results
            - For outpainting, create a mask that extends beyond the image borders
            - Be specific in your prompt about what should appear
            - The fill model automatically downloads on first use (~4GB)
            """)
            
    # Event handlers
    def update_model_choices():
        return gr.Dropdown(choices=get_updated_models())
        
    # Enhance prompt with LLM for fill
    def fill_enhance_prompt(p, t, m1, m2, sp, input_img, mask_img):
        """Enhanced prompt function for Fill with input image context"""
        try:
            # Use input image as context for vision models
            context_image = input_img if input_img else mask_img
            return enhance_prompt(p, t, m1, m2, sp, context_image, tab_name="fill")
        except Exception as e:
            print(f"Error enhancing prompt in fill: {str(e)}")
            return p
    
    # Random seed button
    random_seed_btn.click(
        fn=get_random_seed,
        outputs=seed
    )
    
    # Mask tool handlers
    def clear_mask():
        """Clear the mask canvas"""
        return None
        
    def handle_mask_tool_change(tool):
        """Handle mask tool selection changes"""
        # This would update the mask editor tool in a real implementation
        return mask_image
        
    # Mask tool event handlers
    clear_mask_btn.click(
        fn=clear_mask,
        outputs=mask_image
    )
    
    mask_tool.change(
        fn=handle_mask_tool_change,
        inputs=[mask_tool],
        outputs=[mask_image]
    )
    
    enhance_prompt_btn.click(
        fill_enhance_prompt,
        inputs=[
            prompt,
            llm_components[0],  # LLM type
            llm_components[1],  # Ollama model
            llm_components[4],  # MLX model
            llm_components[2],  # System prompt
            input_image,        # Input image for context
            mask_image          # Mask for additional context
        ],
        outputs=prompt
    )
    
    # Model change handler
    model.change(
        fn=update_guidance_visibility,
        inputs=[model],
        outputs=[guidance]
    )
    
    # Generate button
    generate_btn.click(
        fn=generate_fill_gradio,
        inputs=[
            prompt, input_image, mask_image, seed, height, width, steps, guidance,
            metadata, num_images, low_ram
        ],
        outputs=[output_gallery, status_text, enhanced_prompt_display]
    )
    
    # Return components
    return {
        'prompt': prompt,
        'input_image': input_image,
        'mask_image': mask_image,
        'mask_tool': mask_tool,
        'clear_mask_btn': clear_mask_btn,
        'undo_mask_btn': undo_mask_btn,
        'redo_mask_btn': redo_mask_btn,
        'model': model,
        'seed': seed,
        'width': width,
        'height': height,
        'steps': steps,
        'guidance': guidance,
        'num_images': num_images,
        'metadata': metadata,
        'low_ram': low_ram,
        'output_gallery': output_gallery,
        'status_text': status_text,
        'enhanced_prompt_display': enhanced_prompt_display,
        'generate_btn': generate_btn
    }
