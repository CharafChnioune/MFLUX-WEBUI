import gradio as gr
from backend.ic_edit_manager import generate_ic_edit_gradio
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

def create_ic_edit_tab():
    """Create the IC-Edit tab for in-context image editing"""
    
    with gr.TabItem("IC-Edit"):
        gr.Markdown("""
        ## ✏️ IC-Edit - In-Context Editing
        
        ⚠️ **Experimental Feature**: Edit images using natural language instructions. Simply describe what changes you want, 
        and the AI will apply them while preserving the rest of your image.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                reference_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=400
                )
                
                # Instruction/Prompt choice
                instruction_type = gr.Radio(
                    label="Input Type",
                    choices=["Simple Instruction", "Full Prompt"],
                    value="Simple Instruction"
                )
                
                instruction = gr.Textbox(
                    label="Editing Instruction",
                    placeholder="e.g., 'make the hair black', 'add sunglasses', 'change background to beach'",
                    lines=2,
                    visible=True
                )
                
                full_prompt = gr.Textbox(
                    label="Full Prompt",
                    placeholder="Describe the complete diptych with both images...",
                    lines=4,
                    visible=False
                )
                
                # LLM Enhancement section
                with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                    llm_components = create_llm_settings(tab_name="ic_edit", parent_accordion=llm_section)
                
                with gr.Row():
                    enhance_instruction_btn = gr.Button("🔮 Enhance instruction with LLM")
                    enhance_full_prompt_btn = gr.Button("🔮 Enhance full prompt with LLM")
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["dev"],
                    value="dev",
                    info="IC-Edit requires the dev model"
                )
                
                # Example instructions
                gr.Examples(
                    examples=[
                        ["make everything black and white except the flowers"],
                        ["change the hair color to blonde"],
                        ["add sunglasses"],
                        ["remove the hat"],
                        ["change the background to a beach scene"],
                        ["make it look like a painting"],
                        ["add a smile"],
                        ["change the clothing to formal wear"]
                    ],
                    inputs=instruction,
                    label="Example Instructions"
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
                    
                    guidance = gr.Slider(
                        label="Guidance Scale",
                        minimum=15.0,
                        maximum=40.0,
                        step=0.5,
                        value=30.0,
                        info="Higher values follow instructions more strictly"
                    )
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=30,
                        step=1,
                        value=20
                    )
                    
                    num_images = gr.Slider(
                        label="Number of Variations",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                        info="Generate multiple variations of the edit"
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
                        info="8-bit recommended for IC-Edit"
                    )
                
                generate_btn = gr.Button("Apply Edit", variant="primary")
            
            with gr.Column(scale=1):
                # Output section
                output_gallery = gr.Gallery(
                    label="Edited Results",
                    show_label=True,
                    elem_id="ic_edit_gallery",
                    columns=2,
                    rows=2,
                    height=500
                )
                
                status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Accordion("Generation Info", open=False):
                    generation_info = gr.Textbox(
                        label="Last Generation",
                        interactive=False,
                        lines=5
                    )
                
                # Preview section
                with gr.Accordion("Full Diptych Preview", open=False):
                    preview_gallery = gr.Gallery(
                        label="Side-by-Side View (Original | Edited)",
                        show_label=True,
                        columns=1,
                        height=300
                    )
        
        # Help section
        with gr.Accordion("📖 How to use IC-Edit", open=False):
            gr.Markdown("""
            ### In-Context Editing Guide
            
            IC-Edit allows you to edit images using simple text instructions. The AI understands what changes 
            you want and applies them while keeping everything else intact.
            
            **How it works:**
            1. **Upload an image** - The image you want to edit
            2. **Describe the edit** - Use natural language to describe changes
            3. **Generate** - The AI creates an edited version
            
            **Two Input Modes:**
            
            **1. Simple Instruction (Recommended):**
            - Just describe what you want to change
            - Examples: "make the sky purple", "add a beard", "remove the glasses"
            - The system automatically formats your instruction
            
            **2. Full Prompt (Advanced):**
            - Write a complete diptych description
            - Gives you full control over the generation
            - Useful for complex or artistic edits
            
            **Tips for Best Results:**
            
            **Clear Instructions:**
            - Be specific: "change hair to red" vs just "red hair"
            - One change at a time works best
            - Use action words: make, change, add, remove, turn
            
            **Types of Edits that Work Well:**
            - Color changes (hair, clothes, objects)
            - Adding accessories (glasses, hats, jewelry)
            - Removing objects
            - Style transfers (realistic to cartoon, etc.)
            - Background changes
            - Expression changes
            
            **Advanced Tips:**
            - Higher guidance (30-35) for precise edits
            - Try multiple seeds for different interpretations
            - Generate 2-4 variations to find the best result
            - For complex edits, chain multiple simple edits
            
            **Limitations:**
            - Very fine details may be lost
            - Major structural changes can be challenging
            - Some edits may affect nearby areas
            - Results vary with image complexity
            
            **Resolution Note:**
            Images are automatically resized to 512px width (optimal for IC-Edit) while maintaining aspect ratio.
            """)
        
        # Event handlers
        def toggle_instruction_type(choice):
            if choice == "Simple Instruction":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        def random_seed():
            import random
            return random.randint(0, 2**32-1)
        
        # Wire up events
        instruction_type.change(
            fn=toggle_instruction_type,
            inputs=[instruction_type],
            outputs=[instruction, full_prompt]
        )
        
        random_seed_btn.click(fn=random_seed, outputs=[seed])
        
        # Generation function
        def generate_edit(reference_image, instruction_type, instruction, full_prompt,
                         model_name, seed, guidance, steps, num_images,
                         save_metadata, low_ram_mode, quantize):
            if reference_image is None:
                return None, None, "Please upload a reference image", ""
            
            # Determine which input to use
            if instruction_type == "Simple Instruction":
                if not instruction:
                    return None, None, "Please enter an editing instruction", ""
                prompt_input = None
                instruction_input = instruction
            else:
                if not full_prompt:
                    return None, None, "Please enter a full prompt", ""
                prompt_input = full_prompt
                instruction_input = None
            
            try:
                status_msg = f"Applying edit: {instruction_input or 'custom prompt'}..."
                yield None, None, status_msg, ""
                
                images, full_images, metadata = ic_edit_manager.generate_ic_edit_gradio(
                    reference_image=reference_image,
                    instruction=instruction_input,
                    prompt=prompt_input,
                    model_name=model_name,
                    seed=seed,
                    guidance=guidance,
                    steps=steps,
                    num_images=num_images,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize
                )
                
                info = f"Edit applied successfully!\n"
                info += f"Model: {model_name}\n"
                if instruction_input:
                    info += f"Instruction: {instruction_input}\n"
                info += f"Seed: {metadata.get('seed', 'N/A')}\n"
                info += f"Generated {len(images)} variation(s)"
                
                yield images, full_images, "Edit complete!", info
                
            except Exception as e:
                yield None, None, f"Error: {str(e)}", ""
        
        # Enhance prompts with LLM for ic_edit
        def ic_edit_enhance_instruction(p, t, m1, m2, sp, ref_img):
            """Enhanced instruction function for IC-Edit with reference image context"""
            try:
                return enhance_prompt(p, t, m1, m2, sp, ref_img, tab_name="ic_edit_instruction")
            except Exception as e:
                print(f"Error enhancing instruction in ic_edit: {str(e)}")
                return p
        
        def ic_edit_enhance_full_prompt(p, t, m1, m2, sp, ref_img):
            """Enhanced full prompt function for IC-Edit with reference image context"""
            try:
                return enhance_prompt(p, t, m1, m2, sp, ref_img, tab_name="ic_edit_full")
            except Exception as e:
                print(f"Error enhancing full prompt in ic_edit: {str(e)}")
                return p
        
        enhance_instruction_btn.click(
            ic_edit_enhance_instruction,
            inputs=[
                instruction,
                llm_components[0],  # LLM type
                llm_components[1],  # Ollama model
                llm_components[4],  # MLX model
                llm_components[2],  # System prompt
                reference_image     # Reference image for context
            ],
            outputs=instruction
        )
        
        enhance_full_prompt_btn.click(
            ic_edit_enhance_full_prompt,
            inputs=[
                full_prompt,
                llm_components[0],  # LLM type
                llm_components[1],  # Ollama model
                llm_components[4],  # MLX model
                llm_components[2],  # System prompt
                reference_image     # Reference image for context
            ],
            outputs=full_prompt
        )
        
        # Connect generation button
        generate_btn.click(
            fn=generate_edit,
            inputs=[
                reference_image, instruction_type, instruction, full_prompt,
                model_name, seed, guidance, steps, num_images,
                save_metadata, low_ram_mode, quantize
            ],
            outputs=[output_gallery, preview_gallery, status, generation_info]
        )
    
    return reference_image, instruction, generate_btn
