import gradio as gr
from backend.concept_attention_manager import concept_attention_manager

def create_concept_attention_tab():
    """Create the Concept Attention tab for fine-grained prompt control"""
    
    with gr.TabItem("Concept Attention"):
        gr.Markdown("""
        ## 🧠 Concept Attention
        
        Fine-tune the importance of specific concepts in your prompts using attention weights. 
        This feature allows you to emphasize or de-emphasize particular elements in your generated images.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                prompt = gr.Textbox(
                    label="Prompt with Attention Tags",
                    placeholder="A {beautiful:1.5} sunset over the {ocean:0.8} with {vibrant:1.3} colors",
                    lines=4,
                    info="Use {concept:weight} syntax. Default weight is 1.0"
                )
                
                # Attention syntax helper
                with gr.Accordion("📝 Attention Syntax Guide", open=True):
                    gr.Markdown("""
                    **Basic Syntax:** `{concept:weight}`
                    
                    **Weight Values:**
                    - `> 1.0` : Increase attention (emphasize)
                    - `< 1.0` : Decrease attention (de-emphasize)
                    - `1.0` : Normal attention (default)
                    
                    **Examples:**
                    - `{red:1.5}` - 50% more attention to "red"
                    - `{background:0.5}` - 50% less attention to "background"
                    - `{detailed:2.0}` - Double attention to "detailed"
                    """)
                
                # Quick templates
                gr.Examples(
                    examples=[
                        ["A {majestic:1.5} mountain landscape with {subtle:0.7} clouds and {vibrant:1.3} autumn colors"],
                        ["Portrait of a {wise:1.4} old {wizard:1.2} with a {long:0.8} beard and {piercing:1.6} blue eyes"],
                        ["A {futuristic:1.5} city with {towering:1.3} skyscrapers and {minimal:0.5} traffic"],
                        ["{Detailed:1.5} sketch of a {mechanical:1.3} dragon with {intricate:1.4} gears and {subtle:0.6} steam"],
                        ["A {serene:1.2} Japanese garden with {delicate:1.3} cherry blossoms and {calm:1.1} water"]
                    ],
                    inputs=prompt,
                    label="Example Prompts"
                )
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["schnell", "dev"],
                    value="schnell"
                )
                
                # Attention analysis
                analyze_btn = gr.Button("🔍 Analyze Attention", size="sm")
                attention_preview = gr.Textbox(
                    label="Attention Analysis",
                    interactive=False,
                    lines=3,
                    visible=False
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
                
                generate_btn = gr.Button("Generate with Attention", variant="primary")
            
            with gr.Column(scale=1):
                # Output section
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="concept_gallery",
                    columns=2,
                    rows=2,
                    height=600
                )
                
                status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Accordion("Generation Info", open=False):
                    generation_info = gr.Textbox(
                        label="Last Generation",
                        interactive=False,
                        lines=8
                    )
        
        # Comparison section
        with gr.Accordion("🔬 Compare With/Without Attention", open=False):
            gr.Markdown("""
            Generate the same prompt with and without attention weights to see the difference.
            """)
            compare_btn = gr.Button("Generate Comparison", variant="secondary")
            
            with gr.Row():
                compare_gallery = gr.Gallery(
                    label="Comparison Results",
                    columns=2,
                    rows=1,
                    height=400
                )
            
            compare_info = gr.Textbox(
                label="Comparison Details",
                interactive=False,
                lines=3
            )
        
        # Help section
        with gr.Accordion("📖 Advanced Guide", open=False):
            gr.Markdown("""
            ### Concept Attention Deep Dive
            
            **Understanding Weights:**
            - Weights modify how much the model "pays attention" to specific concepts
            - Think of it as a volume knob for different parts of your prompt
            - Effects are relative - increasing one concept's weight can reduce others' influence
            
            **Best Practices:**
            
            1. **Start Subtle** - Begin with weights between 0.8-1.2
            2. **Balance is Key** - Too many high weights can overwhelm the model
            3. **Test Incrementally** - Adjust weights gradually to see effects
            4. **Context Matters** - Same weight can have different effects in different prompts
            
            **Advanced Techniques:**
            
            **Emphasizing Details:**
            ```
            A portrait with {sharp:1.3} focus on the {eyes:1.5} and {soft:0.7} background
            ```
            
            **Controlling Style:**
            ```
            {Photorealistic:1.5} landscape with {painterly:0.3} elements
            ```
            
            **Managing Complexity:**
            ```
            {Simple:1.2} composition with {minimal:1.3} colors and {clean:1.1} lines
            ```
            
            **Color Control:**
            ```
            A {vibrant:1.4} sunset with {orange:1.3} and {purple:1.2} hues, {blue:0.5} undertones
            ```
            
            **Common Weight Ranges:**
            - `0.5-0.8` : Subtle de-emphasis
            - `0.8-1.2` : Fine adjustments
            - `1.2-1.5` : Noticeable emphasis
            - `1.5-2.0` : Strong emphasis
            - `> 2.0` : Extreme emphasis (use sparingly)
            
            **Troubleshooting:**
            - If results are too extreme, reduce all weights proportionally
            - If no effect is visible, increase weight differences
            - For subtle changes, use more steps (20-30)
            - Higher guidance scale can amplify weight effects
            """)
        
        # Event handlers
        def update_steps(model):
            return 4 if model == "schnell" else 20
        
        def random_seed():
            import random
            return random.randint(0, 2**32-1)
        
        def analyze_attention(prompt):
            """Parse and display attention weights"""
            import re
            pattern = r'\{([^:}]+):([^}]+)\}'
            matches = re.findall(pattern, prompt)
            
            if not matches:
                return gr.update(visible=True, value="No attention tags found. Use {concept:weight} syntax.")
            
            analysis = "Attention weights found:\n"
            for concept, weight in matches:
                try:
                    w = float(weight)
                    if w > 1.0:
                        analysis += f"• {concept}: {w} (↑ emphasized)\n"
                    elif w < 1.0:
                        analysis += f"• {concept}: {w} (↓ de-emphasized)\n"
                    else:
                        analysis += f"• {concept}: {w} (normal)\n"
                except:
                    analysis += f"• {concept}: {weight} (⚠️ invalid weight)\n"
            
            return gr.update(visible=True, value=analysis)
        
        # Wire up events
        model_name.change(fn=update_steps, inputs=[model_name], outputs=[steps])
        random_seed_btn.click(fn=random_seed, outputs=[seed])
        analyze_btn.click(fn=analyze_attention, inputs=[prompt], outputs=[attention_preview])
        
        # Generation function
        def generate_with_attention(prompt, model_name, seed, width, height, guidance, steps,
                                  num_images, save_metadata, low_ram_mode, quantize):
            if not prompt:
                return None, "Please enter a prompt with attention tags", ""
            
            try:
                # Check if prompt has attention tags
                import re
                has_attention = bool(re.search(r'\{[^:}]+:[^}]+\}', prompt))
                
                status_msg = "Generating with concept attention..." if has_attention else "Generating (no attention tags found)..."
                yield None, status_msg, ""
                
                images, metadata = concept_attention_manager.generate_attention_gradio(
                    prompt=prompt,
                    model_name=model_name,
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
                
                info = f"Generated {len(images)} image(s)\n"
                info += f"Model: {model_name}\n"
                info += f"Prompt: {prompt}\n"
                info += f"Seed: {metadata.get('seed', 'N/A')}\n"
                
                # Add attention info
                if 'attention_weights' in metadata:
                    info += "\nAttention weights applied:\n"
                    for concept, weight in metadata['attention_weights'].items():
                        info += f"• {concept}: {weight}\n"
                elif has_attention:
                    info += "\n⚠️ Attention tags found but may not be supported by model"
                
                yield images, "Generation complete!", info
                
            except Exception as e:
                yield None, f"Error: {str(e)}", ""
        
        # Comparison function
        def generate_comparison(prompt, model_name, seed, width, height, guidance, steps,
                              save_metadata, low_ram_mode, quantize):
            if not prompt:
                return None, "Please enter a prompt with attention tags"
            
            try:
                # Generate with attention
                yield None, "Generating with attention weights..."
                
                images_with, _ = concept_attention_manager.generate_attention_gradio(
                    prompt=prompt,
                    model_name=model_name,
                    seed=seed,
                    width=width,
                    height=height,
                    guidance=guidance,
                    steps=steps,
                    num_images=1,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize
                )
                
                # Strip attention tags for comparison
                import re
                prompt_without = re.sub(r'\{([^:}]+):[^}]+\}', r'\1', prompt)
                
                yield None, "Generating without attention weights..."
                
                images_without, _ = concept_attention_manager.generate_attention_gradio(
                    prompt=prompt_without,
                    model_name=model_name,
                    seed=seed,
                    width=width,
                    height=height,
                    guidance=guidance,
                    steps=steps,
                    num_images=1,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize
                )
                
                comparison_images = []
                if images_with and images_without:
                    comparison_images = [images_with[0], images_without[0]]
                
                info = f"Left: With attention weights\nRight: Without attention weights\nSeed: {seed}"
                
                yield comparison_images, info
                
            except Exception as e:
                yield None, f"Error in comparison: {str(e)}"
        
        # Connect generation buttons
        generate_btn.click(
            fn=generate_with_attention,
            inputs=[
                prompt, model_name, seed, width, height, guidance, steps,
                num_images, save_metadata, low_ram_mode, quantize
            ],
            outputs=[output_gallery, status, generation_info]
        )
        
        compare_btn.click(
            fn=generate_comparison,
            inputs=[
                prompt, model_name, seed, width, height, guidance, steps,
                save_metadata, low_ram_mode, quantize
            ],
            outputs=[compare_gallery, compare_info]
        )
    
    return prompt, generate_btn
