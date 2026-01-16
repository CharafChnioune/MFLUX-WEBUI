import gradio as gr
from backend.upscale_manager import upscale_manager, upscale_image_gradio, batch_upscale_gradio, upscale_with_custom_dimensions_gradio
from backend.seedvr2_manager import generate_seedvr2_upscale
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

def create_upscale_tab():
    """Create the Upscale tool tab for image upscaling"""
    
    with gr.TabItem("Upscale"):
        gr.Markdown("""
        ## 🔎 Upscale Tool
        Intelligently increase the resolution of your images while maintaining or enhancing quality and details.
        Uses AI-powered upscaling to produce sharp, detailed results.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image for better upscaling...",
                    lines=2,
                    info="Describing the image helps preserve details during upscaling"
                )
                
                # LLM Enhancement section
                with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                    llm_components = create_llm_settings(tab_name="upscale", parent_accordion=llm_section)
                
                with gr.Row():
                    enhance_prompt_btn = gr.Button("🔮 Enhance prompt with LLM")
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["seedvr2", "schnell", "dev"],
                    value="dev",
                    info="SeedVR2 for fidelity 1-step upscaling; Dev for ControlNet upscaler"
                )
                
                # Upscale settings
                with gr.Row():
                    scale_factor = gr.Textbox(
                        label="Scale Factor / Resolution",
                        value="2x",
                        placeholder="2x, 1024",
                        info="For SeedVR2: shortest edge or Nx; for ControlNet: scale factor"
                    )
                    
                    custom_resolution = gr.Checkbox(
                        label="Mixed Dimensions",
                        value=False,
                        info="ControlNet/PIL only: combine scale factors and absolute values"
                    )
                
                with gr.Row(visible=False) as custom_res_row:
                    target_width = gr.Textbox(
                        label="Width",
                        value="auto",
                        placeholder="2x, 1024, auto",
                        info="Scale factor or absolute pixels"
                    )
                    target_height = gr.Textbox(
                        label="Height",
                        value="auto", 
                        placeholder="2x, 1024, auto",
                        info="Scale factor or absolute pixels"
                    )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False) as adv_opts:
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
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=7.5
                    )
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28
                    )
                    
                    controlnet_strength = gr.Slider(
                        label="ControlNet Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.6,
                        info="Recommended: 0.5-0.7 for best results (ControlNet only)"
                    )
                    
                    vae_tiling = gr.Checkbox(
                        label="VAE Tiling",
                        value=False,
                        info="Enable for high resolution (reduces memory usage)"
                    )
                    
                    vae_tiling_split = gr.Radio(
                        label="VAE Tiling Split Direction",
                        choices=["horizontal", "vertical"],
                        value="horizontal",
                        visible=False
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
                        info="8-bit recommended for memory efficiency"
                    )

                with gr.Accordion("SeedVR2 Options", open=False) as seedvr2_opts:
                    seedvr2_softness = gr.Slider(
                        label="Softness",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.0,
                        info="Pre-downsample factor (0 = off, 1 = max smoothing)"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button("Upscale Image", variant="primary")
                    batch_btn = gr.Button("Batch Upscale", variant="secondary")
            
            with gr.Column(scale=1):
                # Output section
                output_image = gr.Image(
                    label="Upscaled Image",
                    type="pil",
                    height=600
                )
                
                status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Accordion("Resolution Info", open=True):
                    resolution_info = gr.Textbox(
                        label="Resolution Details",
                        interactive=False,
                        lines=3
                    )
        
        # Batch upscale section
        with gr.Row(visible=False) as batch_section:
            with gr.Column():
                gr.Markdown("### Batch Upscale")
                batch_images = gr.File(
                    label="Upload Multiple Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_prompt = gr.Textbox(
                    label="Batch Prompt (Optional)",
                    placeholder="General description for all images...",
                    lines=2
                )
                
                # LLM Enhancement for batch
                with gr.Accordion("⚙️ Batch LLM Settings", open=False) as batch_llm_section:
                    batch_llm_components = create_llm_settings(tab_name="upscale-batch", parent_accordion=batch_llm_section)
                
                with gr.Row():
                    enhance_batch_prompt_btn = gr.Button("🔮 Enhance batch prompt with LLM")
                batch_generate_btn = gr.Button("Start Batch Upscale", variant="primary")
                batch_status = gr.Textbox(label="Batch Status", interactive=False)
        
        # Help section
        with gr.Accordion("📖 How to use", open=False):
            gr.Markdown("""
            ### Upscale Tool Guide
            
            **Single Image Upscale:**
            1. **Upload an image** - The image you want to upscale
            2. **Add a prompt** - Describe the image to help preserve details
            3. **Choose scale factor** - 2x, 3x, or 4x resolution increase
            4. **Adjust settings** - Fine-tune the upscaling parameters
            5. **Click Upscale** - Generate the high-resolution version
            
            **Batch Upscale:**
            1. Click "Batch Upscale" to show batch options
            2. Upload multiple images at once
            3. Optionally add a general prompt
            4. All images will be upscaled with the same settings
            
            **Tips for Best Results:**
            - **Prompting matters**: Accurately describe your image for better detail preservation
            - **ControlNet Strength**: Keep between 0.5-0.7 for optimal results
            - **VAE Tiling**: Enable for very high resolutions (>2048px) to avoid memory issues
            - **Scale Factor**: 2x usually gives best quality, higher factors may introduce artifacts
            - **Dev model**: Generally produces better quality than Schnell for upscaling
            
            **Memory Management:**
            - Enable "Low RAM Mode" if you encounter memory errors
            - Use 8-bit quantization for good balance of quality and memory usage
            - Enable "VAE Tiling" for high resolution outputs
            
            **Common Use Cases:**
            - Enhance low-resolution photos
            - Prepare images for printing
            - Improve AI-generated images
            - Restore old or compressed images
            """)
        
        # Event handlers
        def toggle_custom_resolution(use_custom):
            return gr.update(visible=use_custom)
        
        def toggle_vae_split(use_tiling):
            return gr.update(visible=use_tiling)
        
        def show_batch_section():
            return gr.update(visible=True)

        def parse_scale_value(raw_value):
            """
            Parse scale factor inputs like '2x', '1.5', or numeric values.
            Falls back to 1.0 on bad input to avoid UI crashes.
            """
            try:
                if isinstance(raw_value, str):
                    val = raw_value.strip().lower()
                    if val.endswith("x"):
                        val = val[:-1]
                    return float(val)
                return float(raw_value)
            except Exception:
                return 1.0
        
        def calculate_dimensions(image, scale_factor, use_custom, custom_w, custom_h):
            if image is None:
                return "No image uploaded"
            
            orig_w, orig_h = image.size
            
            if use_custom:
                try:
                    new_w, new_h = int(custom_w), int(custom_h)
                except Exception:
                    return "Invalid custom width/height"
            else:
                try:
                    sf = parse_scale_value(scale_factor)
                except Exception:
                    return "Invalid scale factor"
                new_w = int(orig_w * sf)
                new_h = int(orig_h * sf)
            
            info = f"Original: {orig_w}x{orig_h}\n"
            info += f"Target: {new_w}x{new_h}\n"
            try:
                info += f"Scale: {new_w/float(orig_w):.2f}x"
            except Exception:
                info += "Scale: n/a"
            
            return info
        
        def random_seed():
            import random
            return random.randint(0, 2**32-1)
        
        # Wire up events
        custom_resolution.change(
            fn=toggle_custom_resolution,
            inputs=[custom_resolution],
            outputs=[custom_res_row]
        )
        
        vae_tiling.change(
            fn=toggle_vae_split,
            inputs=[vae_tiling],
            outputs=[vae_tiling_split]
        )
        
        batch_btn.click(
            fn=show_batch_section,
            outputs=[batch_section]
        )
        
        random_seed_btn.click(fn=random_seed, outputs=[seed])
        
        # Update resolution info when inputs change (non-SeedVR2 paths)
        for inp in [input_image, scale_factor, custom_resolution, target_width, target_height]:
            inp.change(
                fn=calculate_dimensions,
                inputs=[input_image, scale_factor, custom_resolution, target_width, target_height],
                outputs=[resolution_info]
            )
        
        # Single image upscale function
        def upscale_single_image(input_image, prompt, model_name, scale_factor, use_custom,
                               target_width, target_height, seed, guidance, steps,
                               controlnet_strength, vae_tiling, vae_tiling_split,
                               save_metadata, low_ram_mode, quantize, seedvr2_softness):
            if input_image is None:
                return None, "Please upload an image to upscale", ""
            
            if model_name != "seedvr2" and not prompt:
                return None, "Please describe the image for better upscaling", ""
            
            try:
                if model_name == "seedvr2":
                    # SeedVR2 path uses shortest-edge resolution or scale factor
                    tmp_path = input_image if isinstance(input_image, str) else None
                    created_tmp = False
                    if not tmp_path:
                        # Save PIL to temp file for backend (keeps Gradio wiring simple)
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            input_image.save(tmp.name)
                            tmp_path = tmp.name
                            created_tmp = True
                    status_msg = f"SeedVR2 upscaling to {scale_factor} (softness {seedvr2_softness})..."
                    yield None, status_msg, resolution_info.value
                    upscaled_image, status_text, _ = generate_seedvr2_upscale(
                        image_path=tmp_path,
                        resolution=scale_factor,
                        softness=seedvr2_softness,
                        seed=seed,
                        metadata=save_metadata,
                        progress=gr.Progress(track_tqdm=False)
                    )
                    yield upscaled_image, status_text, resolution_info.value
                    if created_tmp:
                        import os
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                else:
                    # Calculate target dimensions
                    if use_custom:
                        width, height = int(target_width), int(target_height)
                    else:
                        orig_w, orig_h = input_image.size
                        sf = parse_scale_value(scale_factor)
                        width = int(orig_w * sf)
                        height = int(orig_h * sf)
                    
                    status_msg = f"Upscaling to {width}x{height}..."
                    yield None, status_msg, resolution_info.value
                    
                    upscaled_image, metadata = upscale_manager.upscale_single_gradio(
                        input_image=input_image,
                        prompt=prompt,
                        model_name=model_name,
                        target_width=width,
                        target_height=height,
                        seed=seed,
                        guidance=guidance,
                        steps=steps,
                        controlnet_strength=controlnet_strength,
                        vae_tiling=vae_tiling,
                        vae_tiling_split=vae_tiling_split,
                        save_metadata=save_metadata,
                        low_ram_mode=low_ram_mode,
                        quantize=quantize
                    )
                    
                    yield upscaled_image, "Upscaling complete!", resolution_info.value
                
            except Exception as e:
                yield None, f"Error: {str(e)}", resolution_info.value
        
        # Batch upscale function
        def upscale_batch_images(batch_images, batch_prompt, model_name, scale_factor,
                               seed, guidance, steps, controlnet_strength,
                               vae_tiling, vae_tiling_split, save_metadata,
                               low_ram_mode, quantize):
            if not batch_images:
                return "Please upload images for batch upscaling"
            
            if model_name == "seedvr2":
                return "Batch upscaling is not available for SeedVR2 yet"
            
            try:
                total = len(batch_images)
                yield f"Starting batch upscale of {total} images..."
                
                results = upscale_manager.upscale_batch_gradio(
                    image_files=batch_images,
                    prompt=batch_prompt,
                    model_name=model_name,
                    scale_factor=scale_factor,
                    seed=seed,
                    guidance=guidance,
                    steps=steps,
                    controlnet_strength=controlnet_strength,
                    vae_tiling=vae_tiling,
                    vae_tiling_split=vae_tiling_split,
                    save_metadata=save_metadata,
                    low_ram_mode=low_ram_mode,
                    quantize=quantize,
                    progress_callback=lambda i: f"Processing image {i+1}/{total}..."
                )
                
                yield f"Batch upscale complete! Processed {len(results)} images."
                
            except Exception as e:
                yield f"Error in batch processing: {str(e)}"
        
        # Enhance prompt with LLM for upscale
        def upscale_enhance_prompt(p, t, m1, m2, sp, input_img):
            """Enhanced prompt function for Upscale with input image context"""
            try:
                return enhance_prompt(p, t, m1, m2, sp, input_img, tab_name="upscale")
            except Exception as e:
                print(f"Error enhancing prompt in upscale: {str(e)}")
                return p
        
        def upscale_enhance_batch_prompt(p, t, m1, m2, sp):
            """Enhanced prompt function for batch upscale"""
            try:
                return enhance_prompt(p, t, m1, m2, sp, None, tab_name="upscale-batch")
            except Exception as e:
                print(f"Error enhancing batch prompt in upscale: {str(e)}")
                return p
        
        enhance_prompt_btn.click(
            upscale_enhance_prompt,
            inputs=[
                prompt,
                llm_components[0],  # LLM type
                llm_components[1],  # Ollama model
                llm_components[4],  # MLX model
                llm_components[2],  # System prompt
                input_image         # Input image for context
            ],
            outputs=prompt
        )
        
        enhance_batch_prompt_btn.click(
            upscale_enhance_batch_prompt,
            inputs=[
                batch_prompt,
                batch_llm_components[0],  # LLM type
                batch_llm_components[1],  # Ollama model
                batch_llm_components[4],  # MLX model
                batch_llm_components[2],  # System prompt
            ],
            outputs=batch_prompt
        )
        
        # Connect generation buttons
        generate_btn.click(
            fn=upscale_single_image,
            inputs=[
                input_image, prompt, model_name, scale_factor, custom_resolution,
                target_width, target_height, seed, guidance, steps,
                controlnet_strength, vae_tiling, vae_tiling_split,
                save_metadata, low_ram_mode, quantize, seedvr2_softness
            ],
            outputs=[output_image, status, resolution_info]
        )
        
        batch_generate_btn.click(
            fn=upscale_batch_images,
            inputs=[
                batch_images, batch_prompt, model_name, scale_factor,
                seed, guidance, steps, controlnet_strength,
                vae_tiling, vae_tiling_split, save_metadata,
                low_ram_mode, quantize
            ],
            outputs=[batch_status]
        )
    
    return input_image, prompt, generate_btn
