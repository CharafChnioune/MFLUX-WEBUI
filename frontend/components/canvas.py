import gradio as gr
import os
from pathlib import Path
from backend.flux_manager import (
    generate_image_gradio,
    generate_image_i2i_gradio,
    generate_image_kontext_gradio,
    generate_image_controlnet_gradio,
    generate_image_in_context_lora_gradio,
    get_random_seed,
)
from backend.model_manager import get_updated_models, get_base_model_choices
from backend.lora_manager import (
    get_lora_choices,
    MAX_LORAS,
    process_lora_files,
    update_lora_scales,
)
from backend.prompts_manager import enhance_prompt
from frontend.components.llmsettings import create_llm_settings

def get_controlnet_choices():
    return ["canny", "depth", "openpose", "normal", "scribble"]

def create_canvas_tab():
    """Create the integrated Canvas workspace tab"""
    
    with gr.Column():
        gr.Markdown("# 🎨 Canvas - Integrated Design Workspace")
        gr.Markdown("*Generate, edit, and enhance images in one streamlined workflow*")
        
        # Main workspace layout
        with gr.Row():
            # Left panel - Tools and controls
            with gr.Column(scale=1):
                
                # Tool selector
                with gr.Group():
                    gr.Markdown("### 🛠️ Select Tool")
                    tool_selector = gr.Dropdown(
                        choices=[
                            "🎯 Advanced Generate",
                            "🖼️ Image-to-Image", 
                            "🎭 Kontext Edit",
                            "🎮 ControlNet",
                            "🎨 In-Context LoRA"
                        ],
                        label="Active Tool",
                        value="🎯 Advanced Generate",
                        interactive=True
                    )
                
                # Dynamic tool interface
                with gr.Group():
                    gr.Markdown("### ⚙️ Tool Settings")
                    
                    # Common settings (always visible)
                    with gr.Group():
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe your image...",
                            lines=3,
                            value=""
                        )
                        
                        # LLM Enhancement section
                        with gr.Accordion("⚙️ LLM Settings", open=False) as llm_section:
                            llm_components = create_llm_settings(tab_name="canvas", parent_accordion=llm_section)
                        
                        with gr.Row():
                            enhance_prompt_btn = gr.Button("🔮 Enhance prompt with LLM")
                        
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
                                minimum=0,
                                maximum=20,
                                value=7.0,
                                step=0.1,
                                label="Guidance"
                            )
                    
                    # Tool-specific settings (conditional visibility)
                    
                    # Advanced Generate specific
                    advanced_settings = gr.Group(visible=True)
                    with advanced_settings:
                        model = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="dev",
                            interactive=True
                        )
                        
                        with gr.Row():
                            lora_files = gr.Dropdown(
                                choices=get_lora_choices(),
                                label="LoRA Files",
                                value=[],
                                interactive=True,
                                multiselect=True,
                                allow_custom_value=True,
                                scale=9,
                            )
                            refresh_loras = gr.Button(
                                "🔄",
                                variant="tool",
                                size="sm",
                                scale=1,
                                min_width=30,
                                elem_classes="refresh-button",
                            )
                        refresh_loras.click(
                            lambda: gr.update(choices=get_lora_choices()),
                            outputs=[lora_files],
                        )
                        lora_scales = [
                            gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                step=0.01,
                                label=f"LoRA Weight {idx + 1}",
                                value=1.0,
                                visible=False,
                            )
                            for idx in range(MAX_LORAS)
                        ]
                        lora_files.change(
                            fn=update_lora_scales,
                            inputs=[lora_files],
                            outputs=lora_scales,
                        )
                    
                    # Image-to-Image specific
                    i2i_settings = gr.Group(visible=False)
                    with i2i_settings:
                        i2i_input_image = gr.Image(
                            label="Input Image",
                            type="filepath",
                            height=200
                        )
                        i2i_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.1,
                            label="Denoising Strength"
                        )
                    
                    # Kontext specific
                    kontext_settings = gr.Group(visible=False)
                    with kontext_settings:
                        kontext_reference = gr.Image(
                            label="Reference Image (Required)",
                            type="filepath",
                            height=200
                        )
                        gr.Markdown("*Uses dev-kontext model automatically*")
                    
                    # ControlNet specific
                    controlnet_settings = gr.Group(visible=False)
                    with controlnet_settings:
                        controlnet_input = gr.Image(
                            label="ControlNet Input",
                            type="filepath",
                            height=200
                        )
                        controlnet_type = gr.Dropdown(
                            choices=get_controlnet_choices(),
                            label="ControlNet Type",
                            value="canny"
                        )
                        controlnet_strength = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="ControlNet Strength"
                        )
                    
                    # In-Context LoRA specific
                    in_context_settings = gr.Group(visible=False)
                    with in_context_settings:
                        in_context_image = gr.Image(
                            label="Style Reference Image",
                            type="filepath",
                            height=200,
                        )
                        in_context_style = gr.Dropdown(
                            choices=[
                                "identity",
                                "portrait",
                                "illustration",
                                "couple",
                                "storyboard",
                                "font",
                                "home",
                                "ppt",
                                "sandstorm",
                                "sparklers",
                            ],
                            value="identity",
                            label="In-Context LoRA Style",
                        )
                
                # MFLUX v0.9.0 Advanced Options
                with gr.Accordion("🔧 Advanced Options (MFLUX v0.9.0)", open=False):
                    base_model = gr.Dropdown(
                        choices=["Auto"] + get_base_model_choices(),
                        label="Base Model (--base-model)",
                        value="Auto",
                        info="Select the architecture for third-party checkpoints.",
                    )
                    
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
                
                # Generation settings
                with gr.Group():
                    gr.Markdown("### 🎯 Generation")
                    
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
                    
                    metadata = gr.Checkbox(
                        label="Save Metadata",
                        value=True
                    )
                    
                    # Generate button
                    generate_btn = gr.Button(
                        "🎨 Generate/Process",
                        variant="primary",
                        size="lg"
                    )
            
            # Right panel - Canvas and results
            with gr.Column(scale=2):
                
                # Canvas area
                with gr.Group():
                    gr.Markdown("### 🖼️ Canvas")
                    
                    # Image display with workflow buttons
                    current_image = gr.Image(
                        label="Current Image",
                        type="filepath",
                        height=400,
                        interactive=False
                    )
                    
                    # Quick action buttons
                    with gr.Row():
                        use_as_i2i_btn = gr.Button("➡️ Use as Image-to-Image", size="sm")
                        use_as_kontext_btn = gr.Button("➡️ Use for Kontext Edit", size="sm")
                        use_as_controlnet_btn = gr.Button("➡️ Use for ControlNet", size="sm")
                        use_as_reference_btn = gr.Button("➡️ Use as Reference", size="sm")
                
                # Image gallery
                with gr.Group():
                    gr.Markdown("### 🎭 Generated Images")
                    output_gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=False,
                        elem_id="canvas-gallery",
                        columns=2,
                        rows=2,
                        object_fit="contain",
                        height="auto"
                    )
                
                # Generation info
                with gr.Group():
                    gr.Markdown("### ℹ️ Generation Info")
                    output_info = gr.Textbox(
                        label="Generation Details",
                        lines=3,
                        interactive=False
                    )
                
                # Workflow history
                with gr.Accordion("📋 Workflow History", open=False):
                    workflow_history = gr.Textbox(
                        label="Workflow Steps",
                        lines=5,
                        interactive=False,
                        value="Welcome to Canvas! Start by generating an image."
                    )
    
    # Event handlers
    def update_seed():
        return get_random_seed()
    
    def update_tool_interface(selected_tool):
        """Update interface based on selected tool"""
        updates = {
            "advanced_settings": gr.update(visible=False),
            "i2i_settings": gr.update(visible=False),
            "kontext_settings": gr.update(visible=False),
            "controlnet_settings": gr.update(visible=False),
            "in_context_settings": gr.update(visible=False)
        }
        
        if "Advanced Generate" in selected_tool:
            updates["advanced_settings"] = gr.update(visible=True)
        elif "Image-to-Image" in selected_tool:
            updates["i2i_settings"] = gr.update(visible=True)
        elif "Kontext Edit" in selected_tool:
            updates["kontext_settings"] = gr.update(visible=True)
        elif "ControlNet" in selected_tool:
            updates["controlnet_settings"] = gr.update(visible=True)
        elif "In-Context LoRA" in selected_tool:
            updates["in_context_settings"] = gr.update(visible=True)
        
        return list(updates.values())
    
    def use_image_as_input(image_path, target_tool):
        """Use current image as input for specified tool"""
        if not image_path:
            return None, f"No image selected"
        
        # Switch to target tool and set image
        tool_updates = update_tool_interface(target_tool)
        return image_path, f"Switched to {target_tool} with current image"
    
    def generate_with_canvas(
        tool,
        prompt,
        seed,
        width,
        height,
        steps,
        guidance,
        model,
        lora_files,
        *remaining_inputs,
    ):
        """Generate images based on selected tool"""

        lora_scale_values = remaining_inputs[:MAX_LORAS]
        (
            i2i_input,
            i2i_strength,
            kontext_ref,
            controlnet_input,
            controlnet_type,
            controlnet_strength,
            in_context_img,
            in_context_style,
            base_model,
            prompt_file,
            config_from_metadata,
            stepwise_output_dir,
            vae_tiling,
            vae_tiling_split,
            num_images,
            low_ram,
            metadata,
            current_workflow,
        ) = remaining_inputs[MAX_LORAS:]

        def _file_to_path(value):
            if not value:
                return ""
            if isinstance(value, dict) and "name" in value:
                return value["name"]
            return getattr(value, "name", value)

        prompt_file_path = _file_to_path(prompt_file)
        metadata_file_path = _file_to_path(config_from_metadata)
        normalized_base_model = (
            None if base_model in (None, "", "None", "Auto") else base_model
        )
        num_images = int(num_images) if num_images else 1
        metadata_flag = bool(metadata)
        low_ram = bool(low_ram)
        seed_num = None
        if isinstance(seed, (int, float)) and seed >= 0:
            seed_num = int(seed)
        seed_text = "" if seed_num is None else str(seed_num)

        valid_loras = process_lora_files(lora_files) if lora_files else None
        if valid_loras:
            clean_scales = []
            for idx in range(len(valid_loras)):
                try:
                    clean_scales.append(float(lora_scale_values[idx]))
                except (ValueError, TypeError, IndexError):
                    clean_scales.append(1.0)
        else:
            clean_scales = []

        # Update workflow history
        new_step = f"🎨 {tool} - {prompt[:50]}..." if len(prompt) > 50 else f"🎨 {tool} - {prompt}"
        updated_workflow = f"{current_workflow}\n{new_step}"
        
        try:
            if "Advanced Generate" in tool:
                # Use advanced generate
                result = generate_image_gradio(
                    prompt,
                    model,
                    normalized_base_model,
                    seed_text,
                    width,
                    height,
                    steps,
                    guidance,
                    valid_loras,
                    metadata_flag,
                    None,
                    None,
                    prompt_file_path,
                    metadata_file_path,
                    stepwise_output_dir,
                    vae_tiling,
                    vae_tiling_split,
                    *clean_scales,
                    num_images=num_images,
                    low_ram=low_ram,
                )
                
            elif "Image-to-Image" in tool:
                if not i2i_input:
                    return [], "Error: No input image for Image-to-Image", prompt, updated_workflow, None
                
                result = generate_image_i2i_gradio(
                    prompt,
                    i2i_input,
                    model,
                    normalized_base_model,
                    seed_text,
                    height,
                    width,
                    steps,
                    guidance,
                    i2i_strength,
                    valid_loras,
                    metadata_flag,
                    prompt_file_path,
                    metadata_file_path,
                    stepwise_output_dir,
                    vae_tiling,
                    vae_tiling_split,
                    *clean_scales,
                    num_images=num_images,
                    low_ram=low_ram,
                )
                
            elif "Kontext Edit" in tool:
                if not kontext_ref:
                    return [], "Error: No reference image for Kontext", prompt, updated_workflow, None
                
                result = generate_image_kontext_gradio(
                    prompt,
                    kontext_ref,
                    model,
                    seed_num,
                    width,
                    height,
                    steps,
                    guidance,
                    valid_loras,
                    metadata_flag,
                    *clean_scales,
                    num_images=num_images,
                    low_ram=low_ram,
                )
                
            elif "ControlNet" in tool:
                if not controlnet_input:
                    return [], "Error: No input image for ControlNet", prompt, updated_workflow, None
                
                result = generate_image_controlnet_gradio(
                    prompt,
                    controlnet_input,
                    model,
                    normalized_base_model,
                    seed_text,
                    height,
                    width,
                    steps,
                    guidance,
                    controlnet_strength,
                    valid_loras,
                    metadata_flag,
                    False,
                    prompt_file_path,
                    metadata_file_path,
                    stepwise_output_dir,
                    vae_tiling,
                    vae_tiling_split,
                    *clean_scales,
                    num_images=num_images,
                    low_ram=low_ram,
                )
                
            elif "In-Context LoRA" in tool:
                if not in_context_img:
                    return [], "Error: No reference image for In-Context LoRA", prompt, updated_workflow, None
                
                result = generate_image_in_context_lora_gradio(
                    prompt,
                    in_context_img,
                    model,
                    normalized_base_model,
                    seed_text,
                    height,
                    width,
                    steps,
                    guidance,
                    in_context_style,
                    valid_loras,
                    metadata_flag,
                    prompt_file_path,
                    metadata_file_path,
                    stepwise_output_dir,
                    vae_tiling,
                    vae_tiling_split,
                    *clean_scales,
                    num_images=num_images,
                    low_ram=low_ram,
                )
            
            else:
                return [], "Error: Unknown tool selected", prompt, updated_workflow, None
            
            # Extract results
            images, info, updated_prompt = result
            
            # Set current image to first generated image
            current_img = images[0] if images else None
            
            return images, info, updated_prompt, updated_workflow, current_img
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            return [], error_msg, prompt, updated_workflow, None
    
    # Bind events
    random_seed_btn.click(update_seed, outputs=seed)
    
    # Tool selector updates interface
    tool_selector.change(
        update_tool_interface,
        inputs=tool_selector,
        outputs=[advanced_settings, i2i_settings, kontext_settings, controlnet_settings, in_context_settings]
    )
    
    # Quick action buttons
    use_as_i2i_btn.click(
        lambda img: (img, "🖼️ Image-to-Image"),
        inputs=current_image,
        outputs=[i2i_input_image, tool_selector]
    )
    
    use_as_kontext_btn.click(
        lambda img: (img, "🎭 Kontext Edit"),
        inputs=current_image,
        outputs=[kontext_reference, tool_selector]
    )
    
    use_as_controlnet_btn.click(
        lambda img: (img, "🎮 ControlNet"),
        inputs=current_image,
        outputs=[controlnet_input, tool_selector]
    )
    
    use_as_reference_btn.click(
        lambda img: (img, "🎨 In-Context LoRA"),
        inputs=current_image,
        outputs=[in_context_image, tool_selector]
    )
    
    # Generate button
    generate_btn.click(
        generate_with_canvas,
        inputs=[
            tool_selector, prompt, seed, width, height, steps, guidance,
            model, lora_files, *lora_scales,
            i2i_input_image, i2i_strength,
            kontext_reference,
            controlnet_input, controlnet_type, controlnet_strength,
            in_context_image, in_context_style,
            base_model, prompt_file, config_from_metadata, stepwise_output_dir,
            vae_tiling, vae_tiling_split, num_images, low_ram, metadata,
            workflow_history
        ],
        outputs=[output_gallery, output_info, prompt, workflow_history, current_image]
    )
    
    # Enhance prompt with LLM
    def canvas_enhance_prompt(p, t, m1, m2, sp, current_tool, current_img):
        """Enhanced prompt function for Canvas with context awareness"""
        try:
            # Get current reference image based on selected tool
            context_image = None
            if "Image-to-Image" in current_tool and current_img:
                context_image = current_img
            elif "Kontext Edit" in current_tool and current_img:
                context_image = current_img
            elif "ControlNet" in current_tool and current_img:
                context_image = current_img
            elif "In-Context LoRA" in current_tool and current_img:
                context_image = current_img
                
            return enhance_prompt(p, t, m1, m2, sp, context_image, tab_name="canvas")
        except Exception as e:
            print(f"Error enhancing prompt in canvas: {str(e)}")
            return p
    
    enhance_prompt_btn.click(
        canvas_enhance_prompt,
        inputs=[
            prompt,
            llm_components[0],  # LLM type
            llm_components[1],  # Ollama model
            llm_components[4],  # MLX model
            llm_components[2],  # System prompt
            tool_selector,      # Current tool
            current_image       # Current image for context
        ],
        outputs=prompt
    )
    
    return {
        'canvas': current_image,
        'gallery': output_gallery,
        'info': output_info,
        'workflow': workflow_history
    }
