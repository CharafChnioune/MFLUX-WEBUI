import gradio as gr
from backend.config_manager import get_config_manager
import json
from pathlib import Path

def create_config_tab():
    """Create the Configuration Management tab"""
    
    config_manager = get_config_manager()
    
    with gr.TabItem("Config"):
        gr.Markdown("""
        ## ⚙️ Configuration Manager
        Manage application settings, create presets, and export/import configurations.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Current Configuration Display
                gr.Markdown("### Current Configuration")
                
                config_display = gr.Code(
                    label="Current Settings",
                    value=json.dumps(config_manager.get_current_config(), indent=2),
                    language="json",
                    lines=15,
                    interactive=False
                )
                
                # Quick Settings
                gr.Markdown("### Quick Settings")
                
                with gr.Row():
                    default_model = gr.Dropdown(
                        label="Default Model",
                        choices=["dev", "schnell", "dev-4-bit", "schnell-4-bit"],
                        value=config_manager.get_config_value("generation.default_model", "dev")
                    )
                    
                    default_steps = gr.Number(
                        label="Default Steps",
                        value=config_manager.get_config_value("generation.default_steps", 25),
                        minimum=1,
                        maximum=100,
                        precision=0
                    )
                
                with gr.Row():
                    default_guidance = gr.Slider(
                        label="Default Guidance",
                        minimum=0.0,
                        maximum=20.0,
                        value=config_manager.get_config_value("generation.default_guidance", 7.0),
                        step=0.1
                    )
                    
                    low_ram_mode = gr.Checkbox(
                        label="Low RAM Mode",
                        value=config_manager.get_config_value("generation.low_ram_mode", False)
                    )
                
                with gr.Row():
                    save_metadata = gr.Checkbox(
                        label="Save Metadata",
                        value=config_manager.get_config_value("generation.save_metadata", True)
                    )
                    
                    auto_cleanup = gr.Checkbox(
                        label="Auto Cleanup",
                        value=config_manager.get_config_value("system.auto_cleanup", True)
                    )
                
                # System Settings
                gr.Markdown("### System Settings")
                
                output_directory = gr.Textbox(
                    label="Output Directory",
                    value=config_manager.get_config_value("system.output_directory", "output"),
                    info="Directory where generated images are saved"
                )
                
                lora_library_path = gr.Textbox(
                    label="LoRA Library Path",
                    value=config_manager.get_config_value("lora.library_path", "lora"),
                    info="Path to LoRA files directory"
                )
                
                max_loras = gr.Slider(
                    label="Max LoRAs",
                    minimum=1,
                    maximum=10,
                    value=config_manager.get_config_value("lora.max_loras", 5),
                    step=1,
                    info="Maximum number of LoRAs to load simultaneously"
                )
                
                # Apply Quick Settings
                apply_quick_btn = gr.Button("Apply Quick Settings", variant="primary")
            
            with gr.Column(scale=1):
                # Preset Management
                gr.Markdown("### Preset Management")
                
                preset_name = gr.Textbox(
                    label="Preset Name",
                    placeholder="My Custom Settings"
                )
                
                preset_description = gr.Textbox(
                    label="Description",
                    placeholder="Settings optimized for portraits",
                    lines=2
                )
                
                with gr.Row():
                    save_preset_btn = gr.Button("Save as Preset", variant="primary")
                    load_preset_btn = gr.Button("Load Preset", variant="secondary")
                
                available_presets = gr.Dropdown(
                    label="Available Presets",
                    choices=[p["name"] for p in config_manager.get_available_presets()],
                    info="Select a preset to load"
                )
                
                # File Import/Export
                gr.Markdown("### Import/Export")
                
                with gr.Row():
                    export_btn = gr.Button("Export Config", variant="secondary")
                    import_btn = gr.Button("Import Config", variant="secondary")
                
                config_file = gr.File(
                    label="Config File",
                    file_types=[".json", ".yaml", ".yml"],
                    type="filepath"
                )
                
                with gr.Row():
                    export_format = gr.Radio(
                        label="Export Format",
                        choices=["json", "yaml"],
                        value="json"
                    )
                    
                    include_comments = gr.Checkbox(
                        label="Include Comments",
                        value=True,
                        info="Include explanatory comments in export"
                    )
                
                # Template Export
                gr.Markdown("### Configuration Templates")
                
                export_template_btn = gr.Button("Export Template", variant="secondary")
                
                template_display = gr.Code(
                    label="Configuration Template",
                    language="json",
                    lines=10,
                    visible=False
                )
        
        # Status and validation
        with gr.Row():
            status_message = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3
            )
        
        # Event handlers
        def apply_quick_settings(model, steps, guidance, low_ram, metadata, cleanup, output_dir, lora_path, max_lora_count):
            try:
                # Update configuration
                config_manager.set_config_value("generation.default_model", model)
                config_manager.set_config_value("generation.default_steps", int(steps))
                config_manager.set_config_value("generation.default_guidance", float(guidance))
                config_manager.set_config_value("generation.low_ram_mode", low_ram)
                config_manager.set_config_value("generation.save_metadata", metadata)
                config_manager.set_config_value("system.auto_cleanup", cleanup)
                config_manager.set_config_value("system.output_directory", output_dir)
                config_manager.set_config_value("lora.library_path", lora_path)
                config_manager.set_config_value("lora.max_loras", int(max_lora_count))
                
                # Save as default
                config_manager.save_default_config()
                
                # Update display
                updated_config = json.dumps(config_manager.get_current_config(), indent=2)
                
                return updated_config, "Quick settings applied and saved successfully"
            except Exception as e:
                return config_display.value, f"Error applying settings: {str(e)}"
        
        def save_preset(name, description):
            try:
                if not name.strip():
                    return "Please enter a preset name", gr.update()
                
                preset_file = config_manager.create_preset_config(name, description)
                
                # Update available presets
                updated_presets = [p["name"] for p in config_manager.get_available_presets()]
                
                return f"Preset '{name}' saved successfully to {preset_file.name}", gr.update(choices=updated_presets)
            except Exception as e:
                return f"Error saving preset: {str(e)}", gr.update()
        
        def load_preset(preset_name):
            try:
                if not preset_name:
                    return config_display.value, "Please select a preset"
                
                preset_config = config_manager.load_preset_config(preset_name)
                config_manager.apply_config(preset_config)
                
                # Update display
                updated_config = json.dumps(config_manager.get_current_config(), indent=2)
                
                return updated_config, f"Preset '{preset_name}' loaded successfully"
            except Exception as e:
                return config_display.value, f"Error loading preset: {str(e)}"
        
        def export_config(format_type, include_comment):
            try:
                if include_comment:
                    template = config_manager.export_config_template()
                    return f"Configuration template exported:\n\n{template}"
                else:
                    config = config_manager.get_current_config()
                    if format_type == "yaml":
                        import yaml
                        exported = yaml.dump(config, default_flow_style=False, indent=2)
                    else:
                        exported = json.dumps(config, indent=2)
                    
                    return f"Configuration exported:\n\n{exported}"
            except Exception as e:
                return f"Error exporting config: {str(e)}"
        
        def import_config(file_path):
            try:
                if not file_path:
                    return config_display.value, "Please select a config file"
                
                imported_config = config_manager.load_config_file(file_path)
                
                # Validate configuration
                errors = config_manager.validate_config(imported_config)
                if errors:
                    return config_display.value, f"Configuration validation errors:\n" + "\n".join(errors)
                
                # Apply configuration
                config_manager.apply_config(imported_config)
                
                # Update display
                updated_config = json.dumps(config_manager.get_current_config(), indent=2)
                
                return updated_config, f"Configuration imported successfully from {Path(file_path).name}"
            except Exception as e:
                return config_display.value, f"Error importing config: {str(e)}"
        
        def export_template():
            try:
                template = config_manager.export_config_template()
                return gr.update(value=template, visible=True), "Template exported below"
            except Exception as e:
                return gr.update(visible=False), f"Error exporting template: {str(e)}"
        
        def refresh_config_display():
            return json.dumps(config_manager.get_current_config(), indent=2)
        
        # Wire up events
        apply_quick_btn.click(
            apply_quick_settings,
            inputs=[
                default_model, default_steps, default_guidance, low_ram_mode,
                save_metadata, auto_cleanup, output_directory, lora_library_path, max_loras
            ],
            outputs=[config_display, status_message]
        )
        
        save_preset_btn.click(
            save_preset,
            inputs=[preset_name, preset_description],
            outputs=[status_message, available_presets]
        )
        
        load_preset_btn.click(
            load_preset,
            inputs=[available_presets],
            outputs=[config_display, status_message]
        )
        
        export_btn.click(
            export_config,
            inputs=[export_format, include_comments],
            outputs=[status_message]
        )
        
        import_btn.click(
            import_config,
            inputs=[config_file],
            outputs=[config_display, status_message]
        )
        
        export_template_btn.click(
            export_template,
            outputs=[template_display, status_message]
        )
        
        # Auto-refresh config display when component loads
        # config_display.load(refresh_config_display, outputs=[config_display])  # Disabled - Code component doesn't have load method
