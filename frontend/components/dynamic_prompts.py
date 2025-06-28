import gradio as gr
from backend.dynamic_prompts_manager import get_dynamic_prompts_manager
import json

def create_dynamic_prompts_tab():
    """Create the Dynamic Prompts management tab"""
    
    prompts_manager = get_dynamic_prompts_manager()
    
    with gr.TabItem("Dynamic Prompts"):
        gr.Markdown("""
        ## 🎯 Dynamic Prompts Manager
        Create and manage dynamic prompts with wildcards and variations for automated prompt generation.
        
        **Syntax:** Use `[option1|option2|option3]` for random selection
        
        **Example:** `A [beautiful|stunning|gorgeous] [cat|dog|bird] in a [garden|park|forest]`
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Configuration
                gr.Markdown("### Configuration")
                
                enabled_checkbox = gr.Checkbox(
                    label="Enable Dynamic Prompts",
                    value=prompts_manager.get_config()["enabled"],
                    info="Process dynamic prompts with wildcards"
                )
                
                random_checkbox = gr.Checkbox(
                    label="Random Selection",
                    value=prompts_manager.get_config()["random_selection"],
                    info="Use random selection vs first option"
                )
                
                max_variations = gr.Slider(
                    label="Max Variations",
                    minimum=1,
                    maximum=100,
                    value=prompts_manager.get_config()["max_variations"],
                    step=1,
                    info="Maximum number of variations to generate"
                )
                
                # Prompt Testing
                gr.Markdown("### Test Dynamic Prompts")
                
                test_prompt = gr.Textbox(
                    label="Test Prompt",
                    placeholder="A [beautiful|stunning|gorgeous] [cat|dog|bird]",
                    lines=3,
                    info="Enter a dynamic prompt to test"
                )
                
                with gr.Row():
                    test_btn = gr.Button("Test Prompt", variant="primary")
                    generate_variations_btn = gr.Button("Show All Variations", variant="secondary")
                
                test_result = gr.Textbox(
                    label="Generated Prompt",
                    interactive=False,
                    lines=2
                )
                
                variations_display = gr.Textbox(
                    label="All Variations",
                    interactive=False,
                    lines=10,
                    visible=False
                )
            
            with gr.Column(scale=1):
                # Prompt File Management
                gr.Markdown("### Prompt File Management")
                
                with gr.Row():
                    file_format = gr.Radio(
                        label="File Format",
                        choices=["txt", "json"],
                        value="txt",
                        info="Format for saving/loading prompts"
                    )
                    
                    file_name = gr.Textbox(
                        label="File Name",
                        placeholder="my_prompts",
                        info="Name without extension"
                    )
                
                prompts_text = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=10,
                    placeholder="A beautiful landscape\nA [cute|adorable] animal\nA [modern|vintage] building"
                )
                
                with gr.Row():
                    save_file_btn = gr.Button("Save to File", variant="primary")
                    load_file_btn = gr.Button("Load from File", variant="secondary")
                
                # Available Files
                available_files = gr.Dropdown(
                    label="Available Prompt Files",
                    choices=prompts_manager.get_available_prompt_files(),
                    info="Select a file to load"
                )
                
                load_selected_btn = gr.Button("Load Selected File")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Categories Management
                gr.Markdown("### Categories Management")
                
                category_select = gr.Dropdown(
                    label="Category",
                    choices=list(prompts_manager.get_config()["categories"].keys()),
                    value="styles",
                    info="Select category to manage"
                )
                
                category_prompts = gr.Textbox(
                    label="Category Prompts (one per line)",
                    lines=8,
                    placeholder="hyperrealistic\noil painting\nwatercolor\ndigital art"
                )
                
                with gr.Row():
                    add_to_category_btn = gr.Button("Add to Category", variant="primary")
                    remove_from_category_btn = gr.Button("Remove from Category", variant="stop")
                    clear_category_btn = gr.Button("Clear Category", variant="stop")
                
                # Category Preview
                category_preview = gr.Textbox(
                    label="Current Category Contents",
                    interactive=False,
                    lines=6
                )
            
            with gr.Column(scale=1):
                # Template Management
                gr.Markdown("### Prompt Templates")
                
                template_name = gr.Textbox(
                    label="Template Name",
                    placeholder="Portrait Template"
                )
                
                template_description = gr.Textbox(
                    label="Description",
                    placeholder="Template for portrait generation",
                    lines=2
                )
                
                template_content = gr.Textbox(
                    label="Template Content",
                    placeholder="[Realistic|Artistic] portrait of a [young|old] [man|woman]",
                    lines=4
                )
                
                with gr.Row():
                    save_template_btn = gr.Button("Save Template", variant="primary")
                    load_template_btn = gr.Button("Load Template", variant="secondary")
                
                available_templates = gr.Dropdown(
                    label="Available Templates",
                    choices=[t.get("name", "Unnamed") for t in prompts_manager.get_available_templates()],
                    info="Select a template to load"
                )
        
        # Status messages
        status_message = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2
        )
        
        # Event handlers
        def update_config(enabled, random_sel, max_var):
            try:
                prompts_manager.update_config(
                    enabled=enabled,
                    random_selection=random_sel,
                    max_variations=int(max_var)
                )
                return "Configuration updated successfully"
            except Exception as e:
                return f"Error updating configuration: {str(e)}"
        
        def test_dynamic_prompt(prompt):
            try:
                if not prompt.strip():
                    return "Please enter a prompt to test"
                
                result = prompts_manager.get_random_prompt_variation(prompt)
                return result
            except Exception as e:
                return f"Error testing prompt: {str(e)}"
        
        def show_all_variations(prompt):
            try:
                if not prompt.strip():
                    return "Please enter a prompt", gr.update(visible=False)
                
                variations = prompts_manager.parse_dynamic_prompt(prompt)
                variations_text = "\n".join(f"{i+1}. {var}" for i, var in enumerate(variations))
                
                return f"Generated {len(variations)} variations:", gr.update(value=variations_text, visible=True)
            except Exception as e:
                return f"Error generating variations: {str(e)}", gr.update(visible=False)
        
        def save_prompts_to_file(format_type, filename, prompts):
            try:
                if not filename.strip():
                    return "Please enter a filename"
                
                if not prompts.strip():
                    return "Please enter prompts to save"
                
                prompts_list = [p.strip() for p in prompts.split('\n') if p.strip()]
                file_path = prompts_manager.prompts_dir / f"{filename}.{format_type}"
                
                prompts_manager.save_prompt_file(prompts_list, file_path, format_type)
                
                # Refresh available files
                return f"Saved {len(prompts_list)} prompts to {file_path.name}"
            except Exception as e:
                return f"Error saving file: {str(e)}"
        
        def load_prompts_from_file(filename):
            try:
                if not filename:
                    return "Please select a file", ""
                
                prompts_list = prompts_manager.load_prompt_file(filename)
                prompts_text = "\n".join(prompts_list)
                
                return f"Loaded {len(prompts_list)} prompts from file", prompts_text
            except Exception as e:
                return f"Error loading file: {str(e)}", ""
        
        def update_category_display(category):
            try:
                config = prompts_manager.get_config()
                if category in config["categories"]:
                    prompts_list = config["categories"][category]
                    return "\n".join(prompts_list)
                return ""
            except Exception as e:
                return f"Error loading category: {str(e)}"
        
        def add_to_category(category, prompts_text):
            try:
                if not category or not prompts_text.strip():
                    return "Please select category and enter prompts", ""
                
                prompts_list = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                prompts_manager.add_to_category(category, prompts_list)
                
                # Update display
                updated_display = update_category_display(category)
                return f"Added {len(prompts_list)} prompts to {category}", updated_display
            except Exception as e:
                return f"Error adding to category: {str(e)}", ""
        
        def remove_from_category(category, prompts_text):
            try:
                if not category or not prompts_text.strip():
                    return "Please select category and enter prompts", ""
                
                prompts_list = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                prompts_manager.remove_from_category(category, prompts_list)
                
                # Update display
                updated_display = update_category_display(category)
                return f"Removed {len(prompts_list)} prompts from {category}", updated_display
            except Exception as e:
                return f"Error removing from category: {str(e)}", ""
        
        def save_template(name, description, content):
            try:
                if not name.strip() or not content.strip():
                    return "Please enter template name and content"
                
                template_file = prompts_manager.create_prompt_template(name, content, description)
                return f"Template saved as {template_file.name}"
            except Exception as e:
                return f"Error saving template: {str(e)}"
        
        def load_template(template_name):
            try:
                if not template_name:
                    return "Please select a template", "", "", ""
                
                template_data = prompts_manager.load_prompt_template(template_name.lower().replace(' ', '_'))
                
                if not template_data:
                    return "Template not found", "", "", ""
                
                return (
                    f"Loaded template: {template_data.get('name', 'Unnamed')}",
                    template_data.get('name', ''),
                    template_data.get('description', ''),
                    template_data.get('template', '')
                )
            except Exception as e:
                return f"Error loading template: {str(e)}", "", "", ""
        
        # Wire up events
        for component in [enabled_checkbox, random_checkbox, max_variations]:
            component.change(
                update_config,
                inputs=[enabled_checkbox, random_checkbox, max_variations],
                outputs=[status_message]
            )
        
        test_btn.click(
            test_dynamic_prompt,
            inputs=[test_prompt],
            outputs=[test_result]
        )
        
        generate_variations_btn.click(
            show_all_variations,
            inputs=[test_prompt],
            outputs=[status_message, variations_display]
        )
        
        save_file_btn.click(
            save_prompts_to_file,
            inputs=[file_format, file_name, prompts_text],
            outputs=[status_message]
        )
        
        load_selected_btn.click(
            load_prompts_from_file,
            inputs=[available_files],
            outputs=[status_message, prompts_text]
        )
        
        category_select.change(
            update_category_display,
            inputs=[category_select],
            outputs=[category_preview]
        )
        
        add_to_category_btn.click(
            add_to_category,
            inputs=[category_select, category_prompts],
            outputs=[status_message, category_preview]
        )
        
        remove_from_category_btn.click(
            remove_from_category,
            inputs=[category_select, category_prompts],
            outputs=[status_message, category_preview]
        )
        
        save_template_btn.click(
            save_template,
            inputs=[template_name, template_description, template_content],
            outputs=[status_message]
        )
        
        load_template_btn.click(
            load_template,
            inputs=[available_templates],
            outputs=[status_message, template_name, template_description, template_content]
        )
