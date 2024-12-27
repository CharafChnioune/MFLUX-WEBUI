import gradio as gr
import os
from backend.ollama_manager import get_available_ollama_models
from backend.mlx_vlm_manager import get_available_models
from backend.prompts_manager import (
    read_system_prompt,
    get_prompt_files
)
from backend.settings_manager import save_llm_settings, load_llm_settings

def create_llm_settings(tab_name="easy", parent_accordion=None):
    """Create LLM Settings UI components"""
    settings = load_llm_settings(tab_name)
    llm_type_val = settings.get('llm_type', '')
    
    with gr.Column(scale=1):
        # Model Selection Section
        with gr.Group(visible=True):
            with gr.Row():
                llm_type = gr.Radio(
                    choices=["Ollama", "MLX"],
                    label="LLM Type",
                    value=llm_type_val,
                    scale=1,
                    container=True
                )
            
            with gr.Row():
                # Ollama model dropdown
                ollama_models, _ = get_available_ollama_models()
                ollama_model = gr.Dropdown(
                    choices=ollama_models,
                    label="Ollama Model",
                    value=settings.get('model') if llm_type_val == "Ollama" else "",
                    visible=llm_type_val == "Ollama",
                    allow_custom_value=True,
                    container=True,
                    scale=1
                )
                
                # MLX model dropdown
                mlx_models = get_available_models("llm")
                mlx_model = gr.Dropdown(
                    choices=mlx_models if mlx_models else [],
                    label="MLX Model",
                    value=settings.get('model') if llm_type_val == "MLX" else "",
                    visible=llm_type_val == "MLX",
                    allow_custom_value=True,
                    container=True,
                    scale=1
                )

        # Prompt Section
        with gr.Group(visible=True):
            with gr.Row():
                prompt_files = get_prompt_files(tab_name)
                prompt_type = gr.Dropdown(
                    choices=prompt_files,
                    label="Prompt Template",
                    value=prompt_files[0] if prompt_files else None,
                    allow_custom_value=True,
                    container=True,
                    scale=3
                )
                
                with gr.Column(scale=2, min_width=250):
                    new_prompt_name = gr.Textbox(
                        label="New Template Name",
                        placeholder="Enter name for new template",
                        container=True
                    )
                    with gr.Row(elem_classes="white-bg"):
                        create_prompt_btn = gr.Button("💾 Save", variant="primary", size="sm", min_width=100)
                        delete_prompt_btn = gr.Button("🗑️ Delete", variant="stop", size="sm", min_width=100)
            
            system_prompt = gr.Textbox(
                label="Prompt Content",
                lines=8,
                value=settings.get('prompt', '') or read_system_prompt(tab_name),
                container=True,
                show_label=True
            )
        
        with gr.Row():
            save_btn = gr.Button("Save Settings", variant="primary", size="lg")

    def delete_current_prompt(current_prompt):
        """Delete the current prompt file"""
        if not current_prompt:
            gr.Error("No prompt selected")
            return {
                new_prompt_name: gr.update(value=""),
                prompt_type: gr.update(choices=prompt_type.choices)
            }
            
        try:
            script_dir = os.path.dirname(os.path.dirname(__file__))
            prompt_folder = "easy-advanced-prompt" if tab_name in ["easy", "advanced"] else tab_name
            prompt_path = os.path.join(script_dir, '..', 'prompts', prompt_folder, f"{current_prompt}.md")
            
            if os.path.exists(prompt_path):
                os.remove(prompt_path)
                new_choices = get_prompt_files(tab_name)
                gr.Info(f"Deleted prompt: {current_prompt}")
                return {
                    new_prompt_name: gr.update(value=""),
                    prompt_type: gr.update(choices=new_choices, value=new_choices[0] if new_choices else None)
                }
            else:
                gr.Error(f"Prompt {current_prompt} not found")
                return {
                    new_prompt_name: gr.update(value=""),
                    prompt_type: gr.update(choices=prompt_type.choices)
                }
                
        except Exception as e:
            gr.Error(f"Error deleting prompt: {str(e)}")
            return {
                new_prompt_name: gr.update(value=""),
                prompt_type: gr.update(choices=prompt_type.choices)
            }
    
    def create_new_prompt(name, content, current_type):
        if not name and not current_type:
            gr.Error("Please enter a name for the new prompt or select an existing one")
            return {
                new_prompt_name: gr.update(value=""),
                prompt_type: gr.update(choices=prompt_type.choices)
            }
        
        try:
            script_dir = os.path.dirname(os.path.dirname(__file__))
            prompt_folder = "easy-advanced-prompt" if tab_name in ["easy", "advanced"] else tab_name
            
            # If no new name is provided, update the current prompt
            if not name and current_type:
                prompt_path = os.path.join(script_dir, '..', 'prompts', prompt_folder, f"{current_type}.md")
                with open(prompt_path, 'w') as f:
                    f.write(content or "")
                gr.Info(f"Updated prompt: {current_type}")
                return {
                    new_prompt_name: gr.update(value=""),
                    prompt_type: gr.update(choices=prompt_type.choices, value=current_type)
                }
            
            # Handle new prompt creation
            if not name.endswith('.md'):
                name = f"{name}.md"
                
            prompt_path = os.path.join(script_dir, '..', 'prompts', prompt_folder, name)
            
            # Check if file already exists
            if os.path.exists(prompt_path):
                # Update existing prompt
                with open(prompt_path, 'w') as f:
                    f.write(content or "")
                gr.Info(f"Updated prompt: {name}")
            else:
                # Create new prompt
                with open(prompt_path, 'w') as f:
                    f.write(content or "")
                gr.Info(f"Created new prompt: {name}")
            
            # Update dropdown choices
            new_choices = get_prompt_files(tab_name)
            
            return {
                new_prompt_name: gr.update(value=""),
                prompt_type: gr.update(choices=new_choices, value=name.replace(".md", ""))
            }
            
        except Exception as e:
            gr.Error(f"Error saving prompt: {str(e)}")
            return {
                new_prompt_name: gr.update(value=""),
                prompt_type: gr.update(choices=prompt_type.choices)
            }
    
    def update_prompt_content(prompt_type):
        try:
            script_dir = os.path.dirname(os.path.dirname(__file__))
            prompt_folder = "easy-advanced-prompt" if tab_name in ["easy", "advanced"] else tab_name
            prompt_path = os.path.join(script_dir, '..', 'prompts', prompt_folder, f'{prompt_type}.md')
            
            with open(prompt_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading prompt: {str(e)}")
            return ""
    
    def update_model_visibility(llm_type, current_ollama_model, current_mlx_model):
        if llm_type == "Ollama":
            models, _ = get_available_ollama_models()
            return {
                ollama_model: gr.update(visible=True, choices=models, value=current_ollama_model or ""),
                system_prompt: gr.update(visible=True),
                save_btn: gr.update(visible=True),
                mlx_model: gr.update(visible=False),
                prompt_type: gr.update(visible=True)
            }
        else:  # MLX
            models = get_available_models("llm")
            return {
                ollama_model: gr.update(visible=False),
                system_prompt: gr.update(visible=True),
                save_btn: gr.update(visible=True),
                mlx_model: gr.update(visible=True, choices=models, value=current_mlx_model or ""),
                prompt_type: gr.update(visible=True)
            }
    
    llm_type.change(
        fn=update_model_visibility,
        inputs=[llm_type, ollama_model, mlx_model],
        outputs=[ollama_model, system_prompt, save_btn, mlx_model, prompt_type]
    )
    
    def save_handler(llm_type_val, ollama_val, mlx_val, prompt_content):
        try:
            model = ollama_val if llm_type_val == "Ollama" else mlx_val
            if not model:
                return [gr.Error("Please select a model"), None]
            
            success = save_llm_settings(llm_type_val, model, prompt_content, tab_name)
            if success:
                return [
                    gr.Info("Settings saved successfully"), 
                    gr.update(open=False)
                ]
            return [gr.Error("Failed to save settings"), None]
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
            return [gr.Error(f"Error saving settings: {str(e)}"), None]
    
    save_btn.click(
        fn=save_handler,
        inputs=[
            llm_type,
            ollama_model,
            mlx_model,
            system_prompt
        ],
        outputs=[
            gr.Text(visible=False),
            parent_accordion if parent_accordion else gr.Accordion(visible=False)
        ]
    )
    
    # Connect event handlers
    prompt_type.change(
        fn=update_prompt_content,
        inputs=[prompt_type],
        outputs=[system_prompt]
    )
    
    create_prompt_btn.click(
        fn=create_new_prompt,
        inputs=[new_prompt_name, system_prompt, prompt_type],
        outputs=[new_prompt_name, prompt_type]
    )
    
    delete_prompt_btn.click(
        fn=delete_current_prompt,
        inputs=[prompt_type],
        outputs=[new_prompt_name, prompt_type]
    )
    
    return [llm_type, ollama_model, system_prompt, save_btn, mlx_model] 