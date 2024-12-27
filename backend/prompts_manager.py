import os
import json
import gc
import gradio as gr
from typing import Optional, Any, List
from pathlib import Path
from .ollama_manager import enhance_prompt as enhance_prompt_ollama
from .mlx_vlm_manager import load_mlx_model, VisionModelWrapper, generate_with_model, generate_lm

def get_prompt_files(tab_name):
    """Get prompt files for specific tab"""
    script_dir = os.path.dirname(os.path.dirname(__file__))
    
    folder_map = {
        "easy": "easy-advanced-prompt",
        "advanced": "easy-advanced-prompt",
        "controlnet": "controlnet",
        "image-to-image": "image-to-image"
    }
    
    folder = folder_map.get(tab_name, "easy-advanced-prompt")
    prompt_dir = os.path.join(script_dir, 'prompts', folder)
    
    if not os.path.exists(prompt_dir):
        return []
        
    md_files = [f.replace(".md", "") for f in os.listdir(prompt_dir) if f.endswith('.md')]
    return md_files

def load_prompt_file(prompt_type="system"):
    """
    Load a prompt file based on the type.
    """
    prompt_files = {
        "system": "system_prompt.md",
        "controlnet": "controlnet_prompt.md",
        "image_to_image": "image_to_image_prompt.md"
    }
    
    if prompt_type not in prompt_files:
        return ""
        
    try:
        script_dir = os.path.dirname(os.path.dirname(__file__))
        prompt_path = os.path.join(script_dir, 'prompts', prompt_files[prompt_type])
        
        if not os.path.exists(prompt_path):
            return ""
            
        with open(prompt_path, 'r') as file:
            return file.read()
            
    except Exception as e:
        print(f"Error loading prompt file: {str(e)}")
        return ""

def save_prompt_file(prompt_type, content):
    """
    Save content to a prompt file based on the type.
    """
    prompt_files = {
        "system": "system_prompt.md",
        "controlnet": "controlnet_prompt.md",
        "image_to_image": "image_to_image_prompt.md"
    }
    
    if prompt_type not in prompt_files:
        print(f"Unknown prompt type: {prompt_type}")
        return False
        
    try:
        script_dir = os.path.dirname(os.path.dirname(__file__))
        prompt_path = os.path.join(script_dir, 'prompts', prompt_files[prompt_type])
            
        with open(prompt_path, 'w') as file:
            file.write(content)
        return True
            
    except Exception as e:
        print(f"Error saving prompt file: {str(e)}")
        return False

def read_system_prompt(tab_name="easy"):
    """Read system prompt from file based on tab"""
    try:
        script_dir = os.path.dirname(os.path.dirname(__file__))
        
        folder_map = {
            "easy": "easy-advanced-prompt",
            "advanced": "easy-advanced-prompt",
            "controlnet": "controlnet",
            "image-to-image": "image-to-image"
        }
        
        folder = folder_map.get(tab_name, "easy-advanced-prompt")
        prompt_dir = os.path.join(script_dir, 'prompts', folder)
        
        if os.path.exists(prompt_dir):
            md_files = [f for f in os.listdir(prompt_dir) if f.endswith('.md')]
            if md_files:
                prompt_path = os.path.join(prompt_dir, md_files[0])
                with open(prompt_path, 'r') as file:
                    return file.read()
        return ""
    except Exception as e:
        print(f"Error reading system prompt: {str(e)}")
        return ""

def save_llm_settings(llm_type: str, model: str, prompt_content: str, tab_name: str = "easy") -> bool:
    """
    Save LLM settings to a central JSON file.
    
    Args:
        llm_type: Type of LLM (Ollama or MLX)
        model: Name of the model
        prompt_content: Content of the prompt template
        tab_name: Name of the tab (easy, advanced, controlnet, image-to-image)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'llm_settings.json')
        
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        
        if tab_name not in settings:
            settings[tab_name] = {}
            
        settings[tab_name].update({
            'llm_type': llm_type,
            'model': model,
            'prompt': prompt_content
        })
        
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error saving LLM settings: {str(e)}")
        return False

def load_llm_settings(tab_name: str = "easy") -> dict:
    """
    Load LLM settings from the central JSON file.
    
    Args:
        tab_name: Name of the tab to load settings for
        
    Returns:
        dict: Settings for the tab with defaults if not found
    """
    try:
        settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'llm_settings.json')
        
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                
            tab_settings = settings.get(tab_name, {})
            
            return {
                'llm_type': tab_settings.get('llm_type', 'Ollama'),
                'model': tab_settings.get('model', ''),
                'prompt': tab_settings.get('prompt', '')
            }
            
        return {
            'llm_type': 'Ollama',
            'model': '',
            'prompt': ''
        }
        
    except Exception as e:
        print(f"Error loading LLM settings: {str(e)}")
        return {
            'llm_type': 'Ollama',
            'model': '',
            'prompt': ''
        }

def enhance_prompt(prompt: str, llm_type: str, ollama_model: str, mlx_model: str, system_prompt: Optional[str] = None, image: Optional[Any] = None, tab_name: str = "controlnet") -> str:
    """
    Enhance prompt using either Ollama or MLX.
    For MLX-VLM models in controlnet tab, use the control image for better prompts.
    For Ollama, silently try to use the image with vision models.
    """
    if llm_type == "Ollama":
        return enhance_prompt_ollama(prompt, ollama_model, system_prompt, image)
    else:  
        model, processor, config = load_mlx_model(mlx_model)
        if model is None:
            return prompt
            
        if isinstance(model, VisionModelWrapper):
            if image is not None:
                enhanced = enhance_prompt_with_mlx(prompt, mlx_model, images=[image], tab_name=tab_name)
                return enhanced
            else:
                import gradio as gr
                gr.Warning("Vision models require an image for prompt enhancement. Please upload a control image first.")
                return prompt
        return enhance_prompt_with_mlx(prompt, mlx_model)

def enhance_prompt_with_mlx(prompt: str, model_name: str, images: Optional[List[str]] = None, tab_name: str = "controlnet") -> str:
    """
    Use MLX model to enhance a prompt.
    
    Args:
        prompt: The prompt to enhance
        model_name: Name of the model to use
        images: Optional list of base64 encoded images
        tab_name: The name of the tab (controlnet or image-to-image)
        
    Returns:
        The enhanced prompt text
    """
    try:
        gc.collect()
        
        model, processor, config = load_mlx_model(model_name)
        if model is None:
            return prompt
            
        if isinstance(model, VisionModelWrapper):
            script_dir = os.path.dirname(os.path.dirname(__file__))
            
            folder_map = {
                "easy": "easy-advanced-prompt",
                "advanced": "easy-advanced-prompt",
                "controlnet": "controlnet",
                "image-to-image": "image-to-image"
            }
            
            folder = folder_map.get(tab_name, "controlnet")
            prompt_dir = os.path.join(script_dir, 'prompts', folder)
            
            if not os.path.exists(prompt_dir):
                return prompt
                
            md_files = [f for f in os.listdir(prompt_dir) if f.endswith('.md')]
            if not md_files:
                return prompt
                
            prompt_path = os.path.join(prompt_dir, md_files[0])
            
            with open(prompt_path, 'r') as f:
                prompt_template = f.read()
            
            formatted_prompt = prompt_template.replace("{user_prompt}", prompt)
            
            enhanced_text = generate_with_model(
                model=model,
                processor=processor,
                config=config,
                prompt=formatted_prompt,
                images=images,
                max_tokens=4000,
                temperature=0.0
            )
            
            if "Here is the enhanced prompt:" in enhanced_text:
                enhanced_text = enhanced_text.split("Here is the enhanced prompt:")[1].strip()
            if enhanced_text.startswith('"') and enhanced_text.endswith('"'):
                enhanced_text = enhanced_text[1:-1]
            
            return enhanced_text.strip()
            
        else:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enhanced_text = generate_lm(
                model=model,
                tokenizer=processor,
                prompt=formatted_prompt,
                verbose=True
            )
            
            return enhanced_text.strip()
        
    except Exception as e:
        print(f"Error enhancing prompt: {str(e)}")
        return prompt
    finally:
        gc.collect()

def save_ollama_settings(model, system_prompt):
    """Deprecated: Use save_llm_settings instead"""
    print("Warning: save_ollama_settings is deprecated, use save_llm_settings instead")
    return save_llm_settings("Ollama", model, system_prompt)

def save_settings(model, prompt_type, prompt_content):
    """Deprecated: Use save_llm_settings instead"""
    print("Warning: save_settings is deprecated, use save_llm_settings instead")
    return save_llm_settings("Ollama", model, prompt_content)
