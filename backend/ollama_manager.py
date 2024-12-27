import os
import json
import ollama
import gradio as gr
from pathlib import Path
import traceback

def get_available_ollama_models():
    """
    Get locally available Ollama models via the official API.
    """
    try:
        response = ollama.list()
        model_names = []
        
        if hasattr(response, 'models'):
            for model in response.models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
        
        if not model_names:
            return ["No Ollama models found - use 'ollama pull <model>' first"], None

        return model_names, model_names[0]
        
    except Exception as e:
        return ["Error fetching Ollama models - is Ollama running?"], None

def ensure_llama_model(model_name):
    """
    Pull model via 'ollama pull <model>' if needed.
    """
    try:
        ollama.pull(model_name)
        return True
    except Exception:
        return False

def enhance_prompt(prompt, ollama_model, system_prompt, image=None):
    """
    Use Ollama to improve a prompt, optionally with an image.
    The image will be silently ignored if the model doesn't support vision.
    
    Args:
        prompt: The prompt to enhance
        ollama_model: Name of the Ollama model
        system_prompt: System prompt to use
        image: Optional PIL Image or base64 string
    """
    try:
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        if image is not None:
            try:
                import io
                import base64
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode()
                
                messages.append({
                    'role': 'user',
                    'content': f"Here is an image in base64: {base64_image}\n\nPlease use these instructions: {system_prompt} and help me enhance this prompt: {prompt}"
                })
            except Exception as e:
                messages.append({
                    'role': 'user',
                    'content': f"Please use these instructions: {system_prompt} and help me enhance this prompt: {prompt}"
                })
        else:
            messages.append({
                'role': 'user',
                'content': f"Please use these instructions: {system_prompt} and help me enhance this prompt: {prompt}"
            })

        try:
            response = ollama.chat(
                model=ollama_model,
                messages=messages
            )
            return response['message']['content'].strip()
        except Exception as chat_error:
            response = ollama.generate(
                model=ollama_model,
                prompt=f"Please use these instructions: {system_prompt} and help me enhance this prompt: {prompt}",
                system=system_prompt
            )
            return response['response'].strip()
            
    except Exception as e:
        print(f"Error while improving prompt: {str(e)}")
        return prompt

def load_settings():
    """
    Load Ollama settings from config.json.
    """
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
                return config.get("model", ""), config.get("prompt", "")
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
    return "", ""

def create_ollama_settings(prompt_type="system"):
    """Create Ollama settings UI components"""
    models, default_model = get_available_ollama_models()
    
    ollama_model = gr.Dropdown(
        choices=models,
        label="Model",
        value=default_model,
        visible=True
    )
    
    script_dir = os.path.dirname(os.path.dirname(__file__))
    prompt_file = os.path.join(script_dir, 'prompts', f'{prompt_type}_prompt.md')
    
    try:
        with open(prompt_file, "r") as f:
            default_prompt = f.read()
    except FileNotFoundError:
        print(f"Warning: {prompt_type}_prompt.md not found, using default prompt")
        default_prompt = "You are an expert at writing detailed, descriptive prompts for image generation AI models."
    
    system_prompt = gr.Textbox(
        label="System Prompt",
        lines=3,
        value=default_prompt
    )
    
    save_btn = gr.Button("Save Settings")
    
    return ollama_model, system_prompt, save_btn
