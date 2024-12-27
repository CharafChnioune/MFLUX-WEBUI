import os
import gradio as gr
from huggingface_hub import HfApi, HfFolder, snapshot_download, hf_hub_download
from pathlib import Path
import json
from backend.model_manager import CustomModelConfig, get_custom_model_config

MODELS_DIR = "models"

class CustomModelConfig:
    def __init__(self, hf_model_name, alias, num_train_steps, max_sequence_length):
        self.hf_model_name = hf_model_name
        self.alias = alias
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length

def login_huggingface(api_key):
    """
    Login to Hugging Face with the given API key.
    """
    try:
        if not api_key:
            return "Error: API key is missing"
        
        HfFolder.save_token(api_key)
        api = HfApi()
        
        try:
            api.whoami()
            return "Successfully logged in to Hugging Face"
        except Exception as e:
            return f"Error validating credentials: {str(e)}"
        
    except Exception as e:
        return f"Error logging in to Hugging Face: {str(e)}"

def download_lora_model_huggingface(model_name, hf_api_key):
    """
    Download .safetensors files from Hugging Face repo.
    """
    try:
        if not hf_api_key:
            return "Error: API key is missing"
        
        HfFolder.save_token(hf_api_key)
        api = HfApi()
        
        try:
            api.whoami()
        except Exception as e:
            return f"Error validating credentials: {str(e)}"

        try:
            repo_files = api.list_repo_files(model_name)
            safetensors_files = [f for f in repo_files if f.endswith('.safetensors')]
            
            if not safetensors_files:
                return "No .safetensors files found in repository"
            
            for file in safetensors_files:
                output_path = os.path.join("lora", os.path.basename(file))
                print(f"Downloading {file} to {output_path}")
                
                hf_hub_download(
                    repo_id=model_name,
                    filename=file,
                    local_dir="lora",
                    local_dir_use_symlinks=False,
                    token=hf_api_key
                )
            
            return "Successfully downloaded all .safetensors files"
            
        except Exception as e:
            return f"Error downloading files: {str(e)}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def get_available_models():
    """
    Get a list of all available models, including predefined and custom models.
    """
    predefined_models = ["schnell-4-bit", "dev-4-bit", "schnell-8-bit", "dev-8-bit", "schnell", "dev"]
    custom_models = [f.name for f in Path(MODELS_DIR).iterdir() if f.is_dir()]
    custom_models = [m for m in custom_models if m not in predefined_models]
    custom_models.sort(key=str.lower)
    all_models = predefined_models + custom_models
    return all_models

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length, api_key):
    """
    Download and save a model from Hugging Face.
    """
    try:
        login_result = login_huggingface(api_key)
        if "Error" in login_result:
            return gr.update(), gr.update(), gr.update(), gr.update(), login_result

        model_dir = os.path.join(MODELS_DIR, alias)
        os.makedirs(model_dir, exist_ok=True)

        downloaded_files = snapshot_download(
            repo_id=hf_model_name, 
            local_dir=model_dir, 
            local_dir_use_symlinks=False
        )

        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length)
        get_custom_model_config.__globals__['models'][alias] = new_config

        print(f"Model {hf_model_name} successfully downloaded and saved as {alias}")
        return (
            gr.update(choices=get_available_models()),
            gr.update(choices=get_available_models()),
            gr.update(choices=get_available_models()),
            gr.update(choices=[m for m in get_available_models() if not m.endswith("-4-bit") and not m.endswith("-8-bit")]),
            f"Model {hf_model_name} successfully downloaded and saved as {alias}"
        )

    except Exception as e:
        error_message = f"Error downloading model: {str(e)}"
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), gr.update(), error_message

def load_api_key(key_type="civitai"):
    """Load API key from config file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get(f"{key_type}_api_key", "")
    return ""

def save_api_key(api_key, key_type="civitai"):
    """Save API key to config file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    config[f"{key_type}_api_key"] = api_key
    with open(config_path, "w") as f:
        json.dump(config, f)
    return "API key saved successfully"

def load_hf_api_key():
    """Load HuggingFace API key."""
    return load_api_key("huggingface")

def save_hf_api_key(api_key):
    """Save HuggingFace API key."""
    return save_api_key(api_key, "huggingface")
