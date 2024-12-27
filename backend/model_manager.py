import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi, HfFolder
import gradio as gr

class CustomModelConfig:
    def __init__(self, model_name, alias, num_train_steps, max_sequence_length):
        self.model_name = model_name
        self.alias = alias
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length

    @staticmethod
    def from_alias(alias):
        return get_custom_model_config(alias)

MODELS = {
    "dev": CustomModelConfig("AITRADER/MFLUXUI.1-dev", "dev", 1000, 512),
    "schnell": CustomModelConfig("AITRADER/MFLUXUI.1-schnell", "schnell", 1000, 256),
    "dev-8-bit": CustomModelConfig("AITRADER/MFLUXUI.1-dev-8-bit", "dev-8-bit", 1000, 512),
    "dev-4-bit": CustomModelConfig("AITRADER/MFLUXUI.1-dev-4-bit", "dev-4-bit", 1000, 512),
    "schnell-8-bit": CustomModelConfig("AITRADER/MFLUXUI.1-schnell-8-bit", "schnell-8-bit", 1000, 256),
    "schnell-4-bit": CustomModelConfig("AITRADER/MFLUXUI.1-schnell-4-bit", "schnell-4-bit", 1000, 256),
}

def get_custom_model_config(model_alias):
    config = MODELS.get(model_alias)
    if config is None:
        raise ValueError(
            f"Invalid model alias: {model_alias}. Available aliases: {', '.join(MODELS.keys())}"
        )
    return config

def get_updated_models():
    """
    Get a list of all available models, including predefined and custom models.
    """
    predefined_models = ["schnell-4-bit", "dev-4-bit", "schnell-8-bit", "dev-8-bit", "schnell", "dev"]
    custom_models = [f.name for f in Path("models").iterdir() if f.is_dir()]
    custom_models = [m for m in custom_models if m not in predefined_models]
    custom_models.sort(key=str.lower)
    all_models = predefined_models + custom_models
    return all_models

def save_quantized_model_gradio(model_name, quantize_bits):
    """
    Save a quantized version of a model.
    """
    try:
        if not model_name or model_name.endswith("-4-bit") or model_name.endswith("-8-bit"):
            return gr.update(), gr.update(), gr.update(), "Error: Invalid model name"

        new_alias = f"{model_name}-{quantize_bits}-bit"
        if new_alias in MODELS:
            return gr.update(), gr.update(), gr.update(), f"Error: Model {new_alias} already exists"

        source_config = get_custom_model_config(model_name)
        new_config = CustomModelConfig(
            source_config.model_name,
            new_alias,
            source_config.num_train_steps,
            source_config.max_sequence_length
        )
        MODELS[new_alias] = new_config

        updated_models = get_updated_models()
        return (
            gr.update(choices=updated_models),
            gr.update(choices=updated_models),
            gr.update(choices=updated_models),
            f"Successfully created quantized model: {new_alias}"
        )

    except Exception as e:
        error_message = f"Error creating quantized model: {str(e)}"
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), error_message

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length, api_key):
    """
    Download a model from Hugging Face and save it locally.
    """
    try:
        login_result = login_huggingface(api_key)
        if "Error" in login_result:
            return None, f"HF Login failed: {login_result}"

        model_dir = os.path.join("models", alias)
        os.makedirs(model_dir, exist_ok=True)

        downloaded_files = snapshot_download(
            repo_id=hf_model_name, 
            local_dir=model_dir, 
            use_auth_token=api_key
        )

        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length)
        MODELS[alias] = new_config

        print(f"Model {hf_model_name} successfully downloaded and saved as {alias}")
        return new_config, "Success"

    except Exception as e:
        error_message = f"Error downloading model: {str(e)}"
        print(f"Error: {error_message}")
        return None, error_message

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
