import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi, HfFolder
import gradio as gr

class CustomModelConfig:
    def __init__(self, model_name, alias, num_train_steps, max_sequence_length, base_arch="schnell"):
        self.model_name = model_name
        self.alias = alias
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.base_arch = base_arch
        self.supports_guidance = (base_arch == "dev")
        self.custom_transformer_model = model_name  # Added for compatibility with mflux library
        self.requires_sigma_shift = False  # Added for compatibility with RuntimeConfig

    def is_dev(self):
        """Check if this is a dev model configuration."""
        return self.base_arch == "dev"
        
    def x_embedder_input_dim(self):
        """Return the input dimension for the x_embedder.
        This method is required by the mflux library's Transformer implementation."""
        return 3072

    @staticmethod
    def from_alias(alias):
        return get_custom_model_config(alias)

MODELS = {
    "dev": CustomModelConfig("AITRADER/MFLUXUI.1-dev", "dev", 1000, 512, "dev"),
    "schnell": CustomModelConfig("AITRADER/MFLUXUI.1-schnell", "schnell", 1000, 256, "schnell"),
    "dev-8-bit": CustomModelConfig("AITRADER/MFLUXUI.1-dev-8-bit", "dev-8-bit", 1000, 512, "dev"),
    "dev-4-bit": CustomModelConfig("AITRADER/MFLUXUI.1-dev-4-bit", "dev-4-bit", 1000, 512, "dev"),
    "dev-6-bit": CustomModelConfig("AITRADER/MFLUXUI.1-dev-6-bit", "dev-6-bit", 1000, 512, "dev"),
    "dev-3-bit": CustomModelConfig("AITRADER/MFLUXUI.1-dev-3-bit", "dev-3-bit", 1000, 512, "dev"),
    "schnell-8-bit": CustomModelConfig("AITRADER/MFLUXUI.1-schnell-8-bit", "schnell-8-bit", 1000, 256, "schnell"),
    "schnell-4-bit": CustomModelConfig("AITRADER/MFLUXUI.1-schnell-4-bit", "schnell-4-bit", 1000, 256, "schnell"),
    "schnell-6-bit": CustomModelConfig("AITRADER/MFLUXUI.1-schnell-6-bit", "schnell-6-bit", 1000, 256, "schnell"),
    "schnell-3-bit": CustomModelConfig("AITRADER/MFLUXUI.1-schnell-3-bit", "schnell-3-bit", 1000, 256, "schnell"),
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
    predefined_models = [
        "schnell-4-bit", "dev-4-bit", 
        "schnell-8-bit", "dev-8-bit", 
        "schnell-6-bit", "dev-6-bit", 
        "schnell-3-bit", "dev-3-bit",
        "schnell", "dev"
    ]
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
        if not model_name or any(model_name.endswith(f"-{bits}-bit") for bits in ["3", "4", "6", "8"]):
            return gr.update(), gr.update(), gr.update(), "Error: Invalid model name"

        new_alias = f"{model_name}-{quantize_bits}-bit"
        if new_alias in MODELS:
            return gr.update(), gr.update(), gr.update(), f"Error: Model {new_alias} already exists"

        source_config = get_custom_model_config(model_name)
        new_config = CustomModelConfig(
            source_config.model_name,
            new_alias,
            source_config.num_train_steps,
            source_config.max_sequence_length,
            source_config.base_arch
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

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length, api_key, base_arch="schnell"):
    """
    Download a model from Hugging Face and save it locally.
    """
    try:
        login_result = login_huggingface(api_key)
        if "Error" in login_result:
            # Return None for all model dropdowns and the error message
            return None, None, None, None, None, f"HF Login failed: {login_result}"

        model_dir = os.path.join("models", alias)
        os.makedirs(model_dir, exist_ok=True)

        downloaded_files = snapshot_download(
            repo_id=hf_model_name, 
            local_dir=model_dir, 
            use_auth_token=api_key
        )

        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length, base_arch)
        MODELS[alias] = new_config

        # Get updated model choices for all dropdowns
        model_choices = get_model_choices()
        
        print(f"Model {hf_model_name} successfully downloaded and saved as {alias}")
        # Return updated model choices for all 5 dropdowns and success message
        return model_choices, model_choices, model_choices, model_choices, model_choices, "Success"

    except Exception as e:
        error_message = f"Error downloading model: {str(e)}"
        print(f"Error: {error_message}")
        # Return None for all model dropdowns and the error message
        return None, None, None, None, None, error_message

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

def update_guidance_visibility(model):
    """
    Update de zichtbaarheid van de guidance slider op basis van het model.
    Voor dev modellen is guidance altijd zichtbaar, voor schnell modellen 
    is het wel zichtbaar maar optioneel.
    """
    is_dev = "dev" in model
    return gr.update(visible=True, label="Guidance Scale (required for dev models)" if is_dev else "Guidance Scale (optional)")

def get_model_choices():
    """
    Get model choices for Gradio dropdowns.
    This function wraps get_updated_models() to provide model choices for UI components.
    """
    models = get_updated_models()
    return gr.update(choices=models) if models else gr.update()
