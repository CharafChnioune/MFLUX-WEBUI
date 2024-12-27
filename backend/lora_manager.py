import os
import re
import json
import requests
from pathlib import Path
from tqdm import tqdm
import gradio as gr
from huggingface_hub import hf_hub_download, HfApi, HfFolder

LORA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora")
os.makedirs(LORA_DIR, exist_ok=True)

MAX_LORAS = 5

def get_available_lora_files():
    """
    Get a list of available LoRA files from the LoRA directory.
    
    Returns:
        list: A list of tuples containing (display_name, file_path) for each .safetensors file.
            - display_name: The filename without extension
            - file_path: The full path to the LoRA file
    """
    lora_files = []
    for root, dirs, files in os.walk(LORA_DIR):
        for file in files:
            if file.endswith(".safetensors"):
                display_name = os.path.splitext(file)[0]
                lora_files.append((display_name, os.path.join(root, file)))
    lora_files.sort(key=lambda x: x[0].lower())
    return lora_files

def get_lora_choices():
    """
    Get a list of available LoRA model names.
    """
    return [name for name, _ in get_available_lora_files()]

def process_lora_files(selected_loras):
    """
    Process selected LoRA files and return valid paths.
    """
    if not selected_loras:
        return None

    valid_loras = []
    for lora_name in selected_loras:
        if os.path.isfile(lora_name) and lora_name.endswith('.safetensors'):
            valid_loras.append(lora_name)
            continue

        lora_path = os.path.join(LORA_DIR, f"{lora_name}.safetensors")
        if os.path.exists(lora_path):
            valid_loras.append(lora_path)

    return valid_loras if valid_loras else None

def update_lora_scales(selected_loras):
    """
    Update LoRA scale sliders based on selected LoRAs.
    
    Args:
        selected_loras (list): A list of selected LoRA model names
        
    Returns:
        list: A list of Gradio update objects for scale sliders.
            Each update object contains:
            - visible: Boolean indicating if the slider should be shown
            - label: String label for the slider with LoRA name
            - value: Default scale value (1.0)
    """
    updates = []
    for i, lora_name in enumerate(selected_loras[:5]):
        updates.append(gr.update(visible=True, label=f"Scale: {lora_name}", value=1.0))
    for _ in range(5 - len(selected_loras)):
        updates.append(gr.update(visible=False, value=1.0, label="Scale:"))
    return updates

def download_lora_model_huggingface(model_name, hf_api_key):
    """
    Download LoRA model files (.safetensors) from a Hugging Face repository.
    
    Args:
        model_name (str): The name of the Hugging Face repository containing the LoRA model
        hf_api_key (str): Hugging Face API key for authentication
        
    Returns:
        str: A success or error message indicating the result of the download operation
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

def download_lora_model(hf_model_name, api_key=None):
    """
    Download a LoRA model from Hugging Face.
    """
    try:
        if api_key:
            HfFolder.save_token(api_key)

        model_name = hf_model_name.split("/")[-1]
        local_path = os.path.join(LORA_DIR, f"{model_name}.safetensors")

        downloaded_file = hf_hub_download(
            repo_id=hf_model_name,
            filename="pytorch_lora_weights.safetensors",
            local_dir=LORA_DIR,
            local_dir_use_symlinks=False
        )

        if downloaded_file != local_path:
            os.rename(downloaded_file, local_path)

        return f"Successfully downloaded LoRA model: {model_name}"
    except Exception as e:
        return f"Error downloading LoRA model: {str(e)}"

def refresh_lora_choices():
    """Refresh the list of LoRA choices."""
    choices = get_lora_choices()
    return gr.update(choices=choices)

def get_updated_lora_files():
    """
    Get a list of all available LoRA files and update Gradio UI components.
    
    Returns:
        tuple: A tuple containing three Gradio update objects for UI components and a status message.
            - First update: Choices for the first LoRA selection component
            - Second update: Choices for the second LoRA selection component
            - Third update: Choices for the third LoRA selection component
            - str: Status message indicating success or failure
    """
    try:
        choices = get_lora_choices()
        return gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices), "LoRA files updated successfully"
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"Error updating LoRA files: {str(e)}"
