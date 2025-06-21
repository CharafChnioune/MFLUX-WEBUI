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
    """
    lora_files = []
    for root, dirs, files in os.walk(LORA_DIR):
        for file in files:
            # Filter out .link files and only include .safetensors
            if file.endswith(".safetensors") and not file.endswith(".link"):
                # Remove any .link extension if it somehow got appended
                display_name = os.path.splitext(file)[0].replace(".link", "")
                file_path = os.path.join(root, file)
                
                # Skip if it's a symlink
                if not os.path.islink(file_path):
                    lora_files.append((display_name, file_path))
                    
    # Sort alphabetically, case-insensitive
    lora_files.sort(key=lambda x: x[0].lower())
    return lora_files

def get_lora_choices():
    """
    Get a list of available LoRA model names.
    """
    return [name for name, _ in get_available_lora_files()]

def process_lora_files(selected_loras, lora_scales=None):
    """
    Process selected LoRA files and return valid paths.
    If a HuggingFace repository ID is provided (e.g. "username/repo_name|lora_name"), 
    it will automatically download the LoRA.
    
    Args:
        selected_loras: List of LoRA file names or HuggingFace repository IDs
        lora_scales: Optional list of scaling factors for each LoRA
        
    Returns:
        List of LoRA file paths if only selected_loras is provided
        or
        List of LoRA scaling factors if both selected_loras and lora_scales are provided
    """
    if selected_loras is None:
        return None
        
    valid_loras = []
    
    # Get all available LoRA files
    available_loras = get_available_lora_files()
    lora_dict = {name: path for name, path in available_loras}
    
    for lora in selected_loras:
        if lora:
            if "|" in lora:
                # It's a Hugging Face repository ID
                try:
                    lora_path = download_lora_model_huggingface(lora)
                    if "Error" not in lora_path:
                        valid_loras.append(lora_path)
                except Exception as e:
                    print(f"Error downloading LoRA from HuggingFace: {str(e)}")
            elif os.path.exists(lora) and lora.endswith(".safetensors"):
                # It's a direct file path
                valid_loras.append(lora)
            elif lora in lora_dict:
                # It's a name of a LoRA file in the LoRA directory
                valid_loras.append(lora_dict[lora])
            else:
                print(f"LoRA not found: {lora}")
    
    # Als lora_scales is meegegeven, verwerk deze
    if lora_scales is not None:
        # Convert string scale values to floats
        lora_scales_float = []
        
        # Als lora_scales een tuple is (wat vaak gebeurt bij variadic args van gradio), converteer naar list
        if isinstance(lora_scales, tuple):
            lora_scales = list(lora_scales)
            
        # Als lora_scales een enkele waarde is, maak er een list van
        if not isinstance(lora_scales, list):
            lora_scales = [lora_scales]
        
        # Process scale values
        for i, scale in enumerate(lora_scales):
            if i < len(valid_loras):  # Only process scales for valid loras
                try:
                    if scale is None or scale == "":
                        lora_scales_float.append(1.0)  # Default value
                    else:
                        lora_scales_float.append(float(scale))
                except (ValueError, TypeError):
                    print(f"Invalid LoRA scale value: {scale}, using default value 1.0")
                    lora_scales_float.append(1.0)
        
        # If there are more loras than scales, use default value 1.0 for the rest
        while len(lora_scales_float) < len(valid_loras):
            lora_scales_float.append(1.0)
            
        # Return as tuple since mflux library expects tuple for lora_scales
        return tuple(lora_scales_float)
    
    return valid_loras

def update_lora_scales(selected_loras):
    """
    Update the Gradio components for LoRA scales based on selected LoRAs.
    """
    if not selected_loras:
        return [gr.update(visible=False)] * MAX_LORAS
    
    valid_loras = process_lora_files(selected_loras)
    
    # Bereid updates voor
    updates = []
    for i in range(MAX_LORAS):
        if i < len(valid_loras):
            lora_name = os.path.basename(valid_loras[i])
            updates.append(gr.update(
                visible=True,
                label=f"LoRA Weight {i+1}: {lora_name}"
            ))
        else:
            updates.append(gr.update(visible=False))
    
    return updates

def download_lora_model_huggingface(model_name, hf_api_key=None):
    """
    Download a LoRA model from HuggingFace.
    
    Args:
        model_name (str): HuggingFace model name (username/model_name) 
                          or (username/model_name|lora_name)
        hf_api_key (str, optional): HuggingFace API key for private models
        
    Returns:
        str: Path to the downloaded LoRA model or error message
    """
    try:
        # Set up API key if provided
        if hf_api_key:
            HfFolder.save_token(hf_api_key)
            
        # Check if model_name contains a specific LoRA name
        repo_id = model_name
        lora_name = None
        if "|" in model_name:
            repo_id, lora_name = model_name.split("|", 1)
            
        # Get model name for file naming
        model_name_only = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        if lora_name:
            model_name_only = f"{model_name_only}_{lora_name}"
            
        # Ensure lora directory exists
        os.makedirs(LORA_DIR, exist_ok=True)
        
        # Try to locate or download the .safetensors file
        try:
            # Check if we need to find a specific LoRA file or just download the main one
            if lora_name:
                # List files in the repo to find the specific LoRA
                hf_api = HfApi()
                files = hf_api.list_repo_files(repo_id)
                
                # Find files that match the LoRA name
                matching_files = [f for f in files if f.endswith('.safetensors') and (lora_name in f)]
                
                if not matching_files:
                    # Try without filtering by lora_name (might be the repo name itself)
                    matching_files = [f for f in files if f.endswith('.safetensors')]
                    
                if not matching_files:
                    return f"Error: No .safetensors files found in {repo_id}"
                    
                # Use the first matching file
                file_to_download = matching_files[0]
            else:
                # Just find any .safetensors file in the repo
                hf_api = HfApi()
                files = hf_api.list_repo_files(repo_id)
                safetensors_files = [f for f in files if f.endswith('.safetensors')]
                
                if not safetensors_files:
                    return f"Error: No .safetensors files found in {repo_id}"
                    
                file_to_download = safetensors_files[0]
                
            # Download the selected file
            local_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_to_download,
                local_dir=LORA_DIR,
                token=hf_api_key
            )
            
            # Rename the file to include the model name if necessary
            base_name = os.path.basename(local_file_path)
            new_name = f"{model_name_only}.safetensors"
            new_path = os.path.join(LORA_DIR, new_name)
            
            # Only rename if the file names are different and target doesn't exist
            if base_name != new_name and not os.path.exists(new_path):
                os.rename(local_file_path, new_path)
                local_file_path = new_path
                
            print(f"Successfully downloaded LoRA model to {local_file_path}")
            return local_file_path
            
        except Exception as e:
            error_message = f"Error downloading LoRA file: {str(e)}"
            print(error_message)
            return f"Error: {error_message}"
            
    except Exception as e:
        error_message = f"Error in download_lora_model_huggingface: {str(e)}"
        print(error_message)
        return f"Error: {error_message}"

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

def download_lora(lora_filename, repo_id="ali-vilab/In-Context-LoRA"):
    """
    Download an in-context LoRA file from Hugging Face using the get_lora_filename function from mflux.
    
    Args:
        lora_filename: The filename of the LoRA to download
        repo_id: The Hugging Face repository ID (default is "ali-vilab/In-Context-LoRA")
        
    Returns:
        Path to the downloaded LoRA file, or None if download failed
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Create lora directory if it doesn't exist
        os.makedirs(LORA_DIR, exist_ok=True)
        
        # Download the file from Hugging Face
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=lora_filename,
            cache_dir=LORA_DIR,
            resume_download=True
        )
        
        print(f"Successfully downloaded LoRA file to {local_path}")
        return local_path
    
    except Exception as e:
        print(f"Error downloading LoRA: {str(e)}")
        return None
