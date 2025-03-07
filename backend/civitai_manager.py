import os
import re
import json
import requests
from bs4 import BeautifulSoup
import gradio as gr

def slugify(value):
    """
    Slugify a string by removing special characters and replacing whitespace with hyphens.
    """
    # Remove special characters
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    # Replace whitespace with hyphens
    value = re.sub(r'[-\s]+', '-', value)
    return value

def get_updated_lora_files():
    lora_files = []
    for root, dirs, files in os.walk("lora"):
        for file in files:
            if file.endswith(".safetensors") or file.endswith(".ckpt"):
                lora_files.append(file)
    return lora_files

def download_lora_model(page_url, api_key):
    """
    Download a LoRA model from CivitAI.
    """
    if not api_key:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "Error: API key is missing"

    try:
        print(f"Starting download process for URL: {page_url}")
        model_id_match = re.search(r'/models/(\d+)', page_url)
        if not model_id_match:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Error: Could not extract model ID from the URL: {page_url}"

        model_id = model_id_match.group(1)
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(api_url, headers=headers)
        
        if response.status_code != 200:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Error: API request failed with status {response.status_code}"
        
        model_data = response.json()
        model_name = model_data.get("name", "unknown_model")
        model_version_id = model_data.get("modelVersions", [{}])[0].get("id")
        
        if not model_version_id:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "Error: No model version found"
        
        # Download model files
        download_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}/files"
        response = requests.get(download_url, headers=headers)
        
        if response.status_code != 200:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Error: Failed to get download links, status: {response.status_code}"
        
        files_data = response.json()
        
        # Ensure lora directory exists
        lora_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora")
        os.makedirs(lora_dir, exist_ok=True)
        
        for file_info in files_data:
            file_name = file_info.get("name")
            if not file_name:
                continue
                
            download_url = file_info.get("downloadUrl")
            if not download_url:
                print(f"Warning: No download URL for file {file_name}")
                continue
                
            file_path = os.path.join(lora_dir, file_name)
            response = requests.get(download_url, headers=headers, stream=True)
            
            if response.status_code != 200:
                print(f"Warning: Failed to download file {file_name}, status: {response.status_code}")
                continue
                
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Downloaded {file_name} to {file_path}")
        
        # Update LoRA choices
        lora_choices = get_lora_choices()
        return lora_choices, lora_choices, lora_choices, lora_choices, lora_choices, f"Successfully downloaded LoRA model: {model_name}"
            
    except Exception as e:
        error_message = f"Error downloading LoRA model: {str(e)}"
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), error_message

def save_api_key(api_key, key_type="civitai"):
    """
    Save API key to config file.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    config[f"{key_type}_api_key"] = api_key
    with open(config_path, "w") as f:
        json.dump(config, f)
    return "API key saved successfully"

def load_api_key(key_type="civitai"):
    """
    Load API key from config file.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get(f"{key_type}_api_key", "")
    return ""
