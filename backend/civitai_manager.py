import os
import re
import json
import requests
from bs4 import BeautifulSoup
import gradio as gr

def slugify(value):
    """
    Convert a string to a URL-friendly slug.
    """
    value = str(value)
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
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
        return gr.update(), gr.update(), gr.update(), "Error: API key is missing"

    try:
        print(f"Starting download process for URL: {page_url}")
        model_id_match = re.search(r'/models/(\d+)', page_url)
        if not model_id_match:
            return gr.update(), gr.update(), gr.update(), f"Error: Could not extract model ID from the URL: {page_url}"

        model_id = model_id_match.group(1)
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(api_url, headers=headers)
        
        if response.status_code != 200:
            return gr.update(), gr.update(), gr.update(), f"Error: API request failed with status {response.status_code}"
        
        model_data = response.json()
        model_name = model_data.get('name', 'unknown_model')
        model_name = slugify(model_name)
        
        version = model_data.get('modelVersions', [{}])[0]
        files = version.get('files', [])
        
        if not files:
            return gr.update(), gr.update(), gr.update(), "Error: No files found for this model"
        
        safetensor_files = [f for f in files if f['name'].endswith('.safetensors')]
        if not safetensor_files:
            return gr.update(), gr.update(), gr.update(), "Error: No .safetensors files found"
        
        os.makedirs("lora", exist_ok=True)
        
        for file in safetensor_files:
            download_url = file.get('downloadUrl')
            if not download_url:
                print(f"Warning: No download URL for file {file.get('name')}")
                continue
                
            filename = os.path.join("lora", file['name'])
            print(f"Downloading {download_url} to {filename}")
            
            download_response = requests.get(download_url, headers=headers, stream=True)
            if download_response.status_code != 200:
                return gr.update(), gr.update(), gr.update(), f"Error downloading file: {download_response.status_code}"
            
            with open(filename, 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Successfully downloaded {filename}")
        
        updated_lora_files = get_updated_lora_files()
        return (
            gr.update(choices=updated_lora_files),
            gr.update(choices=updated_lora_files),
            gr.update(choices=updated_lora_files),
            f"Successfully downloaded model files to lora/{model_name}"
        )
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return gr.update(), gr.update(), gr.update(), f"Error: {str(e)}"

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
