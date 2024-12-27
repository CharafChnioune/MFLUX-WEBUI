import os
import json
import requests
import gradio as gr
from typing import Dict, List, Optional, Union

def save_api_key(api_key: str, key_type: str = "openai") -> str:
    """
    Save API key to config file.
    
    Args:
        api_key: The API key to save
        key_type: The type of API key (e.g. "openai", "civitai")
        
    Returns:
        Success/error message
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        config = {}
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                
        config[f"{key_type}_api_key"] = api_key
        
        with open(config_path, "w") as f:
            json.dump(config, f)
            
        return "API key saved successfully"
        
    except Exception as e:
        return f"Error saving API key: {str(e)}"

def load_api_key(key_type: str = "openai") -> str:
    """
    Load API key from config file.
    
    Args:
        key_type: The type of API key to load
        
    Returns:
        The API key if found, empty string otherwise
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get(f"{key_type}_api_key", "")
                
        return ""
        
    except Exception as e:
        print(f"Error loading API key: {str(e)}")
        return ""

def test_api_key(api_key: str, api_type: str) -> str:
    """
    Test if an API key is valid.
    
    Args:
        api_key: The API key to test
        api_type: The type of API (e.g. "openai", "civitai")
        
    Returns:
        Success/error message
    """
    try:
        if not api_key:
            return "Error: API key is missing"
            
        if api_type == "openai":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers
            )
            if response.status_code == 200:
                return "API key is valid"
            else:
                return f"Error: Invalid API key (status code {response.status_code})"
                
        elif api_type == "civitai":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://civitai.com/api/v1/models",
                headers=headers
            )
            if response.status_code == 200:
                return "API key is valid"
            else:
                return f"Error: Invalid API key (status code {response.status_code})"
                
        else:
            return f"Error: Unsupported API type {api_type}"
            
    except Exception as e:
        return f"Error testing API key: {str(e)}"

def get_api_status() -> Dict[str, str]:
    """
    Get status of all API keys.
    
    Returns:
        Dictionary mapping API types to their status
    """
    status = {}
    
    for api_type in ["openai", "civitai"]:
        api_key = load_api_key(api_type)
        if api_key:
            result = test_api_key(api_key, api_type)
            status[api_type] = "Valid" if "valid" in result.lower() else "Invalid"
        else:
            status[api_type] = "Missing"
            
    return status
