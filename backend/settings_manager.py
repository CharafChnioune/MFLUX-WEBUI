import json
import os
from typing import Dict, Any, Optional

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'llm_settings.json')

def load_llm_settings(tab_name: str = "easy") -> dict:
    """
    Load LLM settings for a specific tab.
    
    Args:
        tab_name: Name of the tab to load settings for
        
    Returns:
        dict: Settings for the tab with empty defaults if not found
    """
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                
            tab_settings = settings.get(tab_name, {})
            
            return {
                'llm_type': tab_settings.get('llm_type', ''),
                'model': tab_settings.get('model', ''),
                'prompt': tab_settings.get('prompt', '')
            }
            
        return {
            'llm_type': '',
            'model': '',
            'prompt': ''
        }
        
    except Exception as e:
        print(f"Error loading LLM settings: {str(e)}")
        return {
            'llm_type': '',
            'model': '',
            'prompt': ''
        }

def save_llm_settings(llm_type: str, model: str, prompt_content: str, tab_name: str = "easy") -> bool:
    """
    Save LLM settings for a specific tab while preserving other tabs' settings.
    
    Args:
        llm_type: Type of LLM (Ollama or MLX)
        model: Name of the model
        prompt_content: Content of the prompt template
        tab_name: Name of the tab (easy, advanced, controlnet, image-to-image)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
        
        if tab_name not in settings:
            settings[tab_name] = {}
            
        settings[tab_name].update({
            'llm_type': llm_type,
            'model': model,
            'prompt': prompt_content
        })
        
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error saving LLM settings: {str(e)}")
        return False 