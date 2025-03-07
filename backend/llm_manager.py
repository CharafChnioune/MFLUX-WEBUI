import os
import json
import requests
from crewai import LLM
from typing import Optional, Dict, Any, List
from pathlib import Path
import subprocess

# Lijst van beschikbare LLM providers
LLM_PROVIDERS = [
    "openai", 
    "anthropic", 
    "google", 
    "groq",
    "cohere",
    "ollama",
    "azure",
    "aws",
    "mistral",
    "together",
    "local",
    "lmstudio"  # LM Studio toegevoegd
]

# Lijst van standaard beschikbare modellen per provider
DEFAULT_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
    "groq": ["llama-3-8b", "llama-3-70b", "mixtral-8x7b"],
    "cohere": ["command", "command-light", "command-r"],
    "ollama": ["llama3", "mistral", "mixtral"],
    "azure": ["gpt-4", "gpt-35-turbo"],
    "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
    "together": ["togethercomputer/llama-3-8b", "togethercomputer/llama-3-70b"],
    "local": ["localhost:8000"],
    "lmstudio": ["local-model", "openhermes", "llama"]  # LM Studio standaard modellen
}

# Standaard API-endpoints voor providers
API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com/openai/v1",
    "cohere": "https://api.cohere.ai/v1",
    "ollama": "http://localhost:11434",  # Aangepast voor direct gebruik
    "azure": "https://YOUR_RESOURCE_NAME.openai.azure.com",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz",
    "lmstudio": "http://localhost:1234/v1"  # LM Studio standaard endpoint
}

# Functie om modellen dynamisch op te halen indien mogelijk
def fetch_models_from_api(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[str]:
    """
    Haalt beschikbare modellen op van de API van de provider indien mogelijk.
    Fallback naar standaard modellen als API-aanroep niet lukt.
    """
    # Bepaal de juiste endpoint URL
    endpoint = base_url or API_ENDPOINTS.get(provider)
    if not endpoint:
        return get_fallback_models(provider)
    
    # Provider-specifieke API-aanroepen
    try:
        if provider == "openai":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = requests.get(f"{endpoint}/models", headers=headers, timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models if "gpt" in model["id"]]
                
        elif provider == "ollama":
            # Ollama API endpoint voor modellen
            api_endpoint = f"{endpoint}/api/tags"
            print(f"Querying Ollama API at: {api_endpoint}")
            response = requests.get(api_endpoint, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
                
        elif provider == "lmstudio":
            # LM Studio ondersteunt OpenAI-compatibele endpoints
            response = requests.get(f"{endpoint}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            
            # Als bovenstaande niet werkt, probeer via de terminal de modellen op te halen
            try:
                # Controleer of we actieve modellen kunnen opvragen
                result = subprocess.run(["curl", f"{endpoint}/models"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0 and "data" in result.stdout:
                    data = json.loads(result.stdout)
                    return [model["id"] for model in data.get("data", [])]
            except Exception as e:
                print(f"Kon LM Studio modellen niet ophalen via subprocess: {e}")
    
    except Exception as e:
        print(f"Kon modellen niet ophalen voor provider {provider}: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback naar standaard modellen
    return get_fallback_models(provider)

# Fallback modellen als dynamisch ophalen niet lukt
def get_fallback_models(provider: str) -> List[str]:
    """
    Geeft een lijst met standaard modellen voor een provider als fallback.
    """
    FALLBACK_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "groq": ["llama-3-8b", "llama-3-70b", "mixtral-8x7b"],
        "cohere": ["command", "command-light", "command-r"],
        "ollama": ["llama3", "mistral", "mixtral"],
        "azure": ["gpt-4", "gpt-35-turbo"],
        "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
        "together": ["togethercomputer/llama-3-8b", "togethercomputer/llama-3-70b"],
        "local": ["localhost:8000"],
        "lmstudio": ["local-model", "openhermes", "llama"]  # LM Studio standaard modellen
    }
    
    return FALLBACK_MODELS.get(provider, [])

def setup_llm(provider: str, model: str, api_key: Optional[str] = None, 
              base_url: Optional[str] = None, additional_params: Optional[Dict[str, Any]] = None) -> LLM:
    """
    Configureert een LLM voor gebruik met CrewAI.
    
    Args:
        provider: De LLM provider (openai, anthropic, enz.)
        model: Het specifieke model om te gebruiken
        api_key: De API-sleutel voor de provider (indien nodig)
        base_url: De basis-URL voor API-verzoeken (optioneel)
        additional_params: Extra parameters voor de LLM configuratie
        
    Returns:
        Een geconfigureerde LLM-instantie voor gebruik met CrewAI
    """
    print(f"Setting up LLM with provider: {provider}, model: {model}")
    
    try:
        # Specifieke configuratie voor LM Studio
        if provider == "lmstudio":
            # Expliciete config voor LiteLLM met OpenAI-compatibele endpoint
            api_key_value = api_key or "dummy-key"
            base_url_value = base_url or "http://localhost:1234/v1"
            
            # Zet de OpenAI API key
            os.environ["OPENAI_API_KEY"] = api_key_value
            
            # Gebruik het juiste model format voor LiteLLM
            model_name = f"openai/{model}"
            
            # Direct een LLM instantie maken met alle nodige parameters
            print(f"Using model: {model_name} with OpenAI compatibility for LM Studio")
            return LLM(
                model=model_name,  # Provider zit al in model naam
                api_key=api_key_value,
                base_url=base_url_value
            )
            
        # Specifieke configuratie voor Ollama
        elif provider == "ollama":
            # Zorg dat model naam correct is (met 'ollama/' prefix)
            model_name = model if model.startswith("ollama/") else f"ollama/{model}"
            base_url_value = base_url or "http://localhost:11434"
            
            # Direct een LLM instantie maken
            print(f"Using model: {model_name} with Ollama")
            return LLM(
                model=model_name,  # Provider zit al in model naam
                api_base=base_url_value
            )
            
        # Voor andere providers
        else:
            # Stel API sleutels in indien nodig
            if api_key:
                if provider == "openai":
                    os.environ["OPENAI_API_KEY"] = api_key
                elif provider == "anthropic":
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                elif provider == "google":
                    os.environ["GOOGLE_API_KEY"] = api_key
                elif provider == "groq":
                    os.environ["GROQ_API_KEY"] = api_key
                elif provider == "cohere":
                    os.environ["COHERE_API_KEY"] = api_key
                elif provider == "mistral":
                    os.environ["MISTRAL_API_KEY"] = api_key
                elif provider == "together":
                    os.environ["TOGETHER_API_KEY"] = api_key
            
            # Gebruik het juiste model format voor LiteLLM
            model_name = f"{provider}/{model}"
            
            # Log model format
            print(f"Using model: {model_name}")
            
            # Maak en retourneer de LLM instantie
            if base_url:
                return LLM(model=model_name, base_url=base_url)
            elif provider in API_ENDPOINTS:
                return LLM(model=model_name, base_url=API_ENDPOINTS[provider])
            else:
                return LLM(model=model_name)
    
    except Exception as e:
        import traceback
        print(f"Error setting up LLM: {e}")
        print(traceback.format_exc())
        
        # Fallback naar een lokale/standaard provider
        return LLM(
            model="local/local-model"
        )

def get_available_models(provider: str) -> list:
    """
    Haalt de beschikbare modellen op voor een bepaalde provider.
    Probeert eerst dynamisch op te halen, met fallback naar standaard modellen.
    
    Args:
        provider: De LLM provider
        
    Returns:
        Een lijst met beschikbare modelopties
    """
    # Voor LM Studio gebruiken we OpenAI-compatible API
    api_provider = "openai" if provider == "lmstudio" else provider
    
    try:
        # Probeer modellen dynamisch op te halen
        models = fetch_models_from_api(provider)
        if models:
            return models
    except Exception as e:
        print(f"Kon modellen niet dynamisch ophalen: {e}")
    
    # Fallback naar standaard modellen
    return get_fallback_models(provider) 