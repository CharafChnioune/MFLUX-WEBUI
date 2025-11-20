import gc
from functools import lru_cache
from mlx_vlm import load as load_vlm, generate as generate_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from pathlib import Path
import os
from mlx_lm import load as load_lm, generate as generate_lm
import json
from typing import List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import mlx.core as mx
import numpy as np
from io import BytesIO
import base64
import PIL
import sys
import traceback


class SimpleKVCache:
    """Simple key-value cache implementation"""
    def __init__(self):
        self.cache = {}

    def update(self, key, value):
        self.cache[key] = value

    def get(self, key, default=None):
        return self.cache.get(key, default)


class VisionModelWrapper:
    """Wrapper for vision models to make them compatible with text-only usage"""
    
    def __init__(self, model, processor):
        """Initialize wrapper with model and processor"""
        self.vision_model = model
        self.processor = processor
        
    def __getattr__(self, name):
        """Get attribute from vision model"""
        return getattr(self.vision_model, name)
        
    def __call__(self, *args, **kwargs):
        """Handle model inference"""
        return self.vision_model(*args, **kwargs)


@dataclass
class ModelInfo:
    """Information about a detected model"""
    name: str
    path: Path
    is_vision: bool
    model_type: str


def scan_huggingface_cache(cache_dir: Optional[str] = None) -> List[ModelInfo]:
    """Scan the Hugging Face cache directory for models."""
    if cache_dir is None:
        # Respect HF_HOME if provided, otherwise fall back to default
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(os.path.expanduser(hf_home), "hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Hugging Face cache directory not found: {cache_path}")
    
    models = []
    
    for model_dir in cache_path.glob("**/models--*"):
        try:
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            
            if "mlx" not in model_name.lower():
                continue
                
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue
                
            snapshot_dirs = list(snapshots_dir.iterdir())
            if not snapshot_dirs:
                continue
            latest_snapshot = snapshot_dirs[0]
            
            config_path = latest_snapshot / "config.json"
            if not config_path.exists():
                continue
                
            config_json = json.loads(config_path.read_text())
            
            is_vision_model = any([
                "vision_config" in config_json,
                "image_config" in config_json
            ])
            
            model_type = config_json.get("model_type", "unknown")
            
            models.append(ModelInfo(
                name=model_name,
                path=latest_snapshot,
                is_vision=is_vision_model,
                model_type=model_type
            ))
            
        except Exception as e:
            print(f"Error processing {model_dir}: {e}")
            continue
    
    return models


def classify_models(models: List[ModelInfo]) -> Tuple[List[ModelInfo], List[ModelInfo]]:
    """Split models into language models and vision-language models"""
    lm_models = [m for m in models if not m.is_vision]
    vlm_models = [m for m in models if m.is_vision]
    return lm_models, vlm_models


@lru_cache(maxsize=1)
def load_mlx_model(model_name: str) -> Tuple[Any, Any, Optional[dict]]:
    """
    Load MLX model with caching. Uses MLX-VLM for vision models and MLX-LM for language models.
    Returns: (model, processor, config) tuple
    """
    try:
        gc.collect()
        
        try:
            model, processor = load_vlm(
                model_name,
                processor_config={"trust_remote_code": True}
            )
            config = load_config(model_name)
            
            model = VisionModelWrapper(model, processor)
            return model, processor, config
            
        except Exception as e:
            model, tokenizer = load_lm(model_name)
            return model, tokenizer, None
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None, None, None


def generate_with_model(
    model: Union[VisionModelWrapper, Any],
    processor: Any,
    config: Optional[dict],
    prompt: str,
    images: Optional[List[str]] = None,
    max_tokens: int = 100,
    temperature: float = 0.0
) -> str:
    """
    Generate output with MLX model (VLM or LM).
    """
    try:
        if isinstance(model, VisionModelWrapper):
            if not images:
                raise ValueError("This model requires images. Use a text-only model or add an image.")
                
            if isinstance(images[0], str):
                if images[0].startswith("http"):
                    raise NotImplementedError("URL support not yet implemented")
                else:
                    img_data = base64.b64decode(images[0])
                    image = PIL.Image.open(BytesIO(img_data))
            else:
                image = images[0] 
            
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = apply_chat_template(
                processor, config, prompt, num_images=len(images)
            )
            
            output = generate_vlm(
                model=model,
                processor=processor,
                image=image,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=True
            )
            return output.strip()
            
        else:
            if images:
                print("Warning: Images are ignored when using a language model")
                
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = generate_lm(
                model=model,
                tokenizer=processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=True
            )
            return outputs.strip()
            
    except Exception as e:
        error_msg = str(e)
        if "without any images attached" in error_msg or "images are expected" in error_msg:
            raise ValueError(
                "This model requires images. Use a text-only model or add an image."
            ) from e
        print(f"Error generating output: {error_msg}")
        return f"Error: {error_msg}"


def get_available_mlx_models(model_type: str = "all") -> List[str]:
    """
    Get list of available MLX models.
    
    Args:
        model_type: Filter by type ('all', 'lm', or 'vlm')
    """
    try:
        models = scan_huggingface_cache()
        lm_models, vlm_models = classify_models(models)
        
        if model_type == "lm":
            return [m.name for m in lm_models]
        elif model_type == "vlm":
            return [m.name for m in vlm_models]
        else:
            return [m.name for m in models]
            
    except Exception as e:
        print(f"Error scanning models: {str(e)}")
        return []


def get_available_models(model_type: str = "all") -> List[str]:
    """Wrapper for get_available_mlx_models for backwards compatibility"""
    return get_available_mlx_models(model_type)


def is_vision_model(model_name: str) -> bool:
    """Check if model is a vision model via config"""
    try:
        config = load_config(model_name)
        return any([
            hasattr(config, "vision_config"),
            hasattr(config, "image_config")
        ])
    except Exception as e:
        print(f"Error checking model type: {str(e)}")
        return False


def get_available_mlx_vlm_models() -> List[str]:
    """Get list of available MLX Vision-Language models"""
    return get_available_mlx_models(model_type="vlm")


def generate_caption_with_mlx_vlm(image_path: str, model_name: str) -> str:
    """
    Generate a caption for an image using MLX-VLM.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the MLX-VLM model to use
        
    Returns:
        Generated caption text
    """
    try:
        model, processor, config = load_mlx_model(model_name)
        
        if not isinstance(model, VisionModelWrapper):
            raise ValueError("Model is not a vision model")
            
        from PIL import Image
        image = Image.open(image_path)
        
        caption = generate_with_model(
            model=model,
            processor=processor,
            config=config,
            prompt="Generate a detailed caption for this image:",
            images=[image],
            max_tokens=100,
            temperature=0.0
        )
        
        return caption.strip()
        
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        return ""
    finally:
        gc.collect()
