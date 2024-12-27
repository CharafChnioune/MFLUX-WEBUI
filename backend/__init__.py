# backend/__init__.py

import os
import json

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create config file if it doesn't exist
config_path = os.path.join(os.path.dirname(__file__), "config.json")
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        json.dump({}, f)

# Import all managers
from .api_manager import *
from .civitai_manager import *
from .huggingface_manager import *
from .image_generation import *
from .lora_manager import *
from .mlx_vlm_manager import *
from .model_manager import *
from .ollama_manager import *
from .post_processing import *
from .prompts_manager import *

# Version
__version__ = "0.1.0"
