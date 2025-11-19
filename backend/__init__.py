# backend/__init__.py

import os
import sys
import json
import importlib.util
from pathlib import Path

# Ensure the local mflux checkout is importable so every backend module can
# reference `mflux.*` even when the package is not pip-installed. Per the plan,
# we treat `mflux-main/src` as the canonical reference implementation.
repo_root = Path(__file__).resolve().parents[1]
local_mflux = repo_root / "mflux-main" / "src"
if local_mflux.exists():
    local_path = str(local_mflux)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

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
