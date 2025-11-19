import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

class ConfigManager:
    """Manager for configuration files support from MFLUX v0.9.0"""
    
    def __init__(self):
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        self.default_config_file = self.config_dir / "default_config.json"
        self.current_config = self.load_default_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        default_config = {
            "generation": {
                "default_model": "dev",
                "default_steps": 25,
                "default_guidance": 7.0,
                "default_width": 1024,
                "default_height": 1024,
                "default_num_images": 1,
                "low_ram_mode": False,
                "save_metadata": True
            },
            "lora": {
                "library_path": os.environ.get("LORA_LIBRARY_PATH", "lora"),
                "default_scale": 1.0,
                "max_loras": 5
            },
            "system": {
                "output_directory": "output",
                "cache_directory": "cache",
                "auto_cleanup": True,
                "memory_management": "auto"
            },
            "ui": {
                "theme": "default",
                "show_advanced_options": True,
                "default_tab": "MFLUX Easy",
                "gallery_columns": 2
            },
            "integrations": {
                "ollama_enabled": False,
                "huggingface_cache": True,
                "civitai_enabled": True
            },
            "auto_seeds": {
                "enabled": False,
                "pool_size": 100,
                "shuffle": True
            },
            "quantization": {
                "default_bits": 4,
                "auto_quantize": False,
                "save_quantized": True
            }
        }
        
        # Load from file if exists
        if self.default_config_file.exists():
            try:
                with open(self.default_config_file, 'r') as f:
                    saved_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_config.update(saved_config)
            except Exception as e:
                print(f"Error loading default config: {e}")
        
        return default_config
    
    def save_default_config(self):
        """Save current configuration as default"""
        try:
            with open(self.default_config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
        except Exception as e:
            print(f"Error saving default config: {e}")
    
    def load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file (JSON or YAML)"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Error parsing config file {config_path}: {e}")
    
    def save_config_file(self, config: Dict[str, Any], config_path: Union[str, Path], format: str = "json"):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config, f, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving config file {config_path}: {e}")
    
    def create_preset_config(self, name: str, description: str = "") -> Path:
        """Create a preset configuration file"""
        preset_config = self.current_config.copy()
        preset_config["preset_info"] = {
            "name": name,
            "description": description,
            "created_at": str(Path(__file__).stat().st_mtime)
        }
        
        preset_file = self.config_dir / f"preset_{name.lower().replace(' ', '_')}.json"
        self.save_config_file(preset_config, preset_file)
        return preset_file
    
    def load_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Load a preset configuration"""
        preset_file = self.config_dir / f"preset_{preset_name.lower().replace(' ', '_')}.json"
        return self.load_config_file(preset_file)
    
    def get_available_presets(self) -> list:
        """Get list of available preset configurations"""
        presets = []
        for config_file in self.config_dir.glob("preset_*.json"):
            try:
                config = self.load_config_file(config_file)
                preset_info = config.get("preset_info", {})
                presets.append({
                    "name": preset_info.get("name", config_file.stem),
                    "description": preset_info.get("description", ""),
                    "file": str(config_file)
                })
            except Exception as e:
                print(f"Error reading preset {config_file}: {e}")
        return presets
    
    def apply_config(self, config: Dict[str, Any]):
        """Apply configuration to current session"""
        self.current_config.update(config)
    
    def get_config_value(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'generation.default_model')"""
        keys = key_path.split('.')
        value = self.current_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_config_value(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.current_config
        
        # Navigate to the parent of the final key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
    
    def export_config_template(self) -> str:
        """Export a configuration template with comments"""
        template = {
            "_comment": "MFLUX WebUI Configuration File",
            "_version": "0.9.3",
            "generation": {
                "_comment": "Default generation settings",
                "default_model": "dev",
                "default_steps": 25,
                "default_guidance": 7.0,
                "default_width": 1024,
                "default_height": 1024,
                "default_num_images": 1,
                "low_ram_mode": False,
                "save_metadata": True
            },
            "lora": {
                "_comment": "LoRA configuration",
                "library_path": "lora",
                "default_scale": 1.0,
                "max_loras": 5
            },
            "system": {
                "_comment": "System settings",
                "output_directory": "output",
                "cache_directory": "cache",
                "auto_cleanup": True,
                "memory_management": "auto"
            },
            "auto_seeds": {
                "_comment": "Auto seeds configuration",
                "enabled": False,
                "pool_size": 100,
                "shuffle": True
            }
        }
        
        return json.dumps(template, indent=2)
    
    def validate_config(self, config: Dict[str, Any]) -> list:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate generation settings
        gen_config = config.get("generation", {})
        if "default_steps" in gen_config:
            if not isinstance(gen_config["default_steps"], int) or gen_config["default_steps"] < 1:
                errors.append("generation.default_steps must be a positive integer")
        
        if "default_guidance" in gen_config:
            if not isinstance(gen_config["default_guidance"], (int, float)) or gen_config["default_guidance"] < 0:
                errors.append("generation.default_guidance must be a non-negative number")
        
        # Validate dimensions
        for dim in ["default_width", "default_height"]:
            if dim in gen_config:
                if not isinstance(gen_config[dim], int) or gen_config[dim] < 256:
                    errors.append(f"generation.{dim} must be an integer >= 256")
        
        return errors
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.current_config.copy()

# Global instance
config_manager = ConfigManager()

def get_config_manager():
    """Get the global configuration manager instance"""
    return config_manager

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file path"""
    return config_manager.load_config_file(config_path)

def get_config_value(key_path: str, default=None):
    """Get configuration value using dot notation"""
    return config_manager.get_config_value(key_path, default)

def set_config_value(key_path: str, value: Any):
    """Set configuration value using dot notation"""
    config_manager.set_config_value(key_path, value)
