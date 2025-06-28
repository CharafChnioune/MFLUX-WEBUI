"""
MFLUX WebUI v0.9.0 - Metadata Config Manager
Support for --config-from-metadata functionality
"""

import json
import os
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback


class MetadataConfigManager:
    """Manager for loading configuration from image metadata"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.webp']
        
    def extract_metadata_from_image(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Extract metadata from an image file"""
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                print(f"Error: Image file {image_path} does not exist")
                return None
                
            if image_path.suffix.lower() not in self.supported_formats:
                print(f"Error: Unsupported image format {image_path.suffix}")
                return None
                
            with Image.open(image_path) as img:
                metadata = {}
                
                # Try to get PNG metadata (text chunks)
                if hasattr(img, 'text') and img.text:
                    for key, value in img.text.items():
                        if key.startswith('mflux_'):
                            # Parse JSON metadata
                            try:
                                metadata[key] = json.loads(value)
                            except:
                                metadata[key] = value
                        else:
                            metadata[key] = value
                
                # Try to get EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id in exif:
                        tag = TAGS.get(tag_id, tag_id)
                        data = exif.get(tag_id)
                        if isinstance(data, str) and data.startswith('{'):
                            try:
                                metadata[tag] = json.loads(data)
                            except:
                                metadata[tag] = data
                        else:
                            metadata[tag] = data
                
                return metadata if metadata else None
                
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {str(e)}")
            traceback.print_exc()
            return None
    
    def load_config_from_metadata(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load generation config from image metadata"""
        try:
            metadata = self.extract_metadata_from_image(image_path)
            if not metadata:
                return None
            
            config = {}
            
            # Standard MFLUX metadata fields
            mapping = {
                'prompt': ['prompt', 'mflux_prompt', 'UserComment'],
                'model': ['model', 'mflux_model'],
                'seed': ['seed', 'mflux_seed'],
                'steps': ['steps', 'mflux_steps', 'num_inference_steps'],
                'guidance': ['guidance', 'mflux_guidance', 'guidance_scale'],
                'width': ['width', 'mflux_width'],
                'height': ['height', 'mflux_height'],
                'lora_files': ['lora_files', 'mflux_lora_files'],
                'lora_scales': ['lora_scales', 'mflux_lora_scales'],
                'quantize': ['quantize', 'mflux_quantize'],
                'low_ram': ['low_ram', 'mflux_low_ram']
            }
            
            for config_key, metadata_keys in mapping.items():
                for meta_key in metadata_keys:
                    if meta_key in metadata:
                        value = metadata[meta_key]
                        
                        # Parse JSON strings if needed
                        if isinstance(value, str) and value.startswith('['):
                            try:
                                value = json.loads(value)
                            except:
                                pass
                        
                        # Type conversions
                        if config_key in ['seed', 'steps', 'width', 'height']:
                            try:
                                value = int(value)
                            except:
                                continue
                        elif config_key in ['guidance']:
                            try:
                                value = float(value)
                            except:
                                continue
                        elif config_key in ['low_ram']:
                            if isinstance(value, str):
                                value = value.lower() in ['true', '1', 'yes', 'on']
                            else:
                                value = bool(value)
                        
                        config[config_key] = value
                        break
            
            return config if config else None
            
        except Exception as e:
            print(f"Error loading config from metadata: {str(e)}")
            traceback.print_exc()
            return None
    
    def apply_config_to_generation(self, config: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply loaded config to current generation parameters"""
        try:
            updated_config = current_config.copy()
            
            # Apply loaded values, but allow user overrides for key parameters
            for key, value in config.items():
                if key in updated_config:
                    # Only override if current value is empty/default
                    current_value = updated_config[key]
                    
                    # Define what constitutes "empty" for each parameter type
                    is_empty = False
                    if key == 'prompt':
                        is_empty = not current_value or current_value.strip() == ""
                    elif key in ['seed']:
                        is_empty = current_value is None or current_value == 0
                    elif key in ['steps', 'width', 'height']:
                        is_empty = current_value is None or current_value == 0
                    elif key in ['guidance']:
                        is_empty = current_value is None or current_value == ""
                    elif key in ['lora_files', 'lora_scales']:
                        is_empty = not current_value or current_value == []
                    else:
                        is_empty = not current_value
                    
                    if is_empty:
                        updated_config[key] = value
                        print(f"Applied metadata config: {key} = {value}")
                else:
                    updated_config[key] = value
                    print(f"Added metadata config: {key} = {value}")
            
            return updated_config
            
        except Exception as e:
            print(f"Error applying config: {str(e)}")
            return current_config
    
    def get_available_metadata_files(self, directory: Union[str, Path] = None) -> list:
        """Get list of image files with metadata in a directory"""
        try:
            if directory is None:
                directory = Path("output")
            else:
                directory = Path(directory)
            
            if not directory.exists():
                return []
            
            metadata_files = []
            for ext in self.supported_formats:
                for file_path in directory.glob(f"*{ext}"):
                    if self.extract_metadata_from_image(file_path):
                        metadata_files.append(str(file_path))
            
            return sorted(metadata_files)
            
        except Exception as e:
            print(f"Error getting metadata files: {str(e)}")
            return []


# Global instance
metadata_config_manager = MetadataConfigManager()

def get_metadata_config_manager():
    """Get the global metadata config manager instance"""
    return metadata_config_manager

def load_config_from_metadata(image_path: Union[str, Path]) -> Optional[dict]:
    """Load generation config from image metadata (main interface)"""
    return metadata_config_manager.load_config_from_metadata(image_path)

def load_config_from_image_metadata(image_path: Union[str, Path]) -> Optional[dict]:
    """Alias for load_config_from_metadata (used in flux_manager.py)"""
    return load_config_from_metadata(image_path)

def apply_metadata_config(loaded_config: dict, current_params: dict) -> dict:
    """Apply loaded metadata config to current parameters"""
    return metadata_config_manager.apply_config_to_params(loaded_config, current_params)
