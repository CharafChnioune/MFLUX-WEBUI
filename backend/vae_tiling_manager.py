"""
MFLUX WebUI v0.9.0 - VAE Tiling Manager
Support for --vae-tiling and --vae-tiling-split functionality
"""

import os
import math
from typing import Optional, Tuple, Union
import traceback
from PIL import Image
import numpy as np


class VAETilingManager:
    """Manager for VAE tiling functionality to handle large images"""
    
    def __init__(self):
        self.enabled = False
        self.tile_size = 512  # Default tile size
        self.overlap = 64     # Default overlap between tiles
        self.split_factor = 1 # Default split factor
        
    def setup_vae_tiling(self, enabled: bool = True, tile_size: int = 512, 
                        overlap: int = 64, split_factor: int = 1) -> bool:
        """Setup VAE tiling parameters"""
        try:
            self.enabled = enabled
            self.tile_size = max(256, tile_size)  # Minimum tile size
            self.overlap = max(0, min(overlap, tile_size // 4))  # Max 25% overlap
            self.split_factor = max(1, split_factor)
            
            if self.enabled:
                print(f"VAE tiling enabled: tile_size={self.tile_size}, overlap={self.overlap}, split_factor={self.split_factor}")
            else:
                print("VAE tiling disabled")
                
            return True
            
        except Exception as e:
            print(f"Error setting up VAE tiling: {str(e)}")
            self.enabled = False
            return False
    
    def calculate_tiles(self, width: int, height: int) -> list:
        """Calculate tile positions for the given dimensions"""
        if not self.enabled:
            return [(0, 0, width, height)]
        
        try:
            tiles = []
            
            # Calculate number of tiles needed in each dimension
            effective_tile_width = self.tile_size - self.overlap
            effective_tile_height = self.tile_size - self.overlap
            
            tiles_x = math.ceil(width / effective_tile_width)
            tiles_y = math.ceil(height / effective_tile_height)
            
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Calculate tile boundaries
                    start_x = x * effective_tile_width
                    start_y = y * effective_tile_height
                    
                    end_x = min(start_x + self.tile_size, width)
                    end_y = min(start_y + self.tile_size, height)
                    
                    # Ensure minimum tile size
                    if end_x - start_x < 256 or end_y - start_y < 256:
                        continue
                    
                    tiles.append((start_x, start_y, end_x, end_y))
            
            print(f"Generated {len(tiles)} tiles for {width}x{height} image")
            return tiles
            
        except Exception as e:
            print(f"Error calculating tiles: {str(e)}")
            return [(0, 0, width, height)]
    
    def split_image_for_vae(self, image: Union[Image.Image, np.ndarray]) -> list:
        """Split image into tiles for VAE processing"""
        if not self.enabled:
            return [image]
        
        try:
            if isinstance(image, Image.Image):
                width, height = image.size
                image_array = np.array(image)
            else:
                image_array = image
                height, width = image_array.shape[:2]
            
            tiles = self.calculate_tiles(width, height)
            tile_images = []
            
            for start_x, start_y, end_x, end_y in tiles:
                if isinstance(image, Image.Image):
                    tile_img = image.crop((start_x, start_y, end_x, end_y))
                else:
                    tile_img = image_array[start_y:end_y, start_x:end_x]
                
                tile_images.append({
                    'image': tile_img,
                    'position': (start_x, start_y, end_x, end_y),
                    'size': (end_x - start_x, end_y - start_y)
                })
            
            return tile_images
            
        except Exception as e:
            print(f"Error splitting image for VAE: {str(e)}")
            traceback.print_exc()
            return [{'image': image, 'position': (0, 0, width, height), 'size': (width, height)}]
    
    def merge_vae_tiles(self, processed_tiles: list, target_size: Tuple[int, int]) -> Union[Image.Image, np.ndarray]:
        """Merge processed VAE tiles back into a single image"""
        if not self.enabled or len(processed_tiles) == 1:
            return processed_tiles[0]['image']
        
        try:
            target_width, target_height = target_size
            
            # Determine output format based on first tile
            first_tile = processed_tiles[0]['image']
            if isinstance(first_tile, Image.Image):
                merged_image = Image.new('RGB', (target_width, target_height))
                is_pil = True
            else:
                merged_image = np.zeros((target_height, target_width, first_tile.shape[-1]), dtype=first_tile.dtype)
                is_pil = False
            
            # Track which pixels have been filled (for overlap handling)
            pixel_count = np.zeros((target_height, target_width))
            
            for tile_data in processed_tiles:
                tile_img = tile_data['image']
                start_x, start_y, end_x, end_y = tile_data['position']
                
                if is_pil:
                    # PIL Image merging
                    if isinstance(tile_img, np.ndarray):
                        tile_img = Image.fromarray(tile_img.astype('uint8'))
                    
                    # Simple paste for now (could be improved with blending)
                    merged_image.paste(tile_img, (start_x, start_y))
                else:
                    # NumPy array merging with overlap handling
                    if isinstance(tile_img, Image.Image):
                        tile_img = np.array(tile_img)
                    
                    # Handle overlap by averaging
                    current_count = pixel_count[start_y:end_y, start_x:end_x]
                    current_pixels = merged_image[start_y:end_y, start_x:end_x]
                    
                    # Weighted average for overlap regions
                    mask = current_count > 0
                    if np.any(mask):
                        current_pixels[mask] = (current_pixels[mask] * current_count[mask][..., np.newaxis] + 
                                              tile_img[mask]) / (current_count[mask][..., np.newaxis] + 1)
                        current_pixels[~mask] = tile_img[~mask]
                    else:
                        current_pixels[:] = tile_img
                    
                    merged_image[start_y:end_y, start_x:end_x] = current_pixels
                    pixel_count[start_y:end_y, start_x:end_x] += 1
            
            return merged_image
            
        except Exception as e:
            print(f"Error merging VAE tiles: {str(e)}")
            traceback.print_exc()
            # Return first tile as fallback
            return processed_tiles[0]['image'] if processed_tiles else None
    
    def apply_vae_tiling_to_config(self, config: dict) -> dict:
        """Apply VAE tiling settings to generation config"""
        if not self.enabled:
            return config
        
        try:
            config = config.copy()
            
            # Add VAE tiling parameters
            config['vae_tiling'] = True
            config['vae_tile_size'] = self.tile_size
            config['vae_overlap'] = self.overlap
            config['vae_split_factor'] = self.split_factor
            
            # Adjust memory-related settings
            if 'low_ram' not in config:
                config['low_ram'] = True  # Enable low RAM mode for large images
            
            return config
            
        except Exception as e:
            print(f"Error applying VAE tiling to config: {str(e)}")
            return config
    
    def should_use_tiling(self, width: int, height: int, memory_threshold: int = 1024*1024*4) -> bool:
        """Determine if tiling should be used based on image size"""
        try:
            # Calculate approximate memory usage (rough estimate)
            pixel_count = width * height
            memory_estimate = pixel_count * 4 * 3  # 4 bytes per channel, 3 channels
            
            # Use tiling if enabled and image is large enough
            return self.enabled and (memory_estimate > memory_threshold or 
                                   width > self.tile_size * 2 or 
                                   height > self.tile_size * 2)
            
        except Exception as e:
            print(f"Error determining tiling usage: {str(e)}")
            return False


# Global instance
vae_tiling_manager = VAETilingManager()

def get_vae_tiling_manager():
    """Get the global VAE tiling manager instance"""
    return vae_tiling_manager

def setup_vae_tiling(enabled: bool = True, tile_size: int = 512, 
                    overlap: int = 64, split_factor: int = 1) -> bool:
    """Setup VAE tiling for current generation"""
    return vae_tiling_manager.setup_vae_tiling(enabled, tile_size, overlap, split_factor)

def should_use_vae_tiling(width: int, height: int) -> bool:
    """Check if VAE tiling should be used for the given dimensions"""
    return vae_tiling_manager.should_use_tiling(width, height)

def apply_vae_tiling_config(config: dict) -> dict:
    """Apply VAE tiling settings to generation config"""
    return vae_tiling_manager.apply_vae_tiling_to_config(config)
