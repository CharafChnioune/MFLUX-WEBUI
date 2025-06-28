"""
MFLUX WebUI v0.9.0 - Stepwise Output Manager
Support for --stepwise-image-output-dir functionality
"""

import os
import time
from pathlib import Path
from PIL import Image
from typing import Optional, Union, Callable
import traceback


class StepwiseOutputManager:
    """Manager for stepwise image output during generation"""
    
    def __init__(self):
        self.enabled = False
        self.output_dir = None
        self.current_session = None
        self.step_counter = 0
        
    def setup_stepwise_output(self, output_dir: Union[str, Path], session_name: Optional[str] = None) -> bool:
        """Setup stepwise output directory and session"""
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create session subdirectory
            if session_name is None:
                session_name = f"session_{int(time.time())}"
            
            self.current_session = self.output_dir / session_name
            self.current_session.mkdir(parents=True, exist_ok=True)
            
            self.enabled = True
            self.step_counter = 0
            
            print(f"Stepwise output enabled: {self.current_session}")
            return True
            
        except Exception as e:
            print(f"Error setting up stepwise output: {str(e)}")
            self.enabled = False
            return False
    
    def save_step_image(self, image: Image.Image, step: int, description: str = "", 
                       seed: Optional[int] = None, extra_info: Optional[dict] = None) -> bool:
        """Save an intermediate step image"""
        if not self.enabled or not self.current_session:
            return False
            
        try:
            # Create filename
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            seed_str = f"_seed{seed}" if seed is not None else ""
            desc_str = f"_{description}" if description else ""
            filename = f"step_{step:03d}{seed_str}{desc_str}_{timestamp}.png"
            
            filepath = self.current_session / filename
            
            # Save image
            image.save(filepath, "PNG")
            
            # Save metadata if provided
            if extra_info:
                metadata_file = filepath.with_suffix('.json')
                import json
                metadata = {
                    "step": step,
                    "description": description,
                    "seed": seed,
                    "timestamp": timestamp,
                    "filename": filename,
                    **extra_info
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"Saved step {step} image: {filepath.name}")
            return True
            
        except Exception as e:
            print(f"Error saving step image: {str(e)}")
            traceback.print_exc()
            return False
    
    def create_step_callback(self, seed: Optional[int] = None) -> Callable:
        """Create a callback function for step-by-step image saving"""
        def step_callback(step: int, latents, **kwargs):
            """Callback function to save intermediate images"""
            if not self.enabled:
                return
                
            try:
                # Convert latents to image if possible
                # This is framework-specific and may need adjustment
                if hasattr(latents, 'to_pil'):
                    image = latents.to_pil()
                elif hasattr(latents, 'numpy'):
                    # Convert tensor to PIL Image
                    import numpy as np
                    array = latents.numpy()
                    if array.shape[-1] == 3:  # RGB
                        image = Image.fromarray((array * 255).astype('uint8'))
                    else:
                        # Skip if not RGB format
                        return
                else:
                    # Skip if we can't convert
                    return
                
                # Save the step image
                description = kwargs.get('description', f'intermediate')
                extra_info = {
                    'generation_step': step,
                    'total_steps': kwargs.get('total_steps', 'unknown'),
                    'guidance': kwargs.get('guidance', 'unknown'),
                    'model': kwargs.get('model', 'unknown')
                }
                
                self.save_step_image(image, step, description, seed, extra_info)
                
            except Exception as e:
                print(f"Error in step callback: {str(e)}")
        
        return step_callback
    
    def cleanup_session(self):
        """Clean up current session"""
        if self.current_session and self.enabled:
            print(f"Stepwise output session completed: {self.current_session}")
        
        self.enabled = False
        self.current_session = None
        self.step_counter = 0
    
    def get_session_images(self) -> list:
        """Get list of images from current session"""
        if not self.current_session or not self.current_session.exists():
            return []
        
        try:
            images = []
            for img_file in sorted(self.current_session.glob("*.png")):
                images.append(str(img_file))
            return images
        except Exception as e:
            print(f"Error getting session images: {str(e)}")
            return []
    
    def create_animation_from_steps(self, output_path: Optional[Union[str, Path]] = None, 
                                  duration: int = 500) -> Optional[str]:
        """Create an animated GIF from step images"""
        try:
            images = self.get_session_images()
            if len(images) < 2:
                print("Not enough step images to create animation")
                return None
            
            if output_path is None:
                output_path = self.current_session / "animation.gif"
            else:
                output_path = Path(output_path)
            
            # Load images
            pil_images = []
            for img_path in images[:20]:  # Limit to first 20 steps to avoid huge GIFs
                try:
                    img = Image.open(img_path)
                    # Resize if too large
                    if img.width > 512 or img.height > 512:
                        img.thumbnail((512, 512), Image.LANCZOS)
                    pil_images.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    continue
            
            if pil_images:
                # Save as animated GIF
                pil_images[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=duration,
                    loop=0
                )
                print(f"Created step animation: {output_path}")
                return str(output_path)
            
            return None
            
        except Exception as e:
            print(f"Error creating step animation: {str(e)}")
            traceback.print_exc()
            return None


# Global instance
stepwise_output_manager = StepwiseOutputManager()

def get_stepwise_output_manager():
    """Get the global stepwise output manager instance"""
    return stepwise_output_manager

def setup_stepwise_output(output_dir: str, session_name: str = None) -> bool:
    """Setup stepwise output for current generation"""
    return stepwise_output_manager.setup_stepwise_output(output_dir, session_name)

def save_step_image(image: Image.Image, step: int, description: str = "", 
                   seed: int = None, extra_info: dict = None) -> bool:
    """Save a step image during generation"""
    return stepwise_output_manager.save_step_image(image, step, description, seed, extra_info)

def cleanup_stepwise_output():
    """Cleanup stepwise output session"""
    stepwise_output_manager.cleanup_session()

def create_step_callback(seed: int = None):
    """Create callback for stepwise output"""
    return stepwise_output_manager.create_step_callback(seed)
