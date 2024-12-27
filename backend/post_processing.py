import numpy as np
from PIL import Image
import gradio as gr

def update_dimensions_on_image_change(image):
    """Update dimensions when image changes."""
    if image is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    width, height = image.size
    return width, height, width, height

def update_dimensions_on_scale_change(scale_factor, original_width, original_height):
    """Update dimensions when scale factor changes."""
    if original_width is None or original_height is None:
        return gr.update(), gr.update()
    
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor
    
    new_width = int(round(new_width / 5) * 5)
    new_height = int(round(new_height / 5) * 5)
    
    new_width = max(5, new_width)
    new_height = max(5, new_height)
    
    return new_width, new_height

def update_height_with_aspect_ratio(width: float, original_width: float, original_height: float) -> gr.update:
    """
    Update height while maintaining aspect ratio
    """
    if not width or not original_width or not original_height:
        return gr.update(value=None)
    new_height = round((width * original_height) / original_width)
    new_height = new_height - (new_height % 16)
    return gr.update(value=new_height)

def update_width_with_aspect_ratio(height: float, original_width: float, original_height: float) -> gr.update:
    """
    Update width while maintaining aspect ratio
    """
    if not height or not original_width or not original_height:
        return gr.update(value=None)
    new_width = round((height * original_width) / original_height)
    new_width = new_width - (new_width % 16)
    return gr.update(value=new_width)

def scale_dimensions(image_format):
    """Get dimensions based on image format."""
    formats = {
        "Portrait (576x1024)": (576, 1024),
        "Landscape (1024x576)": (1024, 576),
        "Background (1920x1080)": (1920, 1080),
        "Square (1024x1024)": (1024, 1024),
        "Poster (1080x1920)": (1080, 1920),
        "Wide Screen (2560x1440)": (2560, 1440),
        "Ultra Wide Screen (3440x1440)": (3440, 1440),
        "Banner (728x90)": (728, 90)
    }
    return formats.get(image_format, (576, 1024))

def update_guidance_visibility(model):
    """Show/hide guidance scale based on model type."""
    return gr.update(visible=True)