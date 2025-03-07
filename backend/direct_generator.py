"""
Direct image generator fallback for when CrewAI fails
"""
import os
import time
from .flux_manager import get_or_create_flux, generate_image_batch
from mflux.config.config import Config

def generate_direct_image(topic: str, num_images: int = 1, model: str = "schnell-4-bit", steps: int = 4, 
                         lora_files = None, lora_scales = None):
    """
    Generates an image directly without using CrewAI.
    Simple fallback for when CrewAI fails.
    """
    try:
        # Create a simple prompt from the topic
        prompt = f"""High quality, photorealistic image for a magazine cover featuring the topic: {topic}.
                    The image should be well-lit with balanced composition and professional studio quality.
                    Detailed, sharp focus, and vibrant colors. Suitable for a magazine cover."""
                    
        # Ensure the output directory exists
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the appropriate number of steps based on the model
        if steps is None:
            steps = 4 if "schnell" in model else 20
        
        # Create Flux instance
        flux = get_or_create_flux(model, None, None, lora_files, lora_scales)
        
        print(f"Direct generating image with: {model}, prompt: {topic}")
        
        # Generate images
        images, filenames, seeds = generate_image_batch(
            flux=flux,
            prompt=prompt,
            seed=None,
            steps=steps,
            height=1024,
            width=576,
            guidance=1.0 if "schnell" in model else 7.5,
            num_images=num_images
        )
        
        # Wait a moment to ensure the files are saved
        time.sleep(1)
        
        # Format the result
        result = f"Direct generated {len(filenames)} images for topic: {topic}\n"
        result += "\n".join(filenames)
        
        return result
        
    except Exception as e:
        import traceback
        error_tb = traceback.format_exc()
        error_msg = f"Error in direct generator: {str(e)}\n\n{error_tb}"
        print(error_msg)
        return error_msg