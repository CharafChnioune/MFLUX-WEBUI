import os
import gc
import traceback
from PIL import Image
from functools import lru_cache
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import gradio as gr

@lru_cache(maxsize=1)
def load_caption_model(model_path="mlx-community/Florence-2-large-ft-bf16"):
    """
    Load MLX-VLM model for image captioning.
    """
    try:
        print(f"Loading caption model: {model_path}")
        model, processor = load(model_path, processor_config={"trust_remote_code": True})
        config = load_config(model_path)
        print("Caption model loaded successfully")
        return model, processor, config
    except Exception as e:
        print(f"Error loading caption model: {str(e)}")
        traceback.print_exc()
        return None, None, None

def generate_caption(image_path, model, processor, config):
    """
    Generate caption for an image using MLX-VLM.
    """
    try:
        print(f"\nProcessing image: {image_path}")
        prompt = "<DETAILED_CAPTION>"
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        
        output = generate(
            model,
            processor,
            [image_path],
            formatted_prompt,
            verbose=False,
            max_tokens=100, 
            temp=0.0,
            num_beams=1,
            top_p=0.9,
            repetition_penalty=1.0
        )
        return output.strip()
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        return f"Error: {str(e)}"

def show_uploaded_images(uploaded_files):
    """Show uploaded images in the UI."""
    updates = []
    for i, f in enumerate(uploaded_files[:20]):
        updates.extend([
            gr.update(value=f.name, visible=True),   
            gr.update(value="", visible=True) 
        ])
    
    for _ in range(20 - len(uploaded_files[:20])):
        updates.extend([
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False)
        ])
    return updates

def fill_captions(uploaded_files, model_name, prompt_for_images):
    """Generate captions for uploaded images."""
    updates = []
    if not uploaded_files:
        for _ in range(20):
            updates.extend([gr.update(value=None, visible=False), gr.update(value="", visible=False)])
        return updates

    model, processor, config = load_caption_model(model_name)
    if not model:
        for _ in range(20):
            updates.extend([
                gr.update(value=None, visible=False),
                gr.update(value="Error: Could not load model", visible=True)
            ])
        return updates

    for i, f in enumerate(uploaded_files[:20]):
        try:
            updates.append(gr.update(value=f.name, visible=True))
            
            cap = generate_caption(f.name, model, processor, config)
            if not cap.startswith("Error"):
                cap = cap.rstrip('.')
                if cap:
                    cap = cap[0].lower() + cap[1:]
                cap = f"{prompt_for_images} {cap}"
            updates.append(gr.update(value=cap, visible=True))
            
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            updates.extend([
                gr.update(value=None, visible=False),
                gr.update(value=f"Error: {str(e)}", visible=True)
            ])

    for _ in range(len(uploaded_files[:20]), 20):
        updates.extend([gr.update(value=None, visible=False), gr.update(value="", visible=False)])
    
    return updates 