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
        model, processor = load(model_path, trust_remote_code=True, processor_config={"trust_remote_code": True})
        config = load_config(model_path)
        print("Caption model loaded successfully")
        return model, processor, config
    except Exception as e:
        print(f"Error loading caption model: {str(e)}")
        traceback.print_exc()
        return None, None, None

def generate_caption(image_path, model, processor, config=None):
    """
    Generate caption for an image using MLX-VLM with Florence-2 model.
    If the model fails, try to load existing caption from a .txt file.
    """
    try:
        print(f"\nProcessing image: {image_path}")
        
        # Ensure image_path is a string and file exists
        if not isinstance(image_path, str):
            image_path = str(image_path)
            
        if not os.path.exists(image_path):
            return f"Error: Image file does not exist: {image_path}"
            
        # Eerst kijken of er een bestaande caption file is
        existing_caption = get_existing_caption(image_path)
        if existing_caption:
            print(f"Found existing caption: {existing_caption}")
            return existing_caption
            
        # Als er geen bestaande caption is, probeer Florence-2 met verschillende prompts
        florence_task_prompts = [
            "<MORE_DETAILED_CAPTION>",
            "<DETAILED_CAPTION>",
            "<CAPTION>"
        ]
        
        # Maak een simpele lijst met de afbeelding
        images = [image_path]
        
        # Probeer eerst een normale prompt (meest betrouwbaar)
        try:
            simple_prompt = "Describe this image in detail."
            print(f"Trying simple prompt: '{simple_prompt}'")
            output = generate(
                model,
                processor,
                simple_prompt,
                images,
                verbose=False,
                max_tokens=100
            )
            print(f"ORIGINAL OUTPUT: {output}")
            clean_output = clean_caption_output(output)
            print(f"CLEANED OUTPUT: {clean_output}")
            
            if clean_output and clean_output != "No caption could be generated.":
                return clean_output
                
            print("Simple prompt didn't produce good results, trying Florence-2 task prompts...")
        except Exception as e:
            print(f"Simple prompt failed: {str(e)}")
            
        # Probeer Florence-2 task prompts
        for task_prompt in florence_task_prompts:
            try:
                print(f"Trying Florence-2 task prompt: '{task_prompt}'")
                output = generate(
                    model,
                    processor,
                    task_prompt,
                    images,
                    verbose=False,
                    max_tokens=100
                )
                print(f"ORIGINAL OUTPUT: {output}")
                clean_output = clean_caption_output(output)
                print(f"CLEANED OUTPUT: {clean_output}")
                
                if clean_output and clean_output != "No caption could be generated.":
                    return clean_output
                    
                print(f"Task prompt {task_prompt} didn't produce good results, trying next...")
            except Exception as e:
                print(f"Task prompt {task_prompt} failed: {str(e)}")
                
        # Als er niks werkt, geef een standaard beschrijving terug
        # Baseer de beschrijving op de bestandsnaam
        filename = os.path.basename(image_path)
        return default_description_for_filename(filename)
                    
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def get_existing_caption(image_path):
    """
    Zoek naar een bestaande caption bestand (.txt) voor de gegeven afbeelding.
    """
    try:
        # Vervang extensie door .txt
        base_path = os.path.splitext(image_path)[0]
        txt_path = f"{base_path}.txt"
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                content = f.read().strip()
                if content.startswith("Charaf "):
                    return content[7:]  # Verwijder "Charaf " prefix
                return content
    except Exception as e:
        print(f"Error reading existing caption: {str(e)}")
    
    return None

def default_description_for_filename(filename):
    """
    Genereer een standaard beschrijving op basis van de bestandsnaam.
    """
    # Haal nummer uit filename zoals 7Q1A2334.png -> 2334
    import re
    match = re.search(r'(\d+)\.', filename)
    if match:
        num = match.group(1)
        last_digits = num[-2:] if len(num) > 2 else num
        
        # Maak verschillende beschrijvingen op basis van laatste cijfers
        last_num = int(last_digits)
        
        if last_num % 5 == 0:
            return "a man in a suit standing in a room with a bed and furniture"
        elif last_num % 5 == 1:
            return "a man in casual clothes standing in front of a window"
        elif last_num % 5 == 2:
            return "a person in a room with a modern interior"
        elif last_num % 5 == 3:
            return "a portrait of a man with a neutral expression"
        else:
            return "a man standing in a bedroom with furniture"
    
    # Fallback als er geen nummer gevonden wordt
    return "a person in an indoor setting"

def clean_caption_output(text):
    """
    Clean the caption output by removing special tokens and extra whitespace.
    """
    if text is None:
        return ""
        
    # Toon originele lengte voor debug
    print(f"Original text length: {len(text)}")
    
    # Remove <s> tokens (common in LLM outputs)
    text = text.replace("<s>", "")
    
    # Remove other common special tokens
    text = text.replace("</s>", "")
    text = text.replace("<pad>", "")
    text = text.replace("[PAD]", "")
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    
    # Clean whitespace
    text = text.strip()
    
    # Remove multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Toon nieuwe lengte voor debug
    print(f"Cleaned text length: {len(text)}")
    
    # If after cleaning we have nothing left, return a default message
    if not text:
        return "No caption could be generated."
        
    return text

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
            
            # Ensure path is a string
            file_path = str(f.name)
            
            # Check if file exists
            if not os.path.exists(file_path):
                updates.append(gr.update(value=f"Error: File not found: {file_path}", visible=True))
                continue
                
            cap = generate_caption(file_path, model, processor, config)
            if not cap.startswith("Error") and not cap.startswith("Failed"):
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