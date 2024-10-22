import gradio as gr
import time
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import ollama
import json
from functools import partial
import mlx.core as mx
import gc
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import re
import subprocess
from mflux.config.config import Config, ConfigControlnet
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder

LORA_DIR = os.path.join(os.path.dirname(__file__), "lora")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

flux_cache = {}

class CustomModelConfig:
    def __init__(self, model_name, alias, num_train_steps, max_sequence_length):
        self.model_name = model_name
        self.alias = alias
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length

    @staticmethod
    def from_alias(alias):
        return get_custom_model_config(alias)

def get_custom_model_config(model_alias):
    models = {
        "dev": CustomModelConfig("AITRADER/MFLUXUI.1-dev", "dev", 1000, 512),
        "schnell": CustomModelConfig("AITRADER/MFLUXUI.1-schnell", "schnell", 1000, 256),
        "dev-8-bit": CustomModelConfig("AITRADER/MFLUX.1-dev-8-bit", "dev-8-bit", 1000, 512),
        "dev-4-bit": CustomModelConfig("AITRADER/MFLUX.1-dev-4-bit", "dev-4-bit", 1000, 512),
        "schnell-8-bit": CustomModelConfig("AITRADER/MFLUX.1-schnell-8-bit", "schnell-8-bit", 1000, 256),
        "schnell-4-bit": CustomModelConfig("AITRADER/MFLUX.1-schnell-4-bit", "schnell-4-bit", 1000, 256),
    }
    config = models.get(model_alias)
    if config is None:
        raise ValueError(f"Invalid model alias: {model_alias}. Available aliases are: {', '.join(models.keys())}")
    return config

from huggingface_hub import snapshot_download

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length):
    try:
        local_dir = os.path.join(MODELS_DIR, alias)
        snapshot_download(repo_id=hf_model_name, local_dir=local_dir, local_dir_use_symlinks=False)
        
        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length)
        get_custom_model_config.__globals__['models'][alias] = new_config
        
        return f"Model {hf_model_name} succesvol gedownload en opgeslagen als {alias}"
    except Exception as e:
        return f"Fout bij het downloaden van het model: {str(e)}"

flux_cache = {}

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length):
    try:
        model_dir = os.path.join(MODELS_DIR, alias)
        os.makedirs(model_dir, exist_ok=True)

        downloaded_files = snapshot_download(repo_id=hf_model_name, local_dir=model_dir)
        
        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length)
        get_custom_model_config.__globals__['models'][alias] = new_config
        
        return f"Model {hf_model_name} successfully downloaded and saved as {alias} in {model_dir}. Downloaded files: {len(downloaded_files)}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_or_create_flux(model, quantize, path, lora_paths_tuple, lora_scales_tuple, is_controlnet=False):
    lora_paths = list(lora_paths_tuple) if lora_paths_tuple else None
    lora_scales = list(lora_scales_tuple) if lora_scales_tuple else None
    
    FluxClass = Flux1Controlnet if is_controlnet else Flux1
    
    base_model = model.replace("-8-bit", "").replace("-4-bit", "")
    
    try:
        custom_config = get_custom_model_config(base_model)
        if base_model in ["dev", "schnell", "dev-8-bit", "dev-4-bit", "schnell-8-bit", "schnell-4-bit"]:
            model_path = None
        else:
            model_path = os.path.join(MODELS_DIR, base_model)
    except ValueError:
        custom_config = CustomModelConfig(base_model, base_model, 1000, 512)
        model_path = os.path.join(MODELS_DIR, base_model)
    
    if "-8-bit" in model:
        quantize = 8
    elif "-4-bit" in model:
        quantize = 4
    
    flux = FluxClass(
        model_config=custom_config,
        quantize=quantize,
        local_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    
    return flux

def get_available_lora_files():
    return [(str(f), f.stem) for f in Path(LORA_DIR).rglob("*.safetensors")]

def get_available_models():
    standard_models = ["schnell-4-bit", "schnell-8-bit", "dev-4-bit", "dev-8-bit", "schnell", "dev"]
    custom_models = [f.name for f in Path(MODELS_DIR).iterdir() if f.is_dir()]
    return standard_models + custom_models

def ensure_llama_model(model_name):
    try:
        ollama.pull(model_name)
        return True
    except Exception:
        return False

def load_ollama_settings():
    try:
        with open('ollama_settings.json', 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        _, default_model = get_available_ollama_models()
        settings = {'model': default_model}
    
    settings['system_prompt'] = read_system_prompt()
    return settings

def create_ollama_settings():
    settings = load_ollama_settings()
    available_models, _ = get_available_ollama_models()
    ollama_model = gr.Dropdown(
        choices=available_models,
        value=settings['model'],
        label="Ollama Model"
    )
    system_prompt = gr.Textbox(
        label="System Prompt", lines=10, value=settings['system_prompt']
    )
    save_button = gr.Button("Save Ollama Settings")
    return [ollama_model, system_prompt, save_button]

def save_settings(model, prompt):
    save_ollama_settings(model, prompt)
    gr.Info("Settings saved!")
    model_update = gr.update(choices=get_available_ollama_models(), value=model)
    return gr.update(open=False)

def enhance_prompt(prompt, ollama_model, system_prompt):
    print(f"prompt={prompt}, model={ollama_model}, system_prompt={system_prompt}")
    try:
        response = ollama.generate(
            model=ollama_model,
            prompt=f"Enhance this prompt for an image generation AI: {prompt}",
            system=system_prompt,
            options={"temperature": 0.7}
        )
        enhanced_prompt = response['response'].strip()
        gr.Info(f"Prompt successfully improved with model {ollama_model}.")
        return enhanced_prompt
    except Exception as e:
        gr.Error(f"Error while improving prompt: {str(e)}")
        return prompt

def print_memory_usage(label):
    try:
        active_memory = mx.metal.get_active_memory() / 1e6
        peak_memory = mx.metal.get_peak_memory() / 1e6
        print(f"{label} - Active memory: {active_memory:.2f} MB, Peak memory: {peak_memory:.2f} MB")
    except AttributeError:
        print(f"{label} - Unable to get memory usage information")

def generate_image_gradio(
    prompt, model, seed, height, width, steps, guidance, lora_files, metadata, ollama_model, system_prompt
):
    print(f"\n--- Generating image (Advanced) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Dimensions: {height}x{width}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"LoRA files: {lora_files}")
    print_memory_usage("Before generation")

    start_time = time.time()

    valid_loras = process_lora_files(lora_files)
    lora_paths = valid_loras if valid_loras else None
    lora_scales = [1.0] * len(valid_loras) if valid_loras else None

    seed = None if seed == "" else int(seed)
    steps = None if steps == "" else int(steps)

    flux = get_or_create_flux(model, None, None, tuple(lora_paths) if lora_paths else None, tuple(lora_scales) if lora_scales else None)

    print_memory_usage("After creating flux")

    if steps is None:
        steps = 4 if model == "schnell" else 14

    timestamp = int(time.time())
    output_filename = f"generated_{timestamp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    image = flux.generate_image(
        seed=int(time.time()) if seed is None else seed,
        prompt=prompt,
        config=Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        ),
    )

    print_memory_usage("After generating image")

    image.image.save(output_path)

    print_memory_usage("After saving image")

    # Opruimen
    del flux
    del image
    gc.collect()
    force_mlx_cleanup()

    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    return image.image, output_filename, prompt

def generate_image_controlnet_gradio(
    prompt,
    control_image,
    model,
    seed,
    height,
    width,
    steps,
    guidance,
    controlnet_strength,
    lora_files,
    metadata,
    save_canny,
    ollama_model,
    system_prompt
):
    print(f"\n--- Generating image (ControlNet) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Dimensions: {height}x{width}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"ControlNet Strength: {controlnet_strength}")
    print(f"LoRA files: {lora_files}")
    print_memory_usage("Before generation")

    start_time = time.time()

    valid_loras = process_lora_files(lora_files)
    lora_paths = valid_loras if valid_loras else None
    lora_scales = [1.0] * len(valid_loras) if valid_loras else None

    if "dev" not in model:
        guidance = None

    seed = None if seed == "" else int(seed)
    steps = None if steps == "" else int(steps)

    flux = get_or_create_flux(model, None, None, tuple(lora_paths) if lora_paths else None, tuple(lora_scales) if lora_scales else None, is_controlnet=True)

    print_memory_usage("After creating flux")

    if steps is None:
        steps = 4 if model == "schnell" else 14

    timestamp = int(time.time())
    output_filename = f"generated_controlnet_{timestamp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        if control_image is None:
            raise ValueError("Control image is required for ControlNet generation")

        control_image_path = os.path.join(OUTPUT_DIR, f"control_image_{timestamp}.png")
        control_image.save(control_image_path)

        generated_image = flux.generate_image(
            seed=int(time.time()) if seed is None else seed,
            prompt=prompt,
            controlnet_image_path=control_image_path,
            config=ConfigControlnet(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                controlnet_strength=controlnet_strength,
            ),
            output=output_path
        )
        
        print_memory_usage("After generating image")
        
        os.remove(control_image_path)
        
        # Verwijder de flux-instantie en de afbeelding
        del flux
        del generated_image

        # Roep de garbage collector aan
        import gc
        gc.collect()

        # Roep de geheugenopruimingsfunctie aan
        force_mlx_cleanup()

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        return generated_image.image, f"Image generated successfully! Saved as {output_filename}", prompt
    except Exception as e:
        return None, f"Error generating image: {str(e)}", prompt

def process_lora_files(selected_loras):
    if not selected_loras:
        return []
    lora_files = get_available_lora_files()
    if not lora_files:
        return []
    lora_dict = dict(lora_files)
    valid_loras = []
    for lora in selected_loras:
        matching_loras = [path for path, name in lora_dict.items() if name == lora]
        if matching_loras:
            valid_loras.extend(matching_loras)
    return valid_loras

def save_quantized_model_gradio(model, quantize):
    quantize = int(quantize)
    try:
        custom_config = get_custom_model_config(model)
    except ValueError:
        custom_config = CustomModelConfig(model, model, 1000, 512)
    
    model_path = os.path.join(MODELS_DIR, model)
    
    flux = Flux1(
        model_config=custom_config,
        quantize=quantize,
        local_path=model_path
    )

    save_path = os.path.join(MODELS_DIR, f"{model}-{quantize}-bit")
    flux.save_model(save_path)

    updated_models = get_updated_models()
    return (
        gr.update(choices=updated_models),
        gr.update(choices=updated_models),
        gr.update(choices=updated_models),
        gr.update(choices=[m for m in updated_models if not m.endswith("-4-bit") and not m.endswith("-8-bit")]),
        f"Model gekwantiseerd en opgeslagen als {save_path}"
    )

def process_lora_files(selected_loras):
    if not selected_loras:
        return []
    lora_files = get_available_lora_files()
    if not lora_files:
        return []
    lora_dict = dict(lora_files)
    valid_loras = []
    for lora in selected_loras:
        matching_loras = [path for path, name in lora_dict.items() if name == lora]
        if matching_loras:
            valid_loras.extend(matching_loras)
    return valid_loras

def simple_generate_image(prompt, model, image_format, lora_files, ollama_model, system_prompt):
    print(f"\n--- Generating image ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Image Format: {image_format}")
    print(f"LoRA files: {lora_files}")
    print_memory_usage("Before generation")

    start_time = time.time()

    try:
        width, height = map(int, image_format.split('(')[1].split(')')[0].split('x'))

        valid_loras = process_lora_files(lora_files)
        lora_paths = valid_loras if valid_loras else None
        lora_scales = [1.0] * len(valid_loras) if valid_loras else None

        if "dev" in model:
            steps = 20
        else:
            steps = 4

        flux = get_or_create_flux(model, None, None, tuple(lora_paths) if lora_paths else None, tuple(lora_scales) if lora_scales else None)

        timestamp = int(time.time())
        output_filename = f"generated_simple_{timestamp}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        image = flux.generate_image(
            seed=int(time.time()),
            prompt=prompt,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=7.5,
            ),
        )

        print_memory_usage("After generating image")
        
        image.image.save(output_path)

        print_memory_usage("After saving image")
        
        del flux
        del image
        
        gc.collect()
        
        force_mlx_cleanup()

        print_memory_usage("After cleanup")

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        return output_path, output_filename, prompt

    except Exception as e:
        print(f"Error in image generation: {str(e)}")
        return None, None, prompt
    finally:
        force_mlx_cleanup()
        gc.collect()

def get_available_ollama_models():
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        return available_models, available_models[0] if available_models else None
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return [], None

def save_ollama_settings(model, system_prompt):
    with open('ollama_settings.json', 'w') as f:
        json.dump({'model': model}, f)
    
    with open('system_prompt.md', 'w') as f:
        f.write(system_prompt)

def read_system_prompt():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(script_dir, 'system_prompt.md')
        with open(system_prompt_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print("system_prompt.md niet gevonden. Een lege prompt wordt gebruikt.")
        return ""

def clear_flux_cache():
    global flux_cache
    
    flux_cache.clear()
    
    gc.collect()
    
    try:
        mx.metal.reset_peak_memory()
        
        mx.eval(mx.zeros(1))
        
        if hasattr(mx, 'clear_memory_pool'):
            mx.clear_memory_pool()
        
        if hasattr(mx, 'metal'):
            mx.metal.device_reset()
        
    except AttributeError as e:
        print(f"Waarschuwing: Sommige MLX geheugenbeheerfuncties zijn niet beschikbaar: {e}")
    
    gc.collect()
    
    print_memory_usage("After clearing flux cache")

def force_mlx_cleanup():
    mx.eval(mx.zeros(1))
    
    if hasattr(mx.metal, 'clear_cache'):
        mx.metal.clear_cache()
    
    if hasattr(mx.metal, 'reset_peak_memory'):
        mx.metal.reset_peak_memory()

    gc.collect()

def update_guidance_visibility(model):
    return gr.update(visible="dev" in model)

def save_api_key(api_key):
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    config["civitai_api_key"] = api_key
    with open(config_path, "w") as f:
        json.dump(config, f)
    return "API key saved successfully"

def load_api_key():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("civitai_api_key", "")
    return ""

def download_lora_model(page_url, api_key):
    if not api_key:
        return gr.update(), gr.update(), gr.update(), "Error: API key is missing"

    try:
        print(f"Starting download process for URL: {page_url}")
        model_id = re.search(r'/models/(\d+)', page_url)
        if not model_id:
            return gr.update(), gr.update(), gr.update(), f"Error: Could not extract model ID from the URL: {page_url}"

        model_id = model_id.group(1)
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        model_data = response.json()

        if not model_data.get('modelVersions'):
            return gr.update(), gr.update(), gr.update(), "Error: No model versions found"

        latest_version = model_data['modelVersions'][0]
        download_url = latest_version['downloadUrl']
        
        print(f"Download URL: {download_url}")

        download_response = requests.get(download_url, headers=headers, stream=True)
        download_response.raise_for_status()

        content_disposition = download_response.headers.get('Content-Disposition')
        if content_disposition:
            filename = re.findall("filename=(.+)", content_disposition)[0].strip('"')
        else:
            filename = f"model_{model_id}.safetensors"

        file_path = os.path.join(LORA_DIR, filename)
        total_size = int(download_response.headers.get('content-length', 0))
        
        print(f"Saving to: {file_path}")
        print(f"Total size: {total_size} bytes")

        with open(file_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in download_response.iter_content(chunk_size=8192):
                size = file.write(data)
                progress_bar.update(size)

        print(f"Download completed successfully: {filename}")
        
        updated_lora_files = get_updated_lora_files()
        return (
            gr.update(choices=updated_lora_files),
            gr.update(choices=updated_lora_files),
            gr.update(choices=updated_lora_files),
            f"Download completed successfully: {filename}"
        )

    except requests.exceptions.RequestException as e:
        error_message = f"Network error during download: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f"\nStatus code: {e.response.status_code}"
            error_message += f"\nResponse content: {e.response.text[:500]}..."
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), error_message

    except Exception as e:
        error_message = f"Unexpected error during download: {str(e)}"
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), error_message

def get_updated_lora_files():
    lora_files = get_available_lora_files()
    return [file[1] for file in lora_files]

def get_updated_models():
    return get_available_models()

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length, api_key):
    try:
        login_result = login_huggingface(api_key)
        if "Error" in login_result:
            return gr.update(), gr.update(), gr.update(), gr.update(), login_result

        model_dir = os.path.join(MODELS_DIR, alias)
        os.makedirs(model_dir, exist_ok=True)

        downloaded_files = snapshot_download(repo_id=hf_model_name, local_dir=model_dir, use_auth_token=api_key)
        
        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length)
        get_custom_model_config.__globals__['models'][alias] = new_config
        
        print(f"Model {hf_model_name} successfully downloaded and saved as {alias}")
        
        updated_models = get_updated_models()
        return (
            gr.update(choices=updated_models),
            gr.update(choices=updated_models),
            gr.update(choices=updated_models),
            gr.update(choices=[m for m in updated_models if not m.endswith("-4-bit") and not m.endswith("-8-bit")]),
            f"Model {hf_model_name} successfully downloaded and saved as {alias}"
        )

    except Exception as e:
        error_message = f"Error downloading model: {str(e)}"
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), gr.update(), error_message

def load_hf_api_key():
    try:
        with open('hf_api_key.json', 'r') as f:
            settings = json.load(f)
        return settings.get('api_key', '')
    except FileNotFoundError:
        return ''

def save_hf_api_key(api_key):
    with open('hf_api_key.json', 'w') as f:
        json.dump({'api_key': api_key}, f)

def login_huggingface(api_key):
    try:
        api = HfApi()
        api.set_access_token(api_key)
        HfFolder.save_token(api_key)
        return "Successfully logged in to Hugging Face"
    except Exception as e:
        return f"Error logging in to Hugging Face: {str(e)}"

def download_lora_model_huggingface(model_name, hf_api_key):
    if not model_name:
        return gr.update(), gr.update(), gr.update(), "Error: Model name is missing"

    try:
        api = HfApi(token=hf_api_key if hf_api_key else None)
        # Verkrijg de lijst met bestanden
        files = api.list_repo_files(repo_id=model_name)

        # Filter alleen de .safetensors bestanden
        safetensors_files = [f for f in files if f.endswith(".safetensors")]

        if not safetensors_files:
            return gr.update(), gr.update(), gr.update(), f"Error: No .safetensors files found in the model repository '{model_name}'"

        for filename in safetensors_files:
            # Download het bestand
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=LORA_DIR,
                use_auth_token=hf_api_key
            )

        print(f"Download voltooid: {safetensors_files}")
        updated_lora_files = get_updated_lora_files()
        return (
            gr.update(choices=updated_lora_files),
            gr.update(choices=updated_lora_files),
            gr.update(choices=updated_lora_files),
            f"Download voltooid: {', '.join(safetensors_files)}"
        )

    except Exception as e:
        error_message = f"Fout bij het downloaden van LoRA van HuggingFace: {str(e)}"
        print(f"Error: {error_message}")
        return gr.update(), gr.update(), gr.update(), error_message

def create_ui():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("MFLUX Easy", id=0):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            prompt_simple = gr.Textbox(label="Prompt", lines=2)
                            with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_simple:
                                ollama_components_simple = create_ollama_settings()
                            with gr.Row():
                                enhance_ollama_simple = gr.Button("Enhance prompt with Ollama")
                        
                        ollama_components_simple[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_simple[0], ollama_components_simple[1]],
                            outputs=[ollama_section_simple]
                        )
                        
                        model_simple = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell-4-bit"
                        )
                        image_format = gr.Dropdown(
                            choices=[
                                "Portrait (576x1024)",
                                "Landscape (1024x576)",
                                "Background (1920x1080)",
                                "Square (1024x1024)",
                                "Poster (1080x1920)",
                                "Wide Screen (2560x1440)",
                                "Ultra Wide Screen (3440x1440)",
                                "Banner (728x90)"
                            ],
                            label="Image Format",
                            value="Portrait (576x1024)"
                        )
                        lora_files_simple = gr.Dropdown(
                            choices=get_updated_lora_files(),
                            label="Select LoRA Files",
                            multiselect=True,
                            allow_custom_value=True,
                            value=[]
                        )
                        generate_button_simple = gr.Button("Generate Image")
                    
                    with gr.Column():
                        output_image_simple = gr.Image(label="Generated Image")
                        output_filename_simple = gr.Textbox(label="Saved Image Filename")

                enhance_ollama_simple.click(
                    fn=enhance_prompt,
                    inputs=[prompt_simple, ollama_components_simple[0], ollama_components_simple[1]],
                    outputs=prompt_simple
                )
                generate_button_simple.click(
                    fn=simple_generate_image,
                    inputs=[prompt_simple, model_simple, image_format, lora_files_simple, 
                            ollama_components_simple[0], ollama_components_simple[1]],
                    outputs=[output_image_simple, output_filename_simple, prompt_simple]
                )

            with gr.TabItem("Advanced Generate"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            prompt = gr.Textbox(label="Prompt", lines=2)
                            with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_adv:
                                ollama_components_adv = create_ollama_settings()
                            with gr.Row():
                                enhance_ollama = gr.Button("Enhance prompt with Ollama")
                        
                        ollama_components_adv[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_adv[0], ollama_components_adv[1]],
                            outputs=[ollama_section_adv]
                        )
                        
                        model = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell"
                        )
                        seed = gr.Textbox(label="Seed (optional)", value="")
                        with gr.Row():
                            width = gr.Number(label="Width", value=576, precision=0)
                            height = gr.Number(label="Height", value=1024, precision=0)
                        steps = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance = gr.Number(label="Guidance Scale", value=3.5, visible=False)
                        lora_files = gr.Dropdown(
                            choices=get_updated_lora_files(),
                            label="Select LoRA Files",
                            multiselect=True,
                            allow_custom_value=True,
                            value=[]
                        )
                        metadata = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        generate_button = gr.Button("Generate Image")
                    with gr.Column():
                        output_image = gr.Image(label="Generated Image")
                        output_filename = gr.Textbox(label="Saved Image Filename")

                model.change(
                    fn=update_guidance_visibility,
                    inputs=[model],
                    outputs=[guidance]
                )

                enhance_ollama.click(
                    fn=enhance_prompt,
                    inputs=[prompt, ollama_components_adv[0], ollama_components_adv[1]],
                    outputs=prompt
                )
                generate_button.click(
                    fn=generate_image_gradio,
                    inputs=[prompt, model, seed, width, height, steps, guidance, lora_files, metadata,
                            ollama_components_adv[0], ollama_components_adv[1]],
                    outputs=[output_image, output_filename, prompt]
                )

            with gr.TabItem("ControlNet"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            prompt_cn = gr.Textbox(label="Prompt", lines=2)
                            with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_cn:
                                ollama_components_cn = create_ollama_settings()
                            with gr.Row():
                                enhance_ollama_cn = gr.Button("Enhance prompt with Ollama")
                        
                        ollama_components_cn[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_cn[0], ollama_components_cn[1]],
                            outputs=[ollama_section_cn]
                        )
                        
                        control_image = gr.Image(label="Control Image", type="pil")
                        model_cn = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell"
                        )
                        seed_cn = gr.Textbox(label="Seed (optional)", value="")
                        with gr.Row():
                            width_cn = gr.Number(label="Width", value=576, precision=0)
                            height_cn = gr.Number(label="Height", value=1024, precision=0)
                        steps_cn = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance_cn = gr.Number(label="Guidance Scale", value=3.5, visible=False)
                        controlnet_strength = gr.Number(label="ControlNet Strength", value=0.7)
                        lora_files_cn = gr.Dropdown(
                            choices=get_updated_lora_files(),
                            label="Select LoRA Files",
                            multiselect=True,
                            allow_custom_value=True,
                            value=[]
                        )
                        metadata_cn = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        save_canny = gr.Checkbox(label="Save Canny Edge Detection Image", value=False)
                        generate_button_cn = gr.Button("Generate Image")
                    with gr.Column():
                        output_image_cn = gr.Image(label="Generated Image")
                        output_message_cn = gr.Textbox(label="Status")
                enhance_ollama_cn.click(
                    fn=enhance_prompt,
                    inputs=[prompt, ollama_components_cn[0], ollama_components_cn[1]],
                    outputs=prompt_cn
                )
                
                generate_button_cn.click(
                    fn=generate_image_controlnet_gradio,
                    inputs=[prompt_cn, control_image, model_cn, seed_cn, height_cn, width_cn, steps_cn, 
                            guidance_cn, controlnet_strength, lora_files_cn, metadata_cn, save_canny,
                            ollama_components_cn[0], ollama_components_cn[1]],
                    outputs=[output_image_cn, output_message_cn, prompt_cn]
                )

                gr.Markdown("""
                ⚠️ Note: Controlnet requires [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny), which was trained for the `dev` model. 
                It can work well with `schnell`, but performance is not guaranteed.

                ⚠️ Note: The output can be highly sensitive to the controlnet strength and is very much dependent on the reference image. 
                Too high settings will corrupt the image. A recommended starting point is a value like 0.4. Experiment with different strengths to find the best result.
                """)

            with gr.TabItem("Models"):

                gr.Markdown("### Download LoRA")
                lora_source = gr.Radio(
                    choices=["CivitAI", "HuggingFace"],
                    label="LoRA Source",
                    value="CivitAI"
                )
                lora_input = gr.Textbox(label="LoRA Model Page URL (CivitAI) or Model Name (HuggingFace)")

                api_key_status = gr.Markdown(value=f"API Key Status: {'Saved' if load_api_key() else 'Not saved'}")
                hf_api_key_status = gr.Markdown(value=f"HuggingFace API Key Status: {'Saved' if load_hf_api_key() else 'Not saved'}")

                with gr.Accordion("API Key Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            api_key_input = gr.Textbox(
                                label="CivitAI API Key", 
                                type="password", 
                                value=load_api_key()
                            )
                            hf_api_key_input = gr.Textbox(
                                label="HuggingFace API Key", 
                                type="password", 
                                value=load_hf_api_key()
                            )
                        with gr.Column(scale=1):
                            api_key_options = gr.Dropdown(
                                choices=["Save API Keys", "Clear API Keys"],
                                label="API Key Options",
                                type="index"
                            )
                    gr.Markdown("Don't have an API key? [CivitAI](https://civitai.com/user/account), [HuggingFace](https://huggingface.co/settings/tokens)")

                download_lora_button = gr.Button("Download LoRA")
                lora_download_status = gr.Textbox(label="Download Status", lines=0.5)

                def handle_api_keys(choice, api_key_civitai, api_key_hf):
                    if choice == 0:
                        save_api_key(api_key_civitai)
                        save_hf_api_key(api_key_hf)
                        return "API Key Status: Saved successfully", "API Key Status: Saved successfully"
                    elif choice == 1:
                        save_api_key("")
                        save_hf_api_key("")
                        return "API Key Status: Cleared", "API Key Status: Cleared"

                api_key_options.change(
                    fn=handle_api_keys,
                    inputs=[api_key_options, api_key_input, hf_api_key_input],
                    outputs=[api_key_status, hf_api_key_status]
                )

                def download_lora(model_url_or_name, api_key_civitai, api_key_hf, lora_source):
                    if lora_source == "CivitAI":
                        return download_lora_model(model_url_or_name, api_key_civitai)
                    elif lora_source == "HuggingFace":
                        return download_lora_model_huggingface(model_url_or_name, api_key_hf)
                    else:
                        return gr.update(), gr.update(), gr.update(), "Error: Invalid LoRA source selected"

                download_lora_button.click(
                    fn=download_lora,
                    inputs=[lora_input, api_key_input, hf_api_key_input, lora_source],
                    outputs=[lora_files_simple, lora_files, lora_files_cn, lora_download_status]
                )

                gr.Markdown("## Download and Add Model")
                with gr.Row():
                    with gr.Column(scale=2):
                        hf_model_name = gr.Textbox(label="Hugging Face Model Name")
                        alias = gr.Textbox(label="Model Alias")
                        num_train_steps = gr.Number(label="Number of Training Steps", value=1000)
                        max_sequence_length = gr.Number(label="Max Sequence Length", value=512)
                        
                        hf_api_key_status = gr.Markdown(value=f"API Key Status: {'Saved' if load_hf_api_key() else 'Not saved'}")
                        
                        with gr.Accordion("API Key Settings", open=False):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    hf_api_key_input = gr.Textbox(
                                        label="Hugging Face API Key", 
                                        type="password", 
                                        value=load_hf_api_key()
                                    )
                                with gr.Column(scale=1):
                                    hf_api_key_options = gr.Dropdown(
                                        choices=["Save API Key", "Clear API Key"],
                                        label="API Key Options",
                                        type="index"
                                    )
                            
                            gr.Markdown("Don't have an API key? [Create one here](https://huggingface.co/settings/tokens)")
                        
                    with gr.Column(scale=1):
                        download_button = gr.Button("Download and Add Model")
                        download_output = gr.Textbox(label="Download Status", lines=3)

                def handle_hf_api_key(choice, api_key):
                    if choice == 0:  # Save API Key
                        save_hf_api_key(api_key)
                        return "API Key Status: Saved successfully"
                    elif choice == 1:  # Clear API Key
                        save_hf_api_key("")
                        return "API Key Status: Cleared"
 
                hf_api_key_options.change(
                    fn=handle_hf_api_key,
                    inputs=[hf_api_key_options, hf_api_key_input],
                    outputs=[hf_api_key_status]
                )

                gr.Markdown("## Quantize Model")
                with gr.Row():
                    with gr.Column(scale=2):
                        model_quant = gr.Dropdown(
                            choices=[m for m in get_updated_models() if not m.endswith("-4-bit") and not m.endswith("-8-bit")],
                            label="Model to Quantize",
                            value="dev"
                        )
                        quantize_level = gr.Radio(choices=["4", "8"], label="Quantize Level", value="8")
                    with gr.Column(scale=1):
                        save_button = gr.Button("Save Quantized Model")
                        save_output = gr.Textbox(label="Quantization Output", lines=3)

                download_button.click(
                    fn=download_and_save_model,
                    inputs=[hf_model_name, alias, num_train_steps, max_sequence_length, hf_api_key_input],
                    outputs=[model_simple, model, model_cn, model_quant, download_output]
                )

                save_button.click(
                    fn=save_quantized_model_gradio,
                    inputs=[model_quant, quantize_level],
                    outputs=[model_simple, model, model_cn, model_quant, save_output]
                )

                save_hf_api_key_button = gr.Button("Save Hugging Face API Key")
                save_hf_api_key_button.click(
                    fn=lambda key: save_hf_api_key(key) or "API Key saved successfully",
                    inputs=[hf_api_key_input],
                    outputs=[gr.Textbox(label="API Key Status")]
                )

    return demo

def main():
    demo = create_ui()
    demo.launch()

if __name__ == "__main__":
    main()
