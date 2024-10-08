import gradio as gr
import time
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
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
from mflux.config.model_config import ModelConfig
from mflux.config.config import Config, ConfigControlnet
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder, snapshot_download
from contextlib import contextmanager
from PIL import Image

LORA_DIR = os.path.join(os.path.dirname(__file__), "lora")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface/hub"))
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

@lru_cache(maxsize=None)
def get_or_create_flux(model, quantize, path, lora_paths_tuple, lora_scales_tuple, is_controlnet=False):
    global flux_cache
    
    lora_paths = list(lora_paths_tuple) if lora_paths_tuple else None
    lora_scales = list(lora_scales_tuple) if lora_scales_tuple else None
    
    key = (model, quantize, path, lora_paths_tuple, lora_scales_tuple, is_controlnet)
    if key in flux_cache:
        return flux_cache[key]
    
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
    
    flux_cache[key] = flux
    return flux

def get_available_lora_files():
    return [(str(f), f.stem) for f in Path(LORA_DIR).rglob("*.safetensors")]

def get_available_models():
    standard_models = ["dev", "schnell", "dev-8-bit", "dev-4-bit", "schnell-8-bit", "schnell-4-bit"]
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

@contextmanager
def mflux_context(model, quantize=None, path=None, lora_paths=None, lora_scales=None, is_controlnet=False):
    model_path = os.path.join(MODELS_DIR, model)
    command = ["mflux-generate", "--model", model, "--path", model_path]
    
    if lora_paths:
        command.extend(["--lora-paths"] + lora_paths)
    if lora_scales:
        command.extend(["--lora-scales"] + [str(scale) for scale in lora_scales])
    if is_controlnet:
        command[0] = "mflux-generate-controlnet"
    
    try:
        yield command
    finally:
        clear_flux_cache()
        force_mlx_cleanup()

def ensure_model_downloaded(model_alias):
    model_config = get_custom_model_config(model_alias)
    model_path = os.path.join(MODELS_DIR, model_alias)
    
    if not os.path.exists(model_path):
        print(f"Model {model_alias} not found. Downloading...")
        try:
            snapshot_download(
                repo_id=model_config.model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            print(f"Model {model_alias} successfully downloaded to {model_path}")
        except Exception as e:
            print(f"Error downloading model {model_alias}: {str(e)}")
            return None
    else:
        print(f"Model {model_alias} found at {model_path}")
    
    return model_path

def generate_image_gradio(prompt, model, seed, height, width, steps, guidance, lora_files, metadata, ollama_model, system_prompt):
    print(f"\n--- Generating image (Advanced) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Dimensions: {height}x{width}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"LoRA files: {lora_files}")

    base_model = "dev" if "dev" in model else "schnell"
    model_path = ensure_model_downloaded(base_model)
    if not model_path:
        return None, None, "Failed to download or locate the model. Please check your internet connection and try again."

    timestamp = int(time.time())
    output_filename = f"generated_{timestamp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    command = [
        "mflux-generate",
        "--model", base_model,
        "--path", model_path,
        "--prompt", prompt,
        "--height", str(height),
        "--width", str(width),
        "--output", output_path
    ]

    if "4-bit" in model:
        command.extend(["-q", "4"])
    elif "8-bit" in model:
        command.extend(["-q", "8"])

    if seed:
        command.extend(["--seed", str(seed)])
    if steps:
        command.extend(["--steps", str(steps)])
    if guidance and "dev" in model:
        command.extend(["--guidance", str(guidance)])
    if metadata:
        command.append("--metadata")

    lora_paths = process_lora_files(lora_files)
    if lora_paths:
        command.extend(["--lora-paths"] + lora_paths)
        command.extend(["--lora-scales"] + ["1.0"] * len(lora_paths))

    try:
        print(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        if os.path.exists(output_path):
            return Image.open(output_path), output_filename, prompt
        else:
            print(f"Error: Output file not found: {output_path}")
            return None, None, prompt
    except subprocess.CalledProcessError as e:
        print(f"Error running mflux-generate: {e}")
        print(f"Error output: {e.stderr}")
        return None, None, prompt
    finally:
        clear_flux_cache()
        force_mlx_cleanup()

def generate_image_controlnet_gradio(
    prompt, control_image, model, seed, height, width, steps, guidance, controlnet_strength, lora_files, metadata, save_canny, ollama_model, system_prompt
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

    try:
        if control_image is None:
            raise ValueError("Control image is required for ControlNet generation")

        timestamp = int(time.time())
        control_image_path = os.path.join(OUTPUT_DIR, f"control_image_{timestamp}.png")
        control_image.save(control_image_path)

        output_filename = f"generated_controlnet_{timestamp}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        base_model = "dev" if "dev" in model else "schnell"
        model_path = ensure_model_downloaded(base_model)
        if not model_path:
            raise ValueError("Failed to download or locate the model")

        command = [
            "mflux-generate-controlnet",
            "--model", base_model,
            "--path", model_path,
            "--prompt", prompt,
            "--height", str(height),
            "--width", str(width),
            "--output", output_path,
            "--controlnet-image-path", control_image_path,
            "--controlnet-strength", str(controlnet_strength)
        ]

        if "4-bit" in model:
            command.extend(["-q", "4"])
        elif "8-bit" in model:
            command.extend(["-q", "8"])

        if seed:
            command.extend(["--seed", str(seed)])
        if steps:
            command.extend(["--steps", str(steps)])
        if guidance and "dev" in model:
            command.extend(["--guidance", str(guidance)])
        if metadata:
            command.append("--metadata")
        if save_canny:
            command.append("--controlnet-save-canny")

        lora_paths = process_lora_files(lora_files)
        if lora_paths:
            command.extend(["--lora-paths"] + lora_paths)
            command.extend(["--lora-scales"] + ["1.0"] * len(lora_paths))

        print(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        if os.path.exists(output_path):
            generated_image = Image.open(output_path)
            canny_image_path = os.path.join(OUTPUT_DIR, f"canny_{timestamp}.png")
            canny_image = Image.open(canny_image_path) if os.path.exists(canny_image_path) else None
            return generated_image, output_filename, prompt, canny_image
        else:
            raise FileNotFoundError(f"Output file not found: {output_path}")

    except Exception as e:
        print(f"Error in ControlNet generation: {str(e)}")
        return None, str(e), prompt, None

    finally:
        clear_flux_cache()
        force_mlx_cleanup()
        print_memory_usage("After generation")
        print(f"Total generation time: {time.time() - start_time:.2f} seconds")

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

    width, height = map(int, image_format.split('(')[1].split(')')[0].split('x'))
    
    timestamp = int(time.time())
    output_filename = f"generated_simple_{timestamp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        base_model = "schnell"
        model_path = os.path.join(MODELS_DIR, base_model)
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        command = [
            "mflux-generate",
            "--model", base_model,
            "--path", model_path,
            "--prompt", prompt,
            "--height", str(height),
            "--width", str(width),
            "--output", output_path,
            "--steps", "4"
        ]

        if "4-bit" in model:
            command.extend(["-q", "4"])
        elif "8-bit" in model:
            command.extend(["-q", "8"])

        lora_paths = process_lora_files(lora_files)
        if lora_paths:
            command.extend(["--lora-paths"] + lora_paths)
            command.extend(["--lora-scales"] + ["1.0"] * len(lora_paths))

        print(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)

        if os.path.exists(output_path):
            return Image.open(output_path), output_filename, prompt
        else:
            print(f"Error: Output file not found: {output_path}")
            return None, None, prompt
    except subprocess.CalledProcessError as e:
        print(f"Error running mflux-generate: {e}")
        print(f"Error output: {e.stderr}")
        return None, None, prompt
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None, None, prompt
    finally:
        clear_flux_cache()
        force_mlx_cleanup()

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
        if hasattr(mx.metal, 'reset_peak_memory'):
            mx.metal.reset_peak_memory()
        
        mx.eval(mx.zeros(1))
        
        if hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
        
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
    models = []
    for root, dirs, files in os.walk(MODELS_DIR):
        if os.path.basename(root) == 'hub':
            continue
        for file in files:
            if file.endswith('.safetensors') or file.endswith('.bin'):
                model_name = os.path.relpath(root, MODELS_DIR)
                if model_name not in models:
                    models.append(model_name)
    return sorted(models)

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
                            value="schnell"
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
                        canny_image_cn = gr.Image(label="Canny Edge Detection", visible=False)
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
                    outputs=[output_image_cn, output_message_cn, prompt_cn, canny_image_cn]
                )

                gr.Markdown("""
                ⚠️ Note: Controlnet requires [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny), which was trained for the `dev` model. 
                It can work well with `schnell`, but performance is not guaranteed.

                ⚠️ Note: The output can be highly sensitive to the controlnet strength and is very much dependent on the reference image. 
                Too high settings will corrupt the image. A recommended starting point is a value like 0.4. Experiment with different strengths to find the best result.
                """)

            with gr.TabItem("Models"):

                gr.Markdown("### Download LoRA")
                lora_url = gr.Textbox(label="LoRA Model Page URL (CivitAI)")
                
                api_key_status = gr.Markdown(value=f"API Key Status: {'Saved' if load_api_key() else 'Not saved'}")
                
                with gr.Accordion("API Key Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            api_key_input = gr.Textbox(
                                label="CivitAI API Key", 
                                type="password", 
                                value=load_api_key()
                            )
                        with gr.Column(scale=1):
                            api_key_options = gr.Dropdown(
                                choices=["Save API Key", "Clear API Key"],
                                label="API Key Options",
                                type="index"
                            )
                    
                    gr.Markdown("Don't have an API key? [Create one here](https://civitai.com/user/account)")
                
                download_lora_button = gr.Button("Download LoRA")
                lora_download_status = gr.Textbox(label="Download Status", lines=0.5)

                def handle_api_key(choice, api_key):
                    if choice == 0:
                        save_api_key(api_key)
                        return "API Key Status: Saved successfully"
                    elif choice == 1:
                        save_api_key("")
                        return "API Key Status: Cleared"
 
                api_key_options.change(
                    fn=handle_api_key,
                    inputs=[api_key_options, api_key_input],
                    outputs=[api_key_status]
                )
 
                download_lora_button.click(
                    fn=download_lora_model,
                    inputs=[lora_url, api_key_input],
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
                    if choice == 0:
                        save_hf_api_key(api_key)
                        return "API Key Status: Saved successfully"
                    elif choice == 1:
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