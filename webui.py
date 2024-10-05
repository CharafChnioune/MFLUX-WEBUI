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

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config, ConfigControlnet
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet

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

def download_and_save_model(hf_model_name, alias, num_train_steps, max_sequence_length):
    try:
        model_path = hf_hub_download(repo_id=hf_model_name, filename="pytorch_model.bin")
        
        local_path = os.path.join(MODELS_DIR, f"{alias}.bin")
        os.rename(model_path, local_path)
        
        new_config = CustomModelConfig(hf_model_name, alias, num_train_steps, max_sequence_length)
        get_custom_model_config.__globals__['models'][alias] = new_config
        
        return f"Model {hf_model_name} successfully downloaded and saved as {alias}"
    except Exception as e:
        return f"Error downloading model: {str(e)}"

@lru_cache(maxsize=None)
def get_or_create_flux(model, quantize, path, lora_paths_tuple, lora_scales_tuple, is_controlnet=False):
    global flux_cache
    
    # Converteer tuples terug naar lijsten
    lora_paths = list(lora_paths_tuple) if lora_paths_tuple else None
    lora_scales = list(lora_scales_tuple) if lora_scales_tuple else None
    
    key = (model, quantize, path, lora_paths_tuple, lora_scales_tuple, is_controlnet)
    if key in flux_cache:
        return flux_cache[key]
    
    FluxClass = Flux1Controlnet if is_controlnet else Flux1
    
    # Scheid de quantisatie-informatie van de modelnaam
    base_model = model.replace("-8-bit", "").replace("-4-bit", "")
    
    try:
        custom_config = get_custom_model_config(base_model)
    except ValueError:
        custom_config = CustomModelConfig(base_model, base_model, 1000, 512)
        print(f"Waarschuwing: Onbekend model '{base_model}' gebruikt. Standaard configuratie toegepast.")
    
    # Bepaal de quantisatie op basis van de originele modelnaam
    if "-8-bit" in model:
        quantize = 8
    elif "-4-bit" in model:
        quantize = 4
    
    flux = FluxClass(
        model_config=custom_config,
        quantize=quantize,
        local_path=path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    
    flux_cache[key] = flux
    return flux

def get_available_lora_files():
    return [(str(f), f.stem) for f in Path(LORA_DIR).rglob("*.safetensors")]

def get_available_models():
    standard_models = ["dev", "schnell", "dev-8-bit", "dev-4-bit", "schnell-8-bit", "schnell-4-bit"]
    custom_models = [f.stem for f in Path(MODELS_DIR).glob('*') if f.is_file() and f.name != '.gitkeep']
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

    lora_paths = lora_files if lora_files else None

    seed = None if seed == "" else int(seed)
    steps = None if steps == "" else int(steps)

    flux = get_or_create_flux(model, None, None, tuple(lora_paths) if lora_paths else None, None)

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

    clear_flux_cache()

    print_memory_usage("After clearing cache")

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

    lora_dict = dict(get_available_lora_files())
    
    lora_files = lora_files or []
    
    lora_paths = [lora_dict[lora] for lora in lora_files if lora in lora_dict]

    seed = None if seed == "" else int(seed)
    steps = None if steps == "" else int(steps)

    flux = get_or_create_flux(model, None, None, tuple(lora_paths) if lora_paths else None, None, is_controlnet=True)

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
        
        clear_flux_cache()
        
        print_memory_usage("After clearing cache")

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        return generated_image.image, f"Image generated successfully! Saved as {output_filename}", prompt
    except Exception as e:
        return None, f"Error generating image: {str(e)}", prompt

def save_quantized_model_gradio(model, quantize, save_path):
    quantize = int(quantize)
    custom_config = get_custom_model_config(model)
    flux = Flux1(
        model_config=custom_config,
        quantize=quantize,
    )

    flux.save_model(save_path)

    return f"Model saved at {save_path}"

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
        # Parse the selected image format
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
                            choices=get_available_models(),
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
                            choices=[file[1] for file in get_available_lora_files()], 
                            label="Select LoRA Files", 
                            multiselect=True
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
                    inputs=[
                        prompt_simple,
                        model_simple,
                        image_format,
                        lora_files_simple,
                        ollama_components_simple[0],
                        ollama_components_simple[1],
                    ],
                    outputs=[output_image_simple, output_filename_simple, prompt_simple],
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
                            choices=get_available_models(),
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
                            choices=[file[1] for file in get_available_lora_files()],
                            label="Select LoRA Files",
                            multiselect=True
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
                    inputs=[
                        prompt,
                        model,
                        seed,
                        height,
                        width,
                        steps,
                        guidance,
                        lora_files,
                        metadata,
                        ollama_components_adv[0],
                        ollama_components_adv[1],
                    ],
                    outputs=[output_image, output_filename, prompt],
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
                            choices=get_available_models(),
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
                            choices=[file[1] for file in get_available_lora_files()],
                            label="Select LoRA Files",
                            multiselect=True
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
                    inputs=[
                        prompt_cn,
                        control_image,
                        model_cn,
                        seed_cn,
                        height_cn,
                        width_cn,
                        steps_cn,
                        guidance_cn,
                        controlnet_strength,
                        lora_files_cn,
                        metadata_cn,
                        save_canny,
                        ollama_components_cn[0],
                        ollama_components_cn[1],
                    ],
                    outputs=[output_image_cn, output_message_cn, prompt_cn],
                )

                gr.Markdown("""
                ⚠️ Note: Controlnet requires [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny), which was trained for the `dev` model. 
                It can work well with `schnell`, but performance is not guaranteed.

                ⚠️ Note: The output can be highly sensitive to the controlnet strength and is very much dependent on the reference image. 
                Too high settings will corrupt the image. A recommended starting point is a value like 0.4. Experiment with different strengths to find the best result.
                """)

            with gr.TabItem("Models"):
                with gr.Row():
                    with gr.Column():
                        hf_model_name = gr.Textbox(label="Hugging Face Model Name")
                        alias = gr.Textbox(label="Model Alias")
                        num_train_steps = gr.Number(label="Number of Training Steps", value=1000)
                        max_sequence_length = gr.Number(label="Max Sequence Length", value=512)
                        download_button = gr.Button("Download and Add Model")
                    with gr.Column():
                        download_output = gr.Textbox(label="Download Status")

                download_button.click(
                    fn=download_and_save_model,
                    inputs=[hf_model_name, alias, num_train_steps, max_sequence_length],
                    outputs=download_output
                )

                with gr.Row():
                    with gr.Column():
                        model_quant = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        quantize_level = gr.Radio(choices=["4", "8"], label="Quantize Level", value="8")
                        save_path = gr.Textbox(
                            label="Save Path", placeholder="Enter the path to save the quantized model"
                        )
                        save_button = gr.Button("Save Quantized Model")
                    with gr.Column():
                        save_output = gr.Textbox(label="Quantization Output")
                save_button.click(
                    fn=save_quantized_model_gradio,
                    inputs=[model_quant, quantize_level, save_path],
                    outputs=save_output
                )

        gr.Markdown("**Note:** Lora's are only available for the `dev` and 'schnell' models not for quantized models.")

    return demo

def main():
    demo = create_ui()
    demo.launch()

if __name__ == "__main__":
    main()