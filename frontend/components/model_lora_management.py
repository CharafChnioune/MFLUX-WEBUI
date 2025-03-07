import gradio as gr
from backend.model_manager import (
    get_updated_models,
    download_and_save_model,
    save_quantized_model_gradio
)
from backend.huggingface_manager import (
    download_lora_model_huggingface,
    load_api_key,
    save_api_key,
    load_hf_api_key,
    save_hf_api_key
)
from backend.lora_manager import (
    get_lora_choices,
    update_lora_scales,
    refresh_lora_choices,
    MAX_LORAS,
    process_lora_files
)
from backend.civitai_manager import download_lora_model

def create_model_lora_management_tab(model_simple, model, model_cn, model_i2i, model_icl, lora_files_simple, lora_files, lora_files_cn, lora_files_i2i, lora_files_icl):
    """Create the Model & LoRA Management tab interface"""
    
    # Centrale API Key Settings
    with gr.Accordion("🔑 API Key Settings", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                civitai_api_key = gr.Textbox(
                    label="CivitAI API Key", 
                    type="password", 
                    value=load_api_key("civitai")
                )
                civitai_api_key_status = gr.Markdown(
                    value=f"CivitAI API Key Status: {'Saved' if load_api_key('civitai') else 'Not saved'}"
                )
                
                hf_api_key = gr.Textbox(
                    label="HuggingFace API Key", 
                    type="password", 
                    value=load_hf_api_key()
                )
                hf_api_key_status = gr.Markdown(
                    value=f"HuggingFace API Key Status: {'Saved' if load_hf_api_key() else 'Not saved'}"
                )
            with gr.Column(scale=1):
                save_api_keys_button = gr.Button("Save API Keys")
                clear_api_keys_button = gr.Button("Clear API Keys")
        gr.Markdown("Don't have an API key? [CivitAI](https://civitai.com/user/account), [HuggingFace](https://huggingface.co/settings/tokens)")

    def save_api_keys(api_key_civitai, api_key_hf):
        save_api_key(api_key_civitai, "civitai")
        save_hf_api_key(api_key_hf)
        return (
            f"CivitAI API Key Status: {'Saved successfully' if api_key_civitai else 'Not saved'}",
            f"HuggingFace API Key Status: {'Saved successfully' if api_key_hf else 'Not saved'}"
        )

    def clear_api_keys():
        save_api_key("", "civitai")
        save_api_key("", "huggingface")
        return "CivitAI API Key Status: Cleared", "HuggingFace API Key Status: Cleared"

    save_api_keys_button.click(
        fn=save_api_keys,
        inputs=[civitai_api_key, hf_api_key],
        outputs=[civitai_api_key_status, hf_api_key_status]
    )

    clear_api_keys_button.click(
        fn=clear_api_keys,
        inputs=[],
        outputs=[civitai_api_key_status, hf_api_key_status]
    )

    gr.Markdown("### Download LoRA")
    lora_source = gr.Radio(
        choices=["CivitAI", "HuggingFace"],
        label="LoRA Source",
        value="CivitAI"
    )
    lora_input = gr.Textbox(label="LoRA Model Page URL (CivitAI) or Model Name (HuggingFace)")
    download_lora_button = gr.Button("Download LoRA", variant='primary')
    lora_download_status = gr.Textbox(label="Download Status", lines=0.5)

    def download_lora(model_url_or_name, api_key_civitai, api_key_hf, lora_source):
        try:
            if lora_source == "CivitAI":
                lora_files_simple, lora_files, lora_files_cn, lora_files_i2i, lora_files_icl, status = download_lora_model(model_url_or_name, api_key_civitai)
            elif lora_source == "HuggingFace":
                status = download_lora_model_huggingface(model_url_or_name, api_key_hf)
                if "Error" not in status:
                    lora_files_simple = get_lora_choices()
                    lora_files = lora_files_simple
                    lora_files_cn = lora_files_simple
                    lora_files_i2i = lora_files_simple
                    lora_files_icl = lora_files_simple
                else:
                    lora_files_simple = None
                    lora_files = None
                    lora_files_cn = None
                    lora_files_i2i = None
                    lora_files_icl = None
            else:
                return None, None, None, None, None, "Error: Invalid LoRA source selected"

            return (
                gr.update(choices=lora_files_simple) if lora_files_simple else gr.update(),
                gr.update(choices=lora_files) if lora_files else gr.update(),
                gr.update(choices=lora_files_cn) if lora_files_cn else gr.update(),
                gr.update(choices=lora_files_i2i) if lora_files_i2i else gr.update(),
                gr.update(choices=lora_files_icl) if lora_files_icl else gr.update(),
                status
            )
        except Exception as e:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Error: {str(e)}"

    download_lora_button.click(
        fn=download_lora,
        inputs=[lora_input, civitai_api_key, hf_api_key, lora_source],
        outputs=[lora_files_simple, lora_files, lora_files_cn, lora_files_i2i, lora_files_icl, lora_download_status]
    )

    gr.Markdown("## Download and Add Model")
    with gr.Row():
        with gr.Column(scale=2):
            hf_model_name = gr.Textbox(label="Hugging Face Model Name")
            alias = gr.Textbox(label="Model Alias")
            base_arch = gr.Radio(
                choices=["schnell", "dev"],
                label="Base Architecture (for Third-Party models)",
                value="schnell"
            )
            num_train_steps = gr.Number(label="Number of Training Steps", value=1000)
            max_sequence_length = gr.Number(label="Max Sequence Length", value=512)
        with gr.Column(scale=1):
            download_button = gr.Button("Download and Add Model", variant='primary')
            download_output = gr.Textbox(label="Download Status", lines=3)

    gr.Markdown("## Quantize Model")
    with gr.Row():
        with gr.Column(scale=2):
            model_quant = gr.Dropdown(
                choices=[m for m in get_updated_models() 
                        if not any(m.endswith(f"-{bits}-bit") for bits in ["3", "4", "6", "8"])],
                label="Model to Quantize",
                value="dev"
            )
            quantize_level = gr.Radio(
                choices=["3", "4", "6", "8"], 
                label="Quantize Level", 
                value="8"
            )
        with gr.Column(scale=1):
            save_button = gr.Button("Save Quantized Model", variant='primary')
            save_output = gr.Textbox(label="Quantization Output", lines=3)

    gr.Markdown("## Download Ollama Model")
    with gr.Row():
        with gr.Column(scale=2):
            ollama_model_name = gr.Textbox(
                label="Ollama Model Name(Only the model name, not the full command for example llama3.2-vision)",
                placeholder="e.g. llama2, mistral, codellama"
            )
        with gr.Column(scale=1):
            download_ollama_button = gr.Button("Download Ollama Model", variant='primary')
            ollama_download_status = gr.Textbox(label="Download Status", lines=3)

    gr.Markdown("## Download MLX Model")
    with gr.Row():
        with gr.Column(scale=2):
            mlx_model_name = gr.Textbox(
                label="MLX Model Name from Hugging Face",
                placeholder="e.g. mlx-community/Mistral-7B-v0.1-mlx"
            )
        with gr.Column(scale=1):
            download_mlx_button = gr.Button("Download MLX Model", variant='primary')
            mlx_download_status = gr.Textbox(label="Download Status", lines=3)

    def download_ollama_model(model_name):
        from backend.ollama_manager import ensure_llama_model
        try:
            if ensure_llama_model(model_name):
                return f"Successfully downloaded Ollama model: {model_name}"
            else:
                return f"Error downloading Ollama model: {model_name}"
        except Exception as e:
            return f"Error: {str(e)}"

    def download_mlx_model(model_name, api_key):
        from backend.mlx_vlm_manager import load_mlx_model
        try:
            model, processor, config = load_mlx_model(model_name)
            if model is not None:
                return f"Successfully downloaded MLX model: {model_name}"
            else:
                return f"Error downloading MLX model: {model_name}"
        except Exception as e:
            return f"Error: {str(e)}"

    download_ollama_button.click(
        fn=download_ollama_model,
        inputs=[ollama_model_name],
        outputs=[ollama_download_status]
    )

    download_mlx_button.click(
        fn=download_mlx_model,
        inputs=[mlx_model_name, hf_api_key],
        outputs=[mlx_download_status]
    )

    download_button.click(
        fn=download_and_save_model,
        inputs=[hf_model_name, alias, num_train_steps, max_sequence_length, hf_api_key, base_arch],
        outputs=[model_simple, model, model_cn, model_i2i, model_icl, download_output]
    )

    save_button.click(
        fn=save_quantized_model_gradio,
        inputs=[model_quant, quantize_level],
        outputs=[model_simple, model, model_cn, model_i2i, model_icl, model_quant, save_output]
    )

    return {
        'model_quant': model_quant,
        'lora_download_status': lora_download_status,
        'download_output': download_output,
        'save_output': save_output,
        'ollama_download_status': ollama_download_status,
        'mlx_download_status': mlx_download_status
    } 