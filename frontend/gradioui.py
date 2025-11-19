# frontend/gradioui.py

import os
import json
import random
import gradio as gr
from pathlib import Path
from PIL import Image
from mflux.config.config import Config
from typing import List, Tuple

# Import components
from frontend.components.easy_mflux import create_easy_mflux_tab
from frontend.components.advanced_generate import create_advanced_generate_tab
from frontend.components.controlnet import create_controlnet_tab
from frontend.components.image_to_image import create_image_to_image_tab
from frontend.components.model_lora_management import create_model_lora_management_tab
from frontend.components.dreambooth_fine_tuning import create_dreambooth_fine_tuning_tab
from frontend.components.in_context_lora import create_in_context_lora_tab
# New feature components
from frontend.components.fill import create_fill_tab
from frontend.components.depth import create_depth_tab
from frontend.components.redux import create_redux_tab
from frontend.components.upscale import create_upscale_tab
from frontend.components.auto_seeds import create_auto_seeds_tab
from frontend.components.battery_monitor import create_battery_tab
from frontend.components.dynamic_prompts import create_dynamic_prompts_tab
from frontend.components.config_manager import create_config_tab
from frontend.components.catvton import create_catvton_tab
from frontend.components.ic_edit import create_ic_edit_tab
from frontend.components.concept_attention import create_concept_attention_tab
from frontend.components.kontext import create_kontext_tab
from frontend.components.canvas import create_canvas_tab
from frontend.components.qwen_image import create_qwen_image_tab
from frontend.components.qwen_edit import create_qwen_edit_tab

# Backend imports
from backend.model_manager import (
    get_updated_models,
    get_custom_model_config,
    download_and_save_model,
    save_quantized_model_gradio,
    login_huggingface
)
from backend.lora_manager import (
    get_lora_choices,
    process_lora_files,
    update_lora_scales,
    download_lora_model,
    get_updated_lora_files,
    refresh_lora_choices,
    MAX_LORAS
)
from backend.flux_manager import (
    get_random_seed,
    simple_generate_image,
    generate_image_gradio,
    generate_image_controlnet_gradio,
    generate_image_i2i_gradio,
    generate_image_in_context_lora_gradio
)
from backend.mlx_vlm_manager import (
    get_available_mlx_vlm_models,
    generate_caption_with_mlx_vlm
)
from backend.captions import (
    show_uploaded_images,
    fill_captions
)
from backend.ollama_manager import (
    create_ollama_settings,
    enhance_prompt as enhance_prompt_ollama,
    get_available_ollama_models
)
from backend.prompts_manager import (
    read_system_prompt,
    save_ollama_settings,
    load_prompt_file,
    save_prompt_file,
)
from backend.training_manager import (
    run_training,
    prepare_training_config,
    run_dreambooth_from_ui_no_explicit_quantize
)
from backend.huggingface_manager import (
    get_available_models,
    download_and_save_model,
    login_huggingface,
    load_api_key,
    save_api_key,
    load_hf_api_key,
    save_hf_api_key,
    download_lora_model_huggingface
)
from backend.post_processing import (
    update_dimensions_on_image_change,
    update_dimensions_on_scale_change,
    update_height_with_aspect_ratio,
    update_width_with_aspect_ratio,
    scale_dimensions,
    update_guidance_visibility
)

def create_ui():
    """
    Create the Gradio UI interface following the layout/theming guidance from
    gradiodocs/docs-blocks/blocks.md and gradiodocs/guides-themes/gradio_themes.md.
    """
    theme = gr.themes.Soft()
    with gr.Blocks(
        theme=theme,
        title="MFLUX WebUI",
        fill_width=True,
        analytics_enabled=False,
        css="""
        .refresh-button {
            background-color: white !important;
            border: 1px solid #ccc !important;
            color: black !important;
            padding: 0px 8px !important;
            height: 38px !important;
            margin-left: -10px !important;
        }
        .refresh-button:hover {
            background-color: #f0f0f0 !important;
        }
        .markdown {
            background: none !important;
            border: none !important;
            padding: 0 !important;
        }
        .group {
            background-color: white !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }
        .white-bg {
            background-color: white !important;
        }
    """,
    ) as demo:
        with gr.Tabs():
            with gr.TabItem("MFLUX Easy", id=0):
                easy_mflux_components = create_easy_mflux_tab()
                lora_files_simple = easy_mflux_components['lora_files']
                model_simple = easy_mflux_components['model']

            # Qwen tabs directly after MFLUX Easy
            qwen_image_components = create_qwen_image_tab()
            qwen_edit_components = create_qwen_edit_tab()

            with gr.TabItem("🎨 Canvas"):
                canvas_components = create_canvas_tab()
                
            with gr.TabItem("Advanced Generate"):
                advanced_generate_components = create_advanced_generate_tab()
                lora_files = advanced_generate_components['lora_files']
                model = advanced_generate_components['model']

            with gr.TabItem("ControlNet"):
                controlnet_components = create_controlnet_tab()
                lora_files_cn = controlnet_components['lora_files']
                model_cn = controlnet_components['model']

            with gr.TabItem("Image-to-Image"):
                image_to_image_components = create_image_to_image_tab()
                lora_files_i2i = image_to_image_components['lora_files']
                model_i2i = image_to_image_components['model']
                
            with gr.TabItem("In-Context LoRA"):
                in_context_lora_components = create_in_context_lora_tab()
                lora_files_icl = in_context_lora_components['lora_files']
                model_icl = in_context_lora_components['model']

            with gr.TabItem("Dreambooth Fine-Tuning"):
                dreambooth_components = create_dreambooth_fine_tuning_tab()

            # --- New feature tabs ---
            with gr.TabItem("Fill"):
                fill_components = create_fill_tab()

            with gr.TabItem("Depth"):
                depth_components = create_depth_tab()

            with gr.TabItem("Redux"):
                redux_components = create_redux_tab()

            with gr.TabItem("Upscale"):
                upscale_components = create_upscale_tab()

            with gr.TabItem("Concept Attention"):
                concept_attention_components = create_concept_attention_tab()

            with gr.TabItem("CatVTON"):
                catvton_components = create_catvton_tab()

            with gr.TabItem("IC-Edit"):
                ic_edit_components = create_ic_edit_tab()

            with gr.TabItem("Kontext"):
                kontext_components = create_kontext_tab()

            # --- Utility and Management tabs ---
            with gr.TabItem("Auto Seeds"):
                auto_seeds_components = create_auto_seeds_tab()

            with gr.TabItem("Dynamic Prompts"):
                dynamic_prompts_components = create_dynamic_prompts_tab()

            with gr.TabItem("Battery Monitor"):
                battery_components = create_battery_tab()

            with gr.TabItem("Configuration"):
                config_components = create_config_tab()

            with gr.TabItem("Model & LoRA Management"):
                model_lora_management_components = create_model_lora_management_tab(
                    model_simple=model_simple,
                    model=model,
                    model_cn=model_cn,
                    model_i2i=model_i2i,
                    lora_files_simple=lora_files_simple,
                    lora_files=lora_files,
                    lora_files_cn=lora_files_cn,
                    lora_files_i2i=lora_files_i2i,
                    lora_files_icl=lora_files_icl,
                    model_icl=model_icl
                )

        return demo

if __name__ == "__main__":
    demo = create_ui()
    # Shared queue configuration per gradiodocs/guides-queuing/queuing.md
    demo.queue(default_concurrency_limit=4, status_tracker=True).launch(show_error=True)
