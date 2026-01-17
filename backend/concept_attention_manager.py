import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

from backend.mflux_compat import Config, ModelConfig
from mflux.models.flux.variants.concept_attention.flux_concept import Flux1Concept
from mflux.models.flux.variants.concept_attention.flux_concept_from_image import Flux1ConceptFromImage

from backend.lora_manager import process_lora_files
from backend.model_manager import get_custom_model_config, resolve_local_path

# Refer to:
# - gradiodocs/docs-blocks/blocks.md for layout consistency
# - gradiodocs/guides-streaming-outputs/streaming_outputs.md for progress feedback

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_model(model_name: str, base_model: Optional[str]):
    try:
        custom = get_custom_model_config(model_name)
        return custom.model_name, (base_model or custom.base_arch), resolve_local_path(model_name)
    except ValueError:
        return model_name, base_model, resolve_local_path(model_name)


def _prepare_loras(lora_files, lora_scales):
    if not lora_files:
        return None, None
    paths = process_lora_files(lora_files)
    scales = process_lora_files(lora_files, lora_scales) if paths else None
    return paths, scales


def generate_text_concept_heatmap(
    prompt: str,
    concept: str,
    model_name: str,
    base_model: Optional[str],
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    quantize_bits: Optional[int],
    heatmap_layers: str,
    heatmap_timesteps: str,
    lora_files,
    lora_scales,
    metadata: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[List[Image.Image], List[Image.Image], str]:
    if not prompt or not concept:
        return [], [], "Prompt and concept are required"

    try:
        progress(0.0, desc="Loading Concept model")
        quant_value = None
        if quantize_bits not in (None, "", "None"):
            quant_value = int(quantize_bits)
        model_identifier, resolved_base, local_dir = _resolve_model(model_name, base_model)
        model_cfg = ModelConfig.from_name(model_name=model_identifier, base_model=resolved_base)
        lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

        flux = Flux1Concept(
            model_config=model_cfg,
            quantize=quant_value,
            local_path=local_dir,
            lora_paths=lora_paths,
            lora_scales=lora_scale_values,
        )

        config = Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        layers = [int(idx.strip()) for idx in heatmap_layers.split(",") if idx.strip().isdigit()] or None
        timesteps = [int(idx.strip()) for idx in heatmap_timesteps.split(",") if idx.strip().isdigit()] or None

        progress(0.4, desc="Generating attention heatmap")
        result = flux.generate_image(
            seed=seed,
            prompt=prompt,
            concept=concept,
            heatmap_layer_indices=layers,
            heatmap_timesteps=timesteps,
            config=config,
        )
        filename = f"concept_{int(time.time())}_{seed}.png"
        output_path = OUTPUT_DIR / filename
        result.save_with_heatmap(path=output_path, export_json_metadata=metadata)

        progress(1.0, desc="Done")
        status = f"Saved {filename} with heatmap layers {layers or 'default'}."
        return [result.image], [result.heatmap], status

    except Exception as exc:
        error_message = f"Concept analysis failed: {exc}"
        print(error_message)
        return [], [], error_message


def generate_image_concept_heatmap(
    prompt: str,
    concept: str,
    reference_image: str,
    model_name: str,
    base_model: Optional[str],
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    quantize_bits: Optional[int],
    heatmap_layers: str,
    heatmap_timesteps: str,
    lora_files,
    lora_scales,
    metadata: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[List[Image.Image], List[Image.Image], str]:
    if not reference_image:
        return [], [], "Reference image is required for image-guided concept analysis."

    try:
        progress(0.0, desc="Loading Concept+Image model")
        quant_value = None
        if quantize_bits not in (None, "", "None"):
            quant_value = int(quantize_bits)
        model_identifier, resolved_base, local_dir = _resolve_model(model_name, base_model)
        model_cfg = ModelConfig.from_name(model_name=model_identifier, base_model=resolved_base)
        lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

        flux = Flux1ConceptFromImage(
            model_config=model_cfg,
            quantize=quant_value,
            local_path=local_dir,
            lora_paths=lora_paths,
            lora_scales=lora_scale_values,
        )

        config = Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=reference_image,
        )
        layers = [int(idx.strip()) for idx in heatmap_layers.split(",") if idx.strip().isdigit()] or None
        timesteps = [int(idx.strip()) for idx in heatmap_timesteps.split(",") if idx.strip().isdigit()] or None

        progress(0.4, desc="Generating guided heatmap")
        result = flux.generate_image(
            seed=seed,
            prompt=prompt,
            concept=concept,
            image_path=reference_image,
            heatmap_layer_indices=layers,
            heatmap_timesteps=timesteps,
            config=config,
        )
        filename = f"concept_img_{int(time.time())}_{seed}.png"
        output_path = OUTPUT_DIR / filename
        result.save_with_heatmap(path=output_path, export_json_metadata=metadata)

        progress(1.0, desc="Done")
        status = f"Saved {filename} using reference image."
        return [result.image], [result.heatmap], status

    except Exception as exc:
        error_message = f"Concept image analysis failed: {exc}"
        print(error_message)
        return [], [], error_message
