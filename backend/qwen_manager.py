import random
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gradio as gr
from PIL import Image

from backend.mflux_compat import Config, ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

from backend.lora_manager import process_lora_files
from backend.model_manager import resolve_local_path

# Queue + progress doc references:
# - gradiodocs/guides-queuing/queuing.md
# - gradiodocs/guides-streaming-outputs/streaming_outputs.md

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_loras(lora_files, lora_scales):
    if not lora_files:
        return None, None
    paths = process_lora_files(lora_files)
    scales = process_lora_files(lora_files, lora_scales) if paths else None
    return paths, scales


def _prepare_seeds(requested_seed: Optional[str], count: int) -> List[int]:
    seeds: List[int] = []
    base_seed = None
    if requested_seed and requested_seed.strip():
        if requested_seed.strip().lower() == "random":
            base_seed = random.randint(0, 2**32 - 1)
        else:
            base_seed = int(requested_seed)
    for i in range(count):
        if base_seed is None:
            seeds.append(random.randint(0, 2**32 - 1))
        else:
            seeds.append(base_seed + i)
    return seeds


def generate_qwen_image_gradio(
    prompt: str,
    negative_prompt: str,
    model_name: str,
    base_model: Optional[str],
    seed: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    quantize_bits: Optional[int],
    lora_files,
    lora_scales,
    metadata: bool,
    init_image: Optional[str],
    num_images: int = 1,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not prompt:
        return [], "Prompt is required", prompt

    try:
        progress(0.0, desc="Loading Qwen")
        quant_value = None
        if quantize_bits not in (None, "", "None"):
            quant_value = int(quantize_bits)
        base_model = None if not base_model or base_model == "Auto" else base_model
        local_dir = resolve_local_path(model_name)
        model_cfg = ModelConfig.from_name(model_name or "qwen-image", base_model=base_model)
        lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

        qwen = QwenImage(
            model_config=model_cfg,
            quantize=quant_value,
            local_path=local_dir,
            lora_paths=lora_paths,
            lora_scales=lora_scale_values,
        )

        seeds = _prepare_seeds(seed, num_images)
        generated_images: List[Image.Image] = []
        info_lines: List[str] = []

        for idx, current_seed in enumerate(seeds, start=1):
            progress((idx - 1) / len(seeds), desc=f"Generating image {idx}/{len(seeds)}")
            config = Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                image_path=init_image,
            )
            result = qwen.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=config,
                negative_prompt=negative_prompt,
            )
            filename = f"qwen_{int(time.time())}_{current_seed}.png"
            output_path = OUTPUT_DIR / filename
            result.save(path=output_path, export_json_metadata=metadata)
            generated_images.append(result.image)
            info_lines.append(f"Seed {current_seed} → {filename}")

        progress(1.0, desc="Done")
        status = "\n".join(info_lines)
        return generated_images, status, prompt

    except Exception as exc:
        error_message = f"Qwen generation failed: {exc}"
        print(error_message)
        return [], error_message, prompt


def generate_qwen_edit_gradio(
    prompt: str,
    negative_prompt: str,
    image_paths: Sequence[str],
    seed: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    quantize_bits: Optional[int],
    lora_files,
    lora_scales,
    metadata: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not image_paths:
        return [], "Provide one or more reference images", prompt, []

    try:
        progress(0.0, desc="Loading Qwen Edit")
        local_dir = resolve_local_path("qwen-image-edit")
        lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

        quant_value = None
        if quantize_bits not in (None, "", "None"):
            quant_value = int(quantize_bits)

        qwen_edit = QwenImageEdit(
            quantize=quant_value,
            local_path=local_dir,
            lora_paths=lora_paths,
            lora_scales=lora_scale_values,
        )

        seeds = _prepare_seeds(seed, 1)
        generated_images: List[Image.Image] = []
        output_paths: List[str] = []

        for idx, current_seed in enumerate(seeds, start=1):
            progress((idx - 1) / len(seeds), desc=f"Editing image {idx}/{len(seeds)}")
            config = Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
            result = qwen_edit.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=config,
                negative_prompt=negative_prompt,
                image_paths=list(image_paths),
            )
            filename = f"qwen_edit_{int(time.time())}_{current_seed}.png"
            output_path = OUTPUT_DIR / filename
            result.save(path=output_path, export_json_metadata=metadata)
            generated_images.append(result.image)
            output_paths.append(str(output_path))

        progress(1.0, desc="Done")
        status = f"Generated {len(generated_images)} edit(s)."
        return generated_images, status, prompt, output_paths

    except Exception as exc:
        error_message = f"Qwen edit failed: {exc}"
        print(error_message)
        return [], error_message, prompt, []
