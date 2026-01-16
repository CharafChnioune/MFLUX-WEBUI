import os
import random
import time
from pathlib import Path
from typing import List, Optional

import gradio as gr
from PIL import Image

from backend.lora_manager import process_lora_files

from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_seeds(requested_seed: Optional[str], count: int) -> List[int]:
    seeds: List[int] = []
    base_seed = None
    if requested_seed and str(requested_seed).strip():
        if str(requested_seed).strip().lower() == "random":
            base_seed = random.randint(0, 2**32 - 1)
        else:
            base_seed = int(float(requested_seed))
    for i in range(count):
        if base_seed is None:
            seeds.append(random.randint(0, 2**32 - 1))
        else:
            seeds.append(base_seed + i)
    return seeds


def _ensure_image_path(image_input, prefix: str) -> Optional[str]:
    if not image_input:
        return None
    if isinstance(image_input, (str, Path)):
        return str(image_input)
    if hasattr(image_input, "read"):
        image = Image.open(image_input)
    else:
        image = image_input
    if not isinstance(image, Image.Image):
        raise ValueError("Unsupported image input type")
    timestamp = int(time.time())
    path = OUTPUT_DIR / f"{prefix}_{timestamp}.png"
    image.convert("RGB").save(path)
    return str(path)


def _parse_quantize(value) -> Optional[int]:
    if value in (None, "", "None", 0, "0"):
        return None
    return int(value)


def generate_z_image_gradio(
    prompt: str,
    model_path: str,
    quantize_bits,
    seed: str,
    width: int,
    height: int,
    steps: int,
    scheduler: str,
    lora_files,
    lora_scales,
    metadata: bool,
    init_image,
    image_strength: float,
    num_images: int = 1,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not prompt:
        return [], "Prompt is required", prompt

    try:
        quant_value = _parse_quantize(quantize_bits)
        lora_paths = process_lora_files(lora_files) if lora_files else None
        lora_scales_float = process_lora_files(lora_files, lora_scales) if lora_paths else None

        model = ZImageTurbo(
            quantize=quant_value,
            model_path=model_path or None,
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
        )

        init_image_path = _ensure_image_path(init_image, "z_image_init")
        seeds = _prepare_seeds(seed, int(num_images) if num_images else 1)
        generated_images: List[Image.Image] = []
        info_lines: List[str] = []

        for idx, current_seed in enumerate(seeds, start=1):
            progress((idx - 1) / len(seeds), desc=f"Generating {idx}/{len(seeds)}")
            result = model.generate_image(
                seed=current_seed,
                prompt=prompt,
                num_inference_steps=int(steps),
                height=int(height),
                width=int(width),
                image_path=init_image_path,
                image_strength=image_strength if image_strength not in (None, "", 0, "0") else None,
                scheduler=scheduler or "linear",
            )
            filename = f"z_image_{int(time.time())}_{current_seed}.png"
            output_path = OUTPUT_DIR / filename
            result.save(path=output_path, export_json_metadata=metadata)
            generated_images.append(result.image)
            info_lines.append(f"Seed {current_seed} -> {filename}")

        progress(1.0, desc="Done")
        status = "\n".join(info_lines) if info_lines else "No images generated"
        return generated_images, status, prompt

    except Exception as exc:  # noqa: BLE001
        error_message = f"Z-Image generation failed: {exc}"
        print(error_message)
        return [], error_message, prompt
