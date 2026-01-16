import gc
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import gradio as gr
from PIL import Image

from backend.lora_manager import process_lora_files
from backend.mlx_utils import force_mlx_cleanup

from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

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


def _ensure_json_prompt(
    prompt: str,
    use_vlm: bool,
    vlm_quantize: Optional[int],
    top_p: float,
    temperature: float,
    max_tokens: int,
    vlm_seed: Optional[int],
) -> str:
    try:
        json.loads(prompt)
        return prompt
    except json.JSONDecodeError:
        if not use_vlm:
            raise ValueError("Prompt is not valid JSON. Enable VLM conversion or provide JSON.")
        vlm = FiboVLM(quantize=vlm_quantize)
        try:
            return vlm.generate(
                prompt=prompt,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=vlm_seed,
            )
        finally:
            del vlm
            gc.collect()
            force_mlx_cleanup()


def generate_fibo_gradio(
    prompt_text: str,
    json_prompt_text: str,
    prompt_mode: str,
    negative_prompt: str,
    model_path: str,
    quantize_bits,
    seed: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    scheduler: str,
    lora_files,
    lora_scales,
    metadata: bool,
    init_image,
    image_strength: float,
    use_vlm: bool,
    vlm_quantize,
    vlm_top_p: float,
    vlm_temperature: float,
    vlm_max_tokens: int,
    vlm_seed: Optional[int],
    num_images: int = 1,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if prompt_mode == "JSON":
        if not json_prompt_text:
            return [], "JSON prompt is required", ""
        effective_prompt = json_prompt_text
    else:
        if not prompt_text:
            return [], "Prompt is required", ""
        effective_prompt = prompt_text

    try:
        quant_value = _parse_quantize(quantize_bits)
        vlm_quant_value = _parse_quantize(vlm_quantize)

        json_prompt = _ensure_json_prompt(
            effective_prompt,
            use_vlm,
            vlm_quant_value,
            top_p=float(vlm_top_p),
            temperature=float(vlm_temperature),
            max_tokens=int(vlm_max_tokens),
            vlm_seed=vlm_seed if vlm_seed not in (None, "", "None") else None,
        )

        lora_paths = process_lora_files(lora_files) if lora_files else None
        lora_scales_float = process_lora_files(lora_files, lora_scales) if lora_paths else None

        fibo = FIBO(
            quantize=quant_value,
            model_path=model_path or None,
            lora_paths=lora_paths,
            lora_scales=lora_scales_float,
        )

        init_image_path = _ensure_image_path(init_image, "fibo_init")
        seeds = _prepare_seeds(seed, int(num_images) if num_images else 1)
        generated_images: List[Image.Image] = []
        info_lines: List[str] = []

        for idx, current_seed in enumerate(seeds, start=1):
            progress((idx - 1) / len(seeds), desc=f"Generating {idx}/{len(seeds)}")
            result = fibo.generate_image(
                seed=current_seed,
                prompt=json_prompt,
                num_inference_steps=int(steps),
                height=int(height),
                width=int(width),
                guidance=float(guidance),
                image_path=init_image_path,
                image_strength=image_strength if image_strength not in (None, "", 0, "0") else None,
                negative_prompt=negative_prompt or None,
                scheduler=scheduler or "flow_match_euler_discrete",
            )
            filename = f"fibo_{int(time.time())}_{current_seed}.png"
            output_path = OUTPUT_DIR / filename
            result.save(path=output_path, export_json_metadata=metadata)
            generated_images.append(result.image)
            info_lines.append(f"Seed {current_seed} -> {filename}")

        progress(1.0, desc="Done")
        status = "\n".join(info_lines) if info_lines else "No images generated"
        return generated_images, status, json_prompt

    except Exception as exc:  # noqa: BLE001
        error_message = f"FIBO generation failed: {exc}"
        print(error_message)
        return [], error_message, ""
