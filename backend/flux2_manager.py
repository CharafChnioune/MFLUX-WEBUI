import random
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gradio as gr

from backend.lora_manager import process_lora_files
from backend.model_manager import resolve_local_path, resolve_mflux_model_config, strip_quant_suffix

try:
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
except ModuleNotFoundError:  # pragma: no cover - optional model support
    Flux2Klein = None
    Flux2KleinEdit = None

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_seeds(requested_seed: Optional[str], count: int) -> List[int]:
    seeds: List[int] = []
    base_seed = None
    if requested_seed and requested_seed.strip():
        if requested_seed.strip().lower() == "random":
            base_seed = random.randint(0, 2**32 - 1)
        else:
            base_seed = int(requested_seed)
    for idx in range(count):
        if base_seed is None:
            seeds.append(random.randint(0, 2**32 - 1))
        else:
            seeds.append(base_seed + idx)
    return seeds


def _resolve_quantize(model_name: str) -> Optional[int]:
    if "-8-bit" in model_name:
        return 8
    if "-6-bit" in model_name:
        return 6
    if "-4-bit" in model_name:
        return 4
    if "-3-bit" in model_name:
        return 3
    return None


def _prepare_loras(lora_files, lora_scales) -> Tuple[Optional[List[str]], Optional[List[float]]]:
    if not lora_files:
        return None, None
    paths = process_lora_files(lora_files)
    scales = process_lora_files(lora_files, lora_scales) if paths else None
    return paths, scales


def generate_flux2_image_gradio(
    prompt: str,
    model_name: str,
    seed: str,
    width: int,
    height: int,
    steps: int,
    guidance: float | None,
    lora_files,
    lora_scales,
    metadata: bool,
    num_images: int = 1,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not prompt:
        return [], "Prompt is required", prompt

    if Flux2Klein is None:
        return [], "Flux2 is unavailable. Install mflux>=0.15.0.", prompt

    model_name = model_name or "flux2-klein-4b"
    resolved_name = strip_quant_suffix(model_name)
    is_base = "-base-" in (resolved_name or "").lower()
    model_config = resolve_mflux_model_config(resolved_name, None)
    local_dir = resolve_local_path(resolved_name)
    quantize = _resolve_quantize(model_name)
    lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

    flux2 = Flux2Klein(
        model_config=model_config,
        quantize=quantize,
        model_path=str(local_dir) if local_dir else None,
        lora_paths=lora_paths,
        lora_scales=lora_scale_values,
    )

    try:
        default_steps = 50 if is_base else 4
        steps_int = int(steps) if steps else default_steps
    except (TypeError, ValueError):
        steps_int = 50 if is_base else 4

    try:
        guidance_val = float(guidance) if guidance not in (None, "", "None") else 1.0
    except (TypeError, ValueError):
        guidance_val = 1.0
    if not is_base:
        guidance_val = 1.0

    try:
        total_images = max(1, int(num_images))
    except (TypeError, ValueError):
        total_images = 1

    seeds = _prepare_seeds(seed, total_images)
    generated_images: List = []
    info_lines: List[str] = []

    for idx, current_seed in enumerate(seeds, start=1):
        progress((idx - 1) / len(seeds), desc=f"Generating image {idx}/{len(seeds)}")
        result = flux2.generate_image(
            seed=current_seed,
            prompt=prompt,
            num_inference_steps=steps_int,
            height=height,
            width=width,
            guidance=guidance_val,
        )
        filename = f"flux2_{int(time.time())}_{current_seed}.png"
        output_path = OUTPUT_DIR / filename
        result.save(path=output_path, export_json_metadata=metadata)
        generated_images.append(result.image)
        info_lines.append(f"Seed {current_seed} → {filename}")

    progress(1.0, desc="Done")
    return generated_images, "\n".join(info_lines), prompt


def generate_flux2_edit_gradio(
    prompt: str,
    image_paths: Sequence[str],
    model_name: str,
    seed: str,
    width: int,
    height: int,
    steps: int,
    guidance: float | None,
    lora_files,
    lora_scales,
    metadata: bool,
    num_images: int = 1,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not image_paths:
        return [], "Provide one or more reference images", prompt

    if Flux2KleinEdit is None:
        return [], "Flux2 edit is unavailable. Install mflux>=0.15.0.", prompt

    model_name = model_name or "flux2-klein-4b"
    resolved_name = strip_quant_suffix(model_name)
    is_base = "-base-" in (resolved_name or "").lower()
    model_config = resolve_mflux_model_config(resolved_name, None)
    local_dir = resolve_local_path(resolved_name)
    quantize = _resolve_quantize(model_name)
    lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

    flux2_edit = Flux2KleinEdit(
        model_config=model_config,
        quantize=quantize,
        model_path=str(local_dir) if local_dir else None,
        lora_paths=lora_paths,
        lora_scales=lora_scale_values,
    )

    try:
        default_steps = 50 if is_base else 4
        steps_int = int(steps) if steps else default_steps
    except (TypeError, ValueError):
        steps_int = 50 if is_base else 4

    try:
        guidance_val = float(guidance) if guidance not in (None, "", "None") else 1.0
    except (TypeError, ValueError):
        guidance_val = 1.0
    if not is_base:
        guidance_val = 1.0

    try:
        total_images = max(1, int(num_images))
    except (TypeError, ValueError):
        total_images = 1

    seeds = _prepare_seeds(seed, total_images)
    generated_images: List = []

    normalized_paths = [str(p) for p in image_paths]

    for idx, current_seed in enumerate(seeds, start=1):
        progress((idx - 1) / len(seeds), desc=f"Editing image {idx}/{len(seeds)}")
        result = flux2_edit.generate_image(
            seed=current_seed,
            prompt=prompt,
            num_inference_steps=steps_int,
            height=height,
            width=width,
            guidance=guidance_val,
            image_paths=normalized_paths,
        )
        filename = f"flux2_edit_{int(time.time())}_{current_seed}.png"
        output_path = OUTPUT_DIR / filename
        result.save(path=output_path, export_json_metadata=metadata)
        generated_images.append(result.image)

    progress(1.0, desc="Done")
    status = f"Generated {len(generated_images)} edit(s)."
    return generated_images, status, prompt
