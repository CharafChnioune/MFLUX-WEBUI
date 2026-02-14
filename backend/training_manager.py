from __future__ import annotations

import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from PIL import Image

# Keep debug artifacts inside the repo (workspace-write safe).
configs_dir = Path(__file__).resolve().parent.parent / "configs"
configs_dir.mkdir(parents=True, exist_ok=True)


def _parse_square_size(value: str) -> int:
    """
    UI provides sizes like "512x512". For training we use max_resolution (largest side cap).
    """
    raw = (value or "").strip().lower()
    if "x" not in raw:
        return int(float(raw))
    left = raw.split("x", 1)[0].strip()
    return int(float(left))


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _resize_image(input_path: str, output_path: str, max_side: int) -> None:
    """
    Resize while preserving aspect ratio. Training also supports mixed aspect ratios,
    but this keeps memory predictable for UI users.
    """
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        if max(width, height) <= max_side:
            img.save(output_path)
            return
        ratio = max_side / float(max(width, height))
        new_size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        resized.save(output_path)


def _training_defaults(model_name: str) -> tuple[int, float]:
    """
    Return (steps, guidance) defaults matching MFLUX 0.16+ docs.
    """
    m = (model_name or "").strip().lower()
    if m in {"z-image-turbo", "zimage-turbo"}:
        return 9, 0.0
    if m in {"z-image", "zimage"}:
        return 50, 4.0
    if m.startswith("flux2-") and "-base-" in m:
        return 40, 1.0
    return 50, 1.0


def _lora_targets_for_model(
    model_name: str,
    *,
    lora_rank: int,
    transformer_blocks_enabled: bool,
    transformer_start: int,
    transformer_end: int,
    single_blocks_enabled: bool,
    single_start: int,
    single_end: int,
) -> list[dict]:
    model = (model_name or "").strip().lower()
    rank = int(lora_rank)

    if model in {"z-image", "zimage", "z-image-turbo", "zimage-turbo"}:
        if not transformer_blocks_enabled:
            raise ValueError("Z-Image training requires Transformer Blocks enabled (layers.{block}.* targets).")
        if transformer_end <= transformer_start:
            raise ValueError("Transformer block range must have end > start.")
        blocks = {"start": int(transformer_start), "end": int(transformer_end)}
        # Minimal, stable targets (matches Z-Image README example).
        return [
            {"module_path": "layers.{block}.attention.to_q", "blocks": blocks, "rank": rank},
            {"module_path": "layers.{block}.attention.to_k", "blocks": blocks, "rank": rank},
            {"module_path": "layers.{block}.attention.to_v", "blocks": blocks, "rank": rank},
        ]

    if model.startswith("flux2-") and "-base-" in model:
        targets: list[dict] = []
        if transformer_blocks_enabled:
            if transformer_end <= transformer_start:
                raise ValueError("Transformer block range must have end > start.")
            blocks = {"start": int(transformer_start), "end": int(transformer_end)}
            for module_path in (
                "transformer_blocks.{block}.attn.to_q",
                "transformer_blocks.{block}.attn.to_k",
                "transformer_blocks.{block}.attn.to_v",
                "transformer_blocks.{block}.attn.to_out",
                "transformer_blocks.{block}.attn.add_q_proj",
                "transformer_blocks.{block}.attn.add_k_proj",
                "transformer_blocks.{block}.attn.add_v_proj",
                "transformer_blocks.{block}.attn.to_add_out",
                "transformer_blocks.{block}.ff.linear_in",
                "transformer_blocks.{block}.ff.linear_out",
                "transformer_blocks.{block}.ff_context.linear_in",
                "transformer_blocks.{block}.ff_context.linear_out",
            ):
                targets.append({"module_path": module_path, "blocks": blocks, "rank": rank})

        if single_blocks_enabled:
            if single_end <= single_start:
                raise ValueError("Single transformer block range must have end > start.")
            blocks = {"start": int(single_start), "end": int(single_end)}
            for module_path in (
                "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj",
                "single_transformer_blocks.{block}.attn.to_out",
            ):
                targets.append({"module_path": module_path, "blocks": blocks, "rank": rank})

        if not targets:
            raise ValueError("No LoRA targets selected. Enable Transformer Blocks and/or Single Transformer Blocks.")
        return targets

    raise ValueError(
        f"Unsupported training model: {model_name!r}. Supported: z-image, z-image-turbo, flux2-klein-base-4b/9b."
    )


def prepare_training_config(
    model_name: str,
    *,
    data_path: str,
    output_path: str,
    seed: int,
    steps: int,
    guidance: float,
    quantize: int | None,
    max_resolution: int,
    low_ram: bool,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_frequency: int,
    lora_targets: list[dict],
) -> dict:
    """
    Generate an MFLUX >= 0.16 training config (used by `mflux-train`).
    """
    return {
        "model": model_name,
        "data": data_path,
        "seed": int(seed),
        "steps": int(steps),
        "guidance": float(guidance),
        "quantize": None if quantize in (None, 0, "0", "None") else int(quantize),
        "max_resolution": int(max_resolution) if max_resolution else None,
        "low_ram": bool(low_ram),
        "training_loop": {
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
        },
        "checkpoint": {
            "save_frequency": int(checkpoint_frequency),
            "output_path": str(output_path),
        },
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": int(checkpoint_frequency),
        },
        "lora_layers": {
            "targets": lora_targets,
        },
    }


def run_training(
    base_model,
    uploaded_files,
    trigger_word,
    epochs,
    batch_size,
    lora_rank,
    learning_rate,
    seed,
    checkpoint_frequency,
    validation_prompt,
    guidance_scale,
    quantize_bits,
    low_ram_mode,
    output_dir,
    resume_checkpoint,
    transformer_blocks_enabled,
    transformer_start,
    transformer_end,
    single_blocks_enabled,
    single_start,
    single_end,
    image_size,
    *captions,
) -> Iterator[str]:
    """
    Run LoRA training using `mflux-train` (MFLUX >= 0.16).

    The UI still calls this "Dreambooth", but upstream training is now a common LoRA stack.
    """
    try:
        model_name = str(base_model or "").strip()
        if not model_name:
            raise ValueError("Training model is required.")

        resume_checkpoint = str(resume_checkpoint or "").strip()
        output_root = Path(os.path.expanduser(str(output_dir or ""))).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_suffix = f"{random.randint(1000, 9999)}"
        run_root = output_root / f"lora_train_{run_id}_{run_suffix}"
        run_root.mkdir(parents=True, exist_ok=False)
        data_dir = run_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        yield f"Run folder: {run_root}\n"

        # Resume mode: don't create new data/config; just run resume.
        if resume_checkpoint:
            ckpt_path = Path(os.path.expanduser(resume_checkpoint))
            if not ckpt_path.exists():
                raise ValueError(f"Checkpoint not found: {ckpt_path}")
            cmd = [sys.executable, "-m", "mflux.models.common.cli.train", "--resume", str(ckpt_path)]
            yield f"Starting resume:\n{' '.join(cmd)}\n\n"
            yield from _run_subprocess_stream(cmd)
            return

        if not uploaded_files:
            raise ValueError("Please upload training images (or provide a resume checkpoint).")

        max_resolution = _parse_square_size(str(image_size or "1024x1024"))
        steps_default, guidance_default = _training_defaults(model_name)

        steps = steps_default
        guidance = guidance_default if guidance_scale in (None, "", "None") else _safe_float(guidance_scale, guidance_default)
        if str(model_name).strip().lower() in {"z-image-turbo", "zimage-turbo"}:
            guidance = 0.0

        # Build LoRA targets
        targets = _lora_targets_for_model(
            model_name,
            lora_rank=_safe_int(lora_rank, 8),
            transformer_blocks_enabled=bool(transformer_blocks_enabled),
            transformer_start=_safe_int(transformer_start, 0),
            transformer_end=_safe_int(transformer_end, 30),
            single_blocks_enabled=bool(single_blocks_enabled),
            single_start=_safe_int(single_start, 0),
            single_end=_safe_int(single_end, 20),
        )

        # Prepare dataset (limit to the caption UI)
        captions_list = list(captions)
        used_files = list(uploaded_files)[: max(1, len(captions_list))]

        yield f"Preparing dataset: {len(used_files)} image(s)\n"
        for idx, fdata in enumerate(used_files, start=1):
            in_path = getattr(fdata, "name", None) or str(fdata)
            stem = f"{idx:03d}"
            out_img = data_dir / f"{stem}.png"
            out_txt = data_dir / f"{stem}.txt"

            prompt = ""
            if idx - 1 < len(captions_list):
                prompt = str(captions_list[idx - 1] or "").strip()
            if not prompt:
                prompt = str(trigger_word or "").strip() or "a photo"

            _resize_image(in_path, str(out_img), max_side=max_resolution)
            out_txt.write_text(prompt, encoding="utf-8")

        # Optional preview prompt (if provided)
        preview_prompt = str(validation_prompt or "").strip()
        if preview_prompt:
            (data_dir / "preview_1.txt").write_text(preview_prompt, encoding="utf-8")

        # Training config lives next to the data for reproducibility/resume.
        config_path = run_root / "train.json"
        debug_path = configs_dir / "train_debug.json"

        quantize = None if quantize_bits in (None, "", "None", 0, "0") else _safe_int(quantize_bits, 0)

        config = prepare_training_config(
            model_name=model_name,
            data_path="data",
            output_path=str(run_root / "output"),
            seed=_safe_int(seed, 42),
            steps=steps,
            guidance=guidance,
            quantize=quantize or None,
            max_resolution=max_resolution,
            low_ram=bool(low_ram_mode),
            num_epochs=_safe_int(epochs, 50),
            batch_size=_safe_int(batch_size, 1),
            learning_rate=_safe_float(learning_rate, 1e-4),
            checkpoint_frequency=_safe_int(checkpoint_frequency, 25),
            lora_targets=targets,
        )

        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        debug_path.write_text(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "model": model_name,
                    "image_size": str(image_size),
                    "max_resolution": max_resolution,
                    "uploaded_files": [getattr(f, "name", str(f)) for f in used_files],
                    "quantize": quantize,
                    "guidance": guidance,
                    "steps": steps,
                    "epochs": _safe_int(epochs, 50),
                    "batch_size": _safe_int(batch_size, 1),
                    "learning_rate": _safe_float(learning_rate, 1e-4),
                    "checkpoint_frequency": _safe_int(checkpoint_frequency, 25),
                    "low_ram": bool(low_ram_mode),
                    "transformer_blocks_enabled": bool(transformer_blocks_enabled),
                    "transformer_start": _safe_int(transformer_start, 0),
                    "transformer_end": _safe_int(transformer_end, 30),
                    "single_blocks_enabled": bool(single_blocks_enabled),
                    "single_start": _safe_int(single_start, 0),
                    "single_end": _safe_int(single_end, 20),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        yield f"Saved training config: {config_path}\n"
        yield f"Saved debug config: {debug_path}\n"

        cmd = [sys.executable, "-m", "mflux.models.common.cli.train", "--config", str(config_path)]
        env = os.environ.copy()
        env.setdefault("HF_TOKEN", "")
        env.setdefault("HUGGING_FACE_HUB_TOKEN", "")

        yield f"\nStarting training:\n{' '.join(cmd)}\n\n"
        yield from _run_subprocess_stream(cmd, cwd=str(run_root), env=env)

        yield f"\nTraining finished. Outputs: {run_root / 'output'}\n"

    except Exception as exc:  # noqa: BLE001
        yield f"Error during training: {exc}\n\n{traceback.format_exc()}"


def _run_subprocess_stream(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict | None = None,
) -> Iterator[str]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    output_queue: "queue.Queue[str]" = queue.Queue()

    def enqueue_output(pipe, prefix: str):
        for line in iter(pipe.readline, ""):
            output_queue.put(f"[{prefix}] {line}")
        pipe.close()

    stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, "STDOUT"), daemon=True)
    stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, "STDERR"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    buffer = ""
    last_flush = time.time()

    while process.poll() is None:
        try:
            chunk = output_queue.get_nowait()
            buffer += chunk
            if time.time() - last_flush > 0.5 or len(buffer) > 2000:
                yield buffer
                buffer = ""
                last_flush = time.time()
        except queue.Empty:
            if buffer and (time.time() - last_flush > 0.5):
                yield buffer
                buffer = ""
                last_flush = time.time()
            time.sleep(0.1)

    # Drain remaining output
    while True:
        try:
            buffer += output_queue.get_nowait()
        except queue.Empty:
            break

    if buffer:
        yield buffer

    if process.returncode != 0:
        yield f"\n❌ Training process exited with code {process.returncode}\n"
    else:
        yield "\n✅ Training completed successfully!\n"


def run_dreambooth_from_ui_no_explicit_quantize(*args, **kwargs) -> Iterator[str]:
    """
    Backwards-compatible wrapper: the UI still calls this name.
    """
    yield from run_training(*args, **kwargs)

