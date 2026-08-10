from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _coerce_seed(seed: Optional[int | str]) -> int:
    if seed in (None, "", -1, "-1"):
        return random.randint(0, 2**32 - 1)
    try:
        return int(seed)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return random.randint(0, 2**32 - 1)


def _seedvr2_runtime():
    """Import the optional SeedVR2 runtime only when it is actually used."""
    try:
        from mflux.models.seedvr2 import SeedVR2
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - legacy fallback
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

    try:
        from mflux.models.common.config.model_config import ModelConfig
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - legacy fallback
        from mflux.config.model_config import ModelConfig

    from mflux.utils.scale_factor import ScaleFactor

    return SeedVR2, ModelConfig, ScaleFactor


def coerce_seedvr2_resolution(value: str | int) -> Any:
    """Convert a validated API resolution into MFLUX' native representation."""
    _, _, scale_factor = _seedvr2_runtime()
    if isinstance(value, str) and value.strip().lower().endswith("x"):
        return scale_factor.parse(value.strip().lower())
    return int(value)


def _coerce_resolution(value: str | int) -> Tuple[Any, str]:
    try:
        return coerce_seedvr2_resolution(value), (
            "scale" if isinstance(value, str) and value.strip().lower().endswith("x") else "pixels"
        )
    except Exception:
        return 384, "pixels"


def load_seedvr2_model(model_name: str = "seedvr2-3b"):
    """Load SeedVR2 3B from the shared Hugging Face cache without network access."""
    if model_name != "seedvr2-3b":
        raise ValueError("Only the verified SeedVR2 3B model is supported.")

    SeedVR2, ModelConfig, _ = _seedvr2_runtime()
    from huggingface_hub import snapshot_download

    model_config = ModelConfig.seedvr2_3b()
    repo_id = getattr(model_config, "model_name", None)
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("The installed MFLUX release does not expose the SeedVR2 repository.")
    required_files = [
        "seedvr2_ema_3b_fp16.safetensors",
        "ema_vae_fp16.safetensors",
    ]
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=required_files,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "SeedVR2 3B is not complete in the shared Hugging Face cache; "
            "the local-only batch will not download it automatically."
        ) from exc

    if any(not (Path(model_path) / filename).is_file() for filename in required_files):
        raise RuntimeError(
            "SeedVR2 3B is not complete in the shared Hugging Face cache; "
            "the local-only batch will not download it automatically."
        )

    return SeedVR2(model_config=model_config, model_path=model_path)


def generate_seedvr2_upscale(
    image_path,
    resolution,
    softness,
    seed,
    metadata,
    progress=None,
):
    """Legacy Gradio entrypoint, backed by the same cache-only model loader."""
    if not image_path:
        return None, "Input image is required", ""

    if progress is None:
        progress = lambda *_args, **_kwargs: None

    try:
        progress(0, desc="Loading model")
        model = load_seedvr2_model()
        res_value, res_kind = _coerce_resolution(resolution)
        seed_value = _coerce_seed(seed)

        progress(0.1, desc="Upscaling")
        result = model.generate_image(
            seed=seed_value,
            image_path=image_path,
            resolution=res_value,
            softness=float(softness) if softness is not None else 0.0,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"upscaled_seedvr2_{int(time.time())}_{seed_value}.png"
        output_path = OUTPUT_DIR / filename
        result.save(path=output_path, export_json_metadata=bool(metadata))
    except Exception as exc:
        return None, f"SeedVR2 unavailable: {exc}", ""

    progress(1.0, desc="Done")
    status = f"Saved {filename} (resolution: {resolution}, mode: {res_kind}, seed: {seed_value})"
    return result.image, status, str(output_path)


__all__ = [
    "coerce_seedvr2_resolution",
    "generate_seedvr2_upscale",
    "load_seedvr2_model",
]
