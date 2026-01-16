import gc
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

from backend.mflux_compat import ModelConfig
from backend.mlx_utils import force_mlx_cleanup

try:
    from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
    from mflux.utils.scale_factor import ScaleFactor
except Exception as exc:  # pragma: no cover - optional dependency
    SeedVR2 = None  # type: ignore
    ScaleFactor = None  # type: ignore
    _seedvr2_import_error = exc
else:
    _seedvr2_import_error = None


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_seed(seed: Optional[int | str]) -> int:
    if seed in (None, "", -1, "-1"):
        return random.randint(0, 2**32 - 1)
    try:
        return int(seed)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return random.randint(0, 2**32 - 1)


def _coerce_resolution(value: str | int) -> Tuple[int | "ScaleFactor", str]:
    if ScaleFactor and isinstance(value, str) and value.strip().lower().endswith("x"):
        try:
            return ScaleFactor.parse(value), "scale"
        except Exception:
            pass
    try:
        return int(float(value)), "pixels"
    except Exception:
        return 384, "pixels"


def generate_seedvr2_upscale(
    image_path,
    resolution,
    softness,
    seed,
    metadata,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if SeedVR2 is None:
        return None, f"SeedVR2 unavailable: {_seedvr2_import_error}", ""
    if not image_path:
        return None, "Input image is required", ""

    progress(0, desc="Loading model")
    model = SeedVR2(model_config=ModelConfig.seedvr2_3b())

    res_value, res_kind = _coerce_resolution(resolution)
    seed_value = _coerce_seed(seed)

    progress(0.1, desc="Upscaling")
    result = model.generate_image(
        seed=seed_value,
        image_path=image_path,
        resolution=res_value,
        softness=float(softness) if softness is not None else 0.0,
    )

    filename = f"upscaled_seedvr2_{int(time.time())}_{seed_value}.png"
    output_path = OUTPUT_DIR / filename
    result.save(path=output_path, export_json_metadata=bool(metadata))

    progress(1.0, desc="Done")
    status = f"Saved {filename} (resolution: {resolution}, mode: {res_kind}, seed: {seed_value})"
    return result.image, status, str(output_path)


__all__ = ["generate_seedvr2_upscale"]

