from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
try:
    from mflux.models.depth_pro.model.depth_pro import DepthPro
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    try:
        from mflux.models.depth_pro.depth_pro import DepthPro  # type: ignore
    except ModuleNotFoundError:
        from mflux.depth.depth_pro import DepthPro  # type: ignore

# Component docs referenced:
# - gradiodocs/docs-image/image.md (file inputs/outputs)
# - gradiodocs/guides-progress-bars/progress_bars.md


def export_depth_maps(
    image_paths: List[str],
    output_dir: str,
    quantize_bits: Optional[int] = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[List[str], str]:
    if not image_paths:
        raise ValueError("Select at least one image to export depth maps.")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    quant_value = None
    if quantize_bits not in (None, "", "None"):
        quant_value = int(quantize_bits)

    depth_pro = DepthPro(quantize=quant_value)
    saved_paths: List[str] = []

    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        progress((idx - 1) / total, desc=f"Processing {idx}/{total}")
        depth_result = depth_pro.create_depth_map(image_path=image_path)
        stem = Path(image_path).stem
        target_path = destination / f"{stem}_depth.png"
        depth_result.depth_image.save(target_path)
        saved_paths.append(str(target_path))

    progress(1.0, desc="Depth export complete")
    return saved_paths, f"Saved {len(saved_paths)} depth map(s) to {destination}"
