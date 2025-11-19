import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

from backend.lora_manager import process_lora_files
from backend.model_manager import (
    CustomModelConfig,
    get_custom_model_config,
    get_model_choices,
    register_local_model,
    resolve_local_path,
)

# Progress + file access references:
# - gradiodocs/guides-progress-bars/progress_bars.md
# - gradiodocs/guides-file-access/security_and_file_access.md

MIN_EXPORT_DISK_BYTES = 8 * 1024**3  # conservative default (~8 GB)


def _prepare_loras(selected: Optional[List[str]], scales=None) -> Tuple[Optional[List[str]], Optional[List[float]]]:
    if not selected:
        return None, None
    lora_paths = process_lora_files(selected)
    lora_scales = process_lora_files(selected, scales) if lora_paths else None
    return lora_paths, lora_scales


def _resolve_flux_config(model_name: str, alias: str, base_arch: Optional[str], local_dir: Optional[Path]):
    base_arch = base_arch or "schnell"
    return CustomModelConfig(
        model_name=model_name,
        alias=alias,
        num_train_steps=1000,
        max_sequence_length=512 if base_arch != "schnell" else 256,
        base_arch=base_arch,
        local_dir=local_dir,
    )


def quantize_model(
    source_model: str,
    new_alias: str,
    output_path: str,
    quantize_bits: int,
    base_model: Optional[str] = None,
    lora_files: Optional[List[str]] = None,
    lora_scales=None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """
    Run mflux.save/mflux.save_depth style exports from the UI.
    """
    logs = []
    try:
        progress(0.0, desc="Validating request")
        if not source_model:
            raise ValueError("Select a source model to quantize.")
        if not output_path:
            if not new_alias:
                raise ValueError("Provide an output path or alias.")
            output_path = str(Path("models") / new_alias)

        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        stat = shutil.disk_usage(output_dir.parent)
        if stat.free < MIN_EXPORT_DISK_BYTES:
            raise RuntimeError(
                f"Not enough free space in {output_dir.parent}. Need at least {MIN_EXPORT_DISK_BYTES / (1024 ** 3):.1f} GB."
            )

        progress(0.05, desc="Loading LoRAs")
        if base_model in (None, "", "Auto"):
            base_model = None
        lora_paths, lora_scale_values = _prepare_loras(lora_files, lora_scales)

        model_identifier = source_model
        local_dir = resolve_local_path(source_model)
        base_arch = base_model
        config_found = False
        src_config: Optional[CustomModelConfig] = None
        try:
            src_config = get_custom_model_config(source_model)
            model_identifier = src_config.model_name
            base_arch = base_arch or src_config.base_arch
            local_dir = local_dir or src_config.local_dir
            config_found = True
        except ValueError:
            base_arch = base_arch or "schnell"

        progress(0.15, desc="Instantiating model")
        if "qwen" in model_identifier.lower():
            model_cfg = ModelConfig.from_name(model_name=model_identifier, base_model=base_arch)
            model = QwenImage(
                model_config=model_cfg,
                quantize=int(quantize_bits),
                local_path=local_dir,
                lora_paths=lora_paths,
                lora_scales=lora_scale_values,
            )
        else:
            flux_cfg = src_config if config_found else _resolve_flux_config(
                model_identifier, source_model, base_arch, local_dir
            )
            model = Flux1(
                model_config=flux_cfg,
                quantize=int(quantize_bits),
                local_path=local_dir,
                lora_paths=lora_paths,
                lora_scales=lora_scale_values,
            )

        progress(0.4, desc="Saving quantized weights")
        start_ts = time.time()
        model.save_model(str(output_dir))
        duration = time.time() - start_ts
        logs.append(f"Quantized model saved to {output_dir} in {duration:.1f}s.")

        registered_alias = new_alias or output_dir.name
        register_local_model(registered_alias, model_identifier, base_arch or "schnell", output_dir)
        logs.append(f"Registered '{registered_alias}' for future sessions.")

        model_choices = get_model_choices()
        status = f"Quantization complete ({registered_alias}, {quantize_bits}-bit)."
        progress(1.0, desc="Finished")
        return (
            model_choices,
            model_choices,
            model_choices,
            model_choices,
            model_choices,
            status,
            "\n".join(logs),
        )

    except Exception as exc:
        error_message = f"Quantization failed: {exc}"
        print(error_message)
        model_choices = get_model_choices()
        return (
            model_choices,
            model_choices,
            model_choices,
            model_choices,
            model_choices,
            error_message,
            "\n".join(logs),
        )
