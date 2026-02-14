import os
import re
from pathlib import Path
from shutil import disk_usage
from typing import Dict, List, Optional

import gradio as gr
from huggingface_hub import HfApi, snapshot_download

from backend.mflux_compat import ModelConfig as MfluxModelConfig
# Layout/state docs referenced across the UI:
# - gradiodocs/docs-blocks/blocks.md (consistent Blocks state)
# - gradiodocs/guides-controlling-layout/controlling_layout.md (Row/Column ordering)
# - gradiodocs/guides-interface-state/interface_state.md (shared state between tabs)

BASE_MODEL_CHOICES = ["flux2-klein-4b", "flux2-klein-9b"]
MODELS: Dict[str, "CustomModelConfig"] = {}


class CustomModelConfig:
    """Thin proxy to keep Flux + Qwen loaders happy inside the UI flow."""

    def __init__(
        self,
        model_name: str,
        alias: str,
        num_train_steps: int,
        max_sequence_length: int,
        base_arch: str = "schnell",
        local_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.alias = alias
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.base_arch = base_arch
        self.local_dir = Path(local_dir) if local_dir else None
        self.supports_guidance = base_arch in {"dev", "krea-dev"}
        self.custom_transformer_model = model_name  # compatibility shim
        self.requires_sigma_shift = base_arch == "krea-dev"

    def is_dev(self) -> bool:
        return self.base_arch in {"dev", "krea-dev"}

    def x_embedder_input_dim(self) -> int:
        # Required by the mflux Transformer signature
        return 3072

    @staticmethod
    def from_alias(alias: str) -> "CustomModelConfig":
        return get_custom_model_config(alias)


def _register_default_models():
    """Register official checkpoints plus quantized variants."""
    official = [
        ("flux2-klein-4b", "black-forest-labs/FLUX.2-klein-4B", 512, "flux2"),
        ("flux2-klein-4b-mlx-4bit", "AITRADER/FLUX2-klein-4B-mlx-4bit", 512, "flux2"),
        ("flux2-klein-4b-mlx-8bit", "AITRADER/FLUX2-klein-4B-mlx-8bit", 512, "flux2"),
        ("flux2-klein-9b", "black-forest-labs/FLUX.2-klein-9B", 512, "flux2"),
        ("flux2-klein-9b-mlx-4bit", "AITRADER/FLUX2-klein-9B-mlx-4bit", 512, "flux2"),
        ("flux2-klein-9b-mlx-8bit", "AITRADER/FLUX2-klein-9B-mlx-8bit", 512, "flux2"),
        ("flux2-klein-base-4b", "black-forest-labs/FLUX.2-klein-base-4B", 512, "flux2"),
        ("flux2-klein-base-4b-mlx-4bit", "AITRADER/FLUX2-klein-base-4B-mlx-4bit", 512, "flux2"),
        ("flux2-klein-base-4b-mlx-8bit", "AITRADER/FLUX2-klein-base-4B-mlx-8bit", 512, "flux2"),
        ("flux2-klein-base-9b", "black-forest-labs/FLUX.2-klein-base-9B", 512, "flux2"),
        # Experimental: upstream mflux does not currently ship a Flux2-dev implementation.
        # This entry is for the WebUI's experimental loader branch.
        ("flux2-dev", "black-forest-labs/FLUX.2-dev", 512, "flux2"),
        ("seedvr2", "numz/SeedVR2_comfyUI", 512, "dev"),
    ]
    for alias, repo, seq_len, base_arch in official:
        MODELS[alias] = CustomModelConfig(repo, alias, 1000, seq_len, base_arch)

    for alias, repo, seq_len, base_arch in official:
        if alias == "dev-krea":
            continue  # share canonical entry with krea-dev
        if "-mlx-" in alias:
            continue  # pre-quantized models don't need runtime quantization variants
        for bits in ("3", "4", "6", "8"):
            MODELS[f"{alias}-{bits}-bit"] = CustomModelConfig(
                repo, f"{alias}-{bits}-bit", 1000, seq_len, base_arch
            )


_register_default_models()


def get_custom_model_config(model_alias: str) -> CustomModelConfig:
    config = MODELS.get(model_alias)
    if config is None:
        raise ValueError(
            f"Invalid model alias: {model_alias}. Available aliases: {', '.join(sorted(MODELS.keys()))}"
        )
    return config


def get_base_model_choices() -> List[str]:
    return BASE_MODEL_CHOICES.copy()


def _flux2_ordered() -> List[str]:
    return [
        "flux2-klein-4b",
        "flux2-klein-4b-mlx-4bit",  # Pre-quantized (faster load)
        "flux2-klein-4b-mlx-8bit",  # Pre-quantized (faster load)
        "flux2-klein-4b-3-bit",
        "flux2-klein-4b-4-bit",
        "flux2-klein-4b-6-bit",
        "flux2-klein-4b-8-bit",
        "flux2-klein-9b",
        "flux2-klein-9b-mlx-4bit",  # Pre-quantized (faster load)
        "flux2-klein-9b-mlx-8bit",  # Pre-quantized (faster load)
        "flux2-klein-9b-3-bit",
        "flux2-klein-9b-4-bit",
        "flux2-klein-9b-6-bit",
        "flux2-klein-9b-8-bit",
        "flux2-klein-base-4b",
        "flux2-klein-base-4b-mlx-4bit",  # Pre-quantized (faster load)
        "flux2-klein-base-4b-mlx-8bit",  # Pre-quantized (faster load)
        "flux2-klein-base-4b-3-bit",
        "flux2-klein-base-4b-4-bit",
        "flux2-klein-base-4b-6-bit",
        "flux2-klein-base-4b-8-bit",
        "flux2-klein-base-9b",
        "flux2-klein-base-9b-3-bit",
        "flux2-klein-base-9b-4-bit",
        "flux2-klein-base-9b-6-bit",
        "flux2-klein-base-9b-8-bit",
        "flux2-dev",
        "flux2-dev-3-bit",
        "flux2-dev-4-bit",
        "flux2-dev-6-bit",
        "flux2-dev-8-bit",
    ]


def get_updated_models(include_flux2: bool = True) -> List[str]:
    """Combine official aliases with any folders under models/."""
    ordered = _flux2_ordered()
    ordered.append("seedvr2")
    predefined = [alias for alias in ordered if alias in MODELS]

    custom_entries: List[str] = []
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    for folder in models_dir.iterdir():
        if not folder.is_dir():
            continue
        alias = folder.name
        if alias in predefined:
            continue
        custom_entries.append(alias)
        if alias not in MODELS:
            MODELS[alias] = CustomModelConfig(
                model_name=str(folder),
                alias=alias,
                num_train_steps=1000,
                max_sequence_length=512,
                base_arch="schnell",
                local_dir=folder,
            )

    custom_entries.sort(key=str.lower)
    return predefined + custom_entries


def get_flux2_models() -> List[str]:
    """Return Flux2 Klein aliases (quantized variants included)."""
    return [alias for alias in _flux2_ordered() if alias in MODELS]


def save_quantized_model_gradio(model_name, quantize_bits):
    """
    Legacy shim retained so older layouts don't crash. The real quantization UI
    now lives in the exporter section.
    """
    warning = (
        "Quantization moved into the Export section. Refresh the page to load the new workflow."
    )
    print(warning)
    model_choices = get_model_choices()
    return (
        model_choices,
        model_choices,
        model_choices,
        model_choices,
        model_choices,
        model_name,
        warning,
    )


def register_local_model(alias: str, model_name: str, base_arch: str, target_dir: Path):
    MODELS[alias] = CustomModelConfig(
        model_name=model_name,
        alias=alias,
        num_train_steps=1000,
        max_sequence_length=512 if base_arch != "schnell" else 256,
        base_arch=base_arch,
        local_dir=target_dir,
    )
    return get_model_choices()


def resolve_local_path(alias: str) -> Optional[Path]:
    config = MODELS.get(alias)
    if config and config.local_dir and config.local_dir.exists():
        return config.local_dir
    candidate = Path("models") / alias
    if candidate.exists():
        if config:
            config.local_dir = candidate
        return candidate
    return None


def strip_quant_suffix(model_name: str) -> str:
    return re.sub(r"-(?:3|4|6|8)-bit$", "", model_name, flags=re.IGNORECASE)


def resolve_mflux_model_config(model_name: str, base_model: Optional[str] = None) -> MfluxModelConfig:
    resolved_name = strip_quant_suffix(model_name or "").strip()
    if not resolved_name:
        resolved_name = "schnell"

    base_model = normalize_base_model_choice(base_model)
    if base_model is None:
        custom = MODELS.get(resolved_name)
        if custom:
            base_model = custom.base_arch

    if resolved_name == "dev-krea":
        resolved_name = "krea-dev"

    if resolved_name.startswith(("flux2-", "klein-")):
        base_model = None

    try:
        return MfluxModelConfig.from_name(model_name=resolved_name, base_model=base_model)
    except Exception:
        method_map = {
            "dev": "dev",
            "schnell": "schnell",
            "krea-dev": "krea_dev",
            "flux2-klein": "flux2_klein_4b",
            "flux2-klein-4b": "flux2_klein_4b",
            "flux2-klein-9b": "flux2_klein_9b",
            "flux2-klein-base-4b": "flux2_klein_base_4b",
            "flux2-klein-base-9b": "flux2_klein_base_9b",
            "klein-4b": "flux2_klein_4b",
            "klein-9b": "flux2_klein_9b",
            "dev-kontext": "dev_kontext",
            "dev-fill": "dev_fill",
            "dev-redux": "dev_redux",
            "dev-depth": "dev_depth",
            "dev-controlnet-canny": "dev_controlnet_canny",
            "schnell-controlnet-canny": "schnell_controlnet_canny",
            "dev-controlnet-upscaler": "dev_controlnet_upscaler",
            "dev-fill-catvton": "dev_fill_catvton",
            "qwen-image": "qwen_image",
            "qwen-image-edit": "qwen_image_edit",
            "fibo": "fibo",
            "z-image": "z_image",
            "z-image-turbo": "z_image_turbo",
        }
        method = method_map.get(resolved_name)
        if method and hasattr(MfluxModelConfig, method):
            return getattr(MfluxModelConfig, method)()
        return MfluxModelConfig.from_name(model_name=resolved_name, base_model=base_model)


def download_and_save_model(
    hf_model_name,
    alias,
    num_train_steps,
    max_sequence_length,
    api_key,
    base_arch="schnell",
):
    """
    Download a model from Hugging Face and save it locally.
    """
    try:
        login_result = login_huggingface(api_key)
        if "Error" in login_result:
            return None, None, None, None, None, f"HF Login failed: {login_result}"

        model_dir = Path("models") / alias
        model_dir.mkdir(parents=True, exist_ok=True)

        stat = disk_usage(model_dir.parent)
        if stat.free < 8 * 1024**3:
            return None, None, None, None, None, "Error: Not enough free disk space to download model"

        snapshot_download(
            repo_id=hf_model_name,
            local_dir=str(model_dir),
            use_auth_token=api_key,
        )

        MODELS[alias] = CustomModelConfig(
            hf_model_name,
            alias,
            num_train_steps,
            max_sequence_length,
            base_arch,
            local_dir=model_dir,
        )

        model_choices = get_model_choices()
        print(f"Model {hf_model_name} successfully downloaded and saved as {alias}")
        return (
            model_choices,
            model_choices,
            model_choices,
            model_choices,
            model_choices,
            "Success",
        )

    except Exception as e:
        error_message = f"Error downloading model: {str(e)}"
        print(f"Error: {error_message}")
        return None, None, None, None, None, error_message


def normalize_base_model_choice(choice: Optional[str]) -> Optional[str]:
    """
    Normalize --base-model dropdown values from the UI so backend helpers can
    safely inject overrides. (Docs: gradiodocs/guides-interface-state/interface_state.md)
    """
    if choice is None:
        return None
    if isinstance(choice, str):
        normalized = choice.strip().lower()
        if normalized in {"", "none", "auto"}:
            return None
    return choice


def login_huggingface(api_key):
    """
    Login to Hugging Face with the given API key.
    """
    try:
        if not api_key:
            return "Error: API key is missing"

        os.environ["HUGGINGFACE_HUB_TOKEN"] = api_key
        api = HfApi(token=api_key)

        try:
            api.whoami()
            return "Successfully logged in to Hugging Face"
        except Exception as exc:
            return f"Error validating credentials: {str(exc)}"

    except Exception as exc:
        return f"Error logging in to Hugging Face: {str(exc)}"


def update_guidance_visibility(model):
    """
    Ensure guidance controls follow docs guidance for interactive components
    (gradiodocs/guides-key-component-concepts/gradio_components_the_key_concepts.md).
    """
    model_name = (model or "").lower()
    if model_name.startswith(("flux2-", "klein-")):
        if model_name.startswith("flux2-dev"):
            return gr.update(
                visible=True,
                label="Guidance Scale (FLUX.2-dev, experimental)",
                value=3.5,
                interactive=True,
            )
        if "-base-" in model_name:
            return gr.update(
                visible=True,
                label="Guidance Scale (FLUX.2 base models)",
                value=1.0,
                interactive=True,
            )
        return gr.update(
            visible=True,
            label="Guidance Scale (fixed at 1.0 for FLUX.2)",
            value=1.0,
            interactive=False,
        )
    is_dev = "dev" in model_name
    return gr.update(
        visible=True,
        label="Guidance Scale (required for dev models)" if is_dev else "Guidance Scale (optional)",
        interactive=True,
    )


def get_model_choices():
    models = get_updated_models()
    return gr.update(choices=models) if models else gr.update()
