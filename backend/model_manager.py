import os
from pathlib import Path
from shutil import disk_usage
from typing import Dict, List, Optional

import gradio as gr
from huggingface_hub import HfApi, HfFolder, snapshot_download

# Layout/state docs referenced across the UI:
# - gradiodocs/docs-blocks/blocks.md (consistent Blocks state)
# - gradiodocs/guides-controlling-layout/controlling_layout.md (Row/Column ordering)
# - gradiodocs/guides-interface-state/interface_state.md (shared state between tabs)

BASE_MODEL_CHOICES = ["schnell", "dev", "krea-dev"]
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
        ("schnell", "AITRADER/MFLUXUI.1-schnell", 256, "schnell"),
        ("dev", "AITRADER/MFLUXUI.1-dev", 512, "dev"),
        ("krea-dev", "black-forest-labs/FLUX.1-Krea-dev", 512, "krea-dev"),
        ("dev-krea", "black-forest-labs/FLUX.1-Krea-dev", 512, "krea-dev"),
    ]
    for alias, repo, seq_len, base_arch in official:
        MODELS[alias] = CustomModelConfig(repo, alias, 1000, seq_len, base_arch)

    for alias, repo, seq_len, base_arch in official:
        if alias == "dev-krea":
            continue  # share canonical entry with krea-dev
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


def get_updated_models() -> List[str]:
    """Combine official aliases with any folders under models/."""
    ordered = [
        "schnell",
        "schnell-3-bit",
        "schnell-4-bit",
        "schnell-6-bit",
        "schnell-8-bit",
        "dev",
        "dev-3-bit",
        "dev-4-bit",
        "dev-6-bit",
        "dev-8-bit",
        "krea-dev",
        "krea-dev-3-bit",
        "krea-dev-4-bit",
        "krea-dev-6-bit",
        "krea-dev-8-bit",
        "dev-krea",
    ]
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

        HfFolder.save_token(api_key)
        api = HfApi()

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
    is_dev = "dev" in model
    return gr.update(
        visible=True,
        label="Guidance Scale (required for dev models)" if is_dev else "Guidance Scale (optional)",
    )


def get_model_choices():
    models = get_updated_models()
    return gr.update(choices=models) if models else gr.update()
