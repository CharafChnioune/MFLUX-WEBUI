from __future__ import annotations

from typing import Any

try:
    from mflux.models.common.config.config import Config as _Config
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    from mflux.config.config import Config as _Config  # type: ignore

try:
    from mflux.models.common.config.model_config import ModelConfig as _ModelConfig
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    from mflux.config.model_config import ModelConfig as _ModelConfig  # type: ignore

try:
    from mflux.models.common.config.runtime_config import RuntimeConfig as _RuntimeConfig
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    try:
        from mflux.config.runtime_config import RuntimeConfig as _RuntimeConfig  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional feature
        class _RuntimeConfig:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ModuleNotFoundError(
                    "RuntimeConfig is unavailable in this MFLUX version. "
                    "Install a version that provides it or update training code."
                )


Config = _Config
ModelConfig = _ModelConfig
RuntimeConfig = _RuntimeConfig


def create_config(model_config, **kwargs: Any):
    """
    Create a Config instance across MFLUX versions.
    Newer MFLUX requires model_config; older releases ignore it.
    """
    try:
        return Config(model_config=model_config, **kwargs)
    except TypeError:
        return Config(**kwargs)
