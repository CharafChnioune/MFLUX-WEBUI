from __future__ import annotations

from typing import Any

try:
    from mflux.models.common.config.config import Config as _Config
    from mflux.models.common.config.model_config import ModelConfig as _ModelConfig
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    from mflux.config.config import Config as _Config  # type: ignore
    from mflux.config.model_config import ModelConfig as _ModelConfig  # type: ignore


Config = _Config
ModelConfig = _ModelConfig


def create_config(model_config, **kwargs: Any):
    """
    Create a Config instance across MFLUX versions.
    Newer MFLUX requires model_config; older releases ignore it.
    """
    try:
        return Config(model_config=model_config, **kwargs)
    except TypeError:
        return Config(**kwargs)
