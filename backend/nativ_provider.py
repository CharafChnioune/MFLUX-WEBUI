"""Small, local-only client for Nativ's OpenAI-compatible MLX server."""
from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen


def _base_url() -> str:
    return os.getenv("NATIV_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def _request(path: str) -> dict:
    request = Request(f"{_base_url()}{path}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=2) as response:  # nosec B310: local endpoint is configurable by the owner
        return json.loads(response.read().decode("utf-8"))


def status() -> dict:
    """Return a privacy-safe Nativ status without exposing credentials or paths."""
    try:
        health = _request("/health")
        models = _request("/v1/models")
        return {
            "provider": "nativ",
            "available": True,
            "base_url": _base_url(),
            "health": health,
            "models": [item.get("id") for item in models.get("data", []) if item.get("id")],
        }
    except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"provider": "nativ", "available": False, "base_url": _base_url(), "reason": str(exc)}
