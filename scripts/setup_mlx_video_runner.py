#!/usr/bin/env python3
"""Provision the pinned, isolated MLX video runtime and Wan 2.1 model.

This script deliberately keeps mlx-video and its dependencies out of the MFLUX
photo environment. Model source snapshots use Hugging Face's shared cache; only
the derived MLX checkpoint and isolated runtime live beside that cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ENGINE_REPOSITORY = "https://github.com/Blaizzy/mlx-video.git"
ENGINE_REVISION = "87db56a51758fefb748a359b90a5283bb8ba4837"
MODEL_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B"
MODEL_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
TOKENIZER_REPOSITORY = "google/umt5-xxl"
TOKENIZER_REVISION = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
TORCH_VERSION = "2.7.1"
MINIMUM_FREE_BYTES = 40 * 1024**3
MINIMUM_MEMORY_BYTES = 64 * 1024**3
MODEL_MANIFEST_FILENAME = "conversion-manifest.json"
CONVERTED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
)


class SetupError(RuntimeError):
    pass


def _hf_home() -> Path:
    configured = os.environ.get("HF_HOME")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".cache" / "huggingface"


def runtime_paths() -> dict[str, Path]:
    root = Path(
        os.environ.get("MFLUX_VIDEO_RUNTIME_ROOT", _hf_home() / "mlx-media" / "video-runner")
    ).expanduser().resolve()
    engine = root / "engines" / f"mlx-video-{ENGINE_REVISION[:12]}"
    return {
        "root": root,
        "engine": engine,
        "python": engine / ".venv" / "bin" / "python",
        "model": root / "models" / f"wan2.1-t2v-1.3b-bf16-{MODEL_REVISION[:12]}",
        "state": root / "state" / "wan-2.1-t2v-1.3b.json",
    }


def preflight() -> dict[str, object]:
    paths = runtime_paths()
    paths["root"].mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(paths["root"]).free
    try:
        memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        memory_bytes = 0
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "apple_silicon": apple_silicon,
        "free_bytes": free_bytes,
        "memory_bytes": memory_bytes,
        "disk_ok": free_bytes >= MINIMUM_FREE_BYTES,
        "memory_ok": memory_bytes >= MINIMUM_MEMORY_BYTES,
        "paths": {name: str(path) for name, path in paths.items()},
        "pins": {
            "engine_repository": ENGINE_REPOSITORY,
            "engine_revision": ENGINE_REVISION,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "tokenizer_repository": TOKENIZER_REPOSITORY,
            "tokenizer_revision": TOKENIZER_REVISION,
            "conversion_torch": TORCH_VERSION,
            "conversion_dtype": "bfloat16",
        },
    }


def _require_preflight(report: dict[str, object]) -> None:
    if not report["apple_silicon"]:
        raise SetupError("The isolated MLX video runner requires Apple silicon.")
    if not report["disk_ok"]:
        raise SetupError("At least 40 GiB of free space is required for source and converted weights.")
    if not report["memory_ok"]:
        raise SetupError("This initial tested profile requires a Mac with at least 64 GiB unified memory.")


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _find_uv() -> str:
    configured = os.environ.get("MFLUX_UV_BIN")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("uv")
    if found:
        return found
    raise SetupError("uv is required to reproduce the upstream frozen environment.")


def _provision_engine(paths: dict[str, Path]) -> None:
    engine = paths["engine"]
    if not engine.exists():
        engine.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=engine.parent, prefix="mlx-video-clone-") as temp:
            checkout = Path(temp) / "checkout"
            _run(["git", "clone", "--no-checkout", ENGINE_REPOSITORY, str(checkout)])
            _run(["git", "checkout", "--detach", ENGINE_REVISION], cwd=checkout)
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
            if actual != ENGINE_REVISION:
                raise SetupError("The mlx-video checkout did not resolve to the audited revision.")
            checkout.rename(engine)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
    if actual != ENGINE_REVISION:
        raise SetupError("The existing mlx-video checkout does not match the audited revision.")

    uv = _find_uv()
    _run([uv, "sync", "--frozen", "--no-dev"], cwd=engine)
    _run(
        [uv, "pip", "install", "--python", str(paths["python"]), f"torch=={TORCH_VERSION}"],
        cwd=engine,
    )


def _download_pinned_snapshots() -> tuple[Path, Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SetupError("huggingface_hub is required in the setup environment.") from exc

    model = Path(snapshot_download(repo_id=MODEL_REPOSITORY, revision=MODEL_REVISION)).resolve()
    tokenizer = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPOSITORY,
            revision=TOKENIZER_REVISION,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "spiece.model",
            ],
        )
    ).resolve()
    if model.name != MODEL_REVISION or tokenizer.name != TOKENIZER_REVISION:
        raise SetupError("A Hugging Face snapshot did not resolve to the audited revision.")
    return model, tokenizer


def _pin_tokenizer_ref(tokenizer_snapshot: Path) -> None:
    """Make the hard-coded upstream tokenizer lookup deterministic offline."""
    repo_cache = tokenizer_snapshot.parent.parent
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    # huggingface_hub reads refs verbatim; a trailing newline becomes part of
    # the revision and breaks offline resolution.
    (refs / "main").write_text(TOKENIZER_REVISION, encoding="utf-8")


def _link_local_tokenizer(paths: dict[str, Path], tokenizer_snapshot: Path) -> None:
    """Satisfy upstream's hard-coded relative tokenizer path without network."""
    google = paths["root"] / "google"
    google.mkdir(parents=True, exist_ok=True)
    link = google / "umt5-xxl"
    if link.is_symlink() and link.resolve() == tokenizer_snapshot:
        return
    if link.exists() or link.is_symlink():
        raise SetupError("Refusing to replace an existing local tokenizer path.")
    link.symlink_to(tokenizer_snapshot, target_is_directory=True)


def _convert_model(paths: dict[str, Path], source: Path) -> None:
    model = paths["model"]
    required = set(CONVERTED_MODEL_FILES)
    if model.is_dir() and required.issubset({path.name for path in model.iterdir()}):
        return
    model.parent.mkdir(parents=True, exist_ok=True)
    partial = model.with_name(f"{model.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    command = [
        str(paths["python"]),
        "-m",
        "mlx_video.models.wan_2.convert",
        "--checkpoint-dir",
        str(source),
        "--output-dir",
        str(partial),
        "--dtype",
        "bfloat16",
        "--model-version",
        "2.1",
    ]
    _run(command, cwd=paths["engine"])
    produced = {path.name for path in partial.iterdir()}
    missing = required - produced
    if missing:
        raise SetupError(f"Converted model is incomplete: missing {', '.join(sorted(missing))}.")
    if model.exists():
        raise SetupError("Refusing to replace an existing converted model directory.")
    partial.rename(model)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_model_manifest(paths: dict[str, Path]) -> None:
    """Bind the converted files to the exact audited source and engine pins."""
    model = paths["model"]
    files = {}
    for filename in CONVERTED_MODEL_FILES:
        artifact = model / filename
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise SetupError(f"Converted model is incomplete: missing {filename}.")
        files[filename] = {
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
        }
    payload = {
        "schema_version": 1,
        "purpose": "mlx-video-conversion-manifest",
        "engine_revision": ENGINE_REVISION,
        "source": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "conversion": {"model_version": "2.1", "dtype": "bfloat16"},
        "files": files,
    }
    destination = model / MODEL_MANIFEST_FILENAME
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def provision() -> dict[str, object]:
    report = preflight()
    _require_preflight(report)
    paths = runtime_paths()
    _provision_engine(paths)
    model_source, tokenizer = _download_pinned_snapshots()
    _pin_tokenizer_ref(tokenizer)
    _link_local_tokenizer(paths, tokenizer)
    _convert_model(paths, model_source)
    _write_model_manifest(paths)
    report["runtime"] = {
        "python_ready": paths["python"].is_file(),
        "model_ready": paths["model"].is_dir(),
        "generation_tested": paths["state"].is_file(),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "provision"), nargs="?", default="plan")
    args = parser.parse_args()
    try:
        report = provision() if args.action == "provision" else preflight()
    except (SetupError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
