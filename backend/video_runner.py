"""Isolated, local-only MLX video runner for one audited Wan capability.

The web application deliberately does not import ``mlx_video``.  A separately
configured Python interpreter owns that dependency graph and is invoked through
an argv-only subprocess.  This module contains the complete trust boundary:
strict request validation, readiness proof checks, process-group cancellation,
artifact containment, and provenance that never publishes server paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


VIDEO_SCHEMA_VERSION = 1
VIDEO_JOB_TYPE_ID = "video"
VIDEO_CAPABILITY_ID = "wan-2.1-t2v-1.3b"
VIDEO_OPERATION_ID = "text-to-video"

VIDEO_ENGINE_SOURCE = "Blaizzy/mlx-video"
VIDEO_ENGINE_REVISION = "87db56a51758fefb748a359b90a5283bb8ba4837"
VIDEO_ENGINE_LICENSE = "MIT"
VIDEO_MODEL_SOURCE = "Wan-AI/Wan2.1-T2V-1.3B"
VIDEO_MODEL_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
VIDEO_MODEL_LICENSE = "Apache-2.0"
VIDEO_TOKENIZER_SOURCE = "google/umt5-xxl"
VIDEO_TOKENIZER_REVISION = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"

VIDEO_RUNTIME_PYTHON_ENV = "MFLUX_VIDEO_RUNNER_PYTHON"
VIDEO_MODEL_DIR_ENV = "MFLUX_VIDEO_MODEL_DIR"
VIDEO_SMOKE_PROVENANCE_ENV = "MFLUX_VIDEO_SMOKE_PROVENANCE"
VIDEO_OUTPUT_ROOT_ENV = "MFLUX_VIDEO_OUTPUT_ROOT"
VIDEO_RUNTIME_ROOT_ENV = "MFLUX_VIDEO_RUNTIME_ROOT"

_ENTRYPOINT = "mlx_video.models.wan_2.generate"
_OUTPUT_DIRECTORY_NAME = "video-jobs"
_PROMPT_MAX_LENGTH = 2_000
_ALLOWED_TILING = {"auto", "default"}
_REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
)
_MODEL_MANIFEST_FILENAME = "conversion-manifest.json"
_PUBLIC_ARTIFACT_FILENAMES = {"video.mp4", "provenance.json", "request.json"}
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_DIFFUSION_PROGRESS = re.compile(r"Diffusion:.*?(\d+)\s*/\s*(\d+)")
_DENOISING_STEPS = re.compile(r"Denoising\s*\(\s*(\d+)\s+steps?\s*\)", re.I)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VideoValidationError(ValueError):
    """Raised before an invalid request or artifact path can touch the runner."""


class VideoRuntimeError(RuntimeError):
    """Raised when the isolated runner cannot produce a verified artifact."""


@dataclass
class _RunningProcess:
    process: Any
    cancelled: threading.Event


_RUNNING_PROCESSES: dict[str, _RunningProcess] = {}
_RUNNING_PROCESSES_LOCK = threading.Lock()


def get_video_capabilities() -> dict[str, Any]:
    """Return the single audited capability without publishing local paths."""
    runtime = get_video_runtime_status()
    ready = runtime["ready"]
    capability = {
        "id": VIDEO_CAPABILITY_ID,
        "label": "Wan 2.1 T2V 1.3B",
        "type": VIDEO_JOB_TYPE_ID,
        "operations": [VIDEO_OPERATION_ID],
        "availability": "ready" if ready else "setup-required",
        "availability_reason": (
            "The pinned isolated runner passed its local smoke test."
            if ready
            else "The pinned isolated runner, converted model, and smoke proof are required."
        ),
        "engine": {
            "source": VIDEO_ENGINE_SOURCE,
            "revision": VIDEO_ENGINE_REVISION,
            "license": VIDEO_ENGINE_LICENSE,
            "tested": runtime["engine"]["tested"],
        },
        "model": {
            "source": VIDEO_MODEL_SOURCE,
            "revision": VIDEO_MODEL_REVISION,
            "license": VIDEO_MODEL_LICENSE,
            "converted_local": True,
            "configured": runtime["model"]["configured"],
            "cached": runtime["model"]["cached"],
            "smoke_tested": runtime["model"]["smoke_tested"],
        },
        "parameters": {
            "prompt": {"min_length": 1, "max_length": _PROMPT_MAX_LENGTH},
            "width": {"fixed": 832},
            "height": {"fixed": 480},
            "num_frames": {"minimum": 5, "maximum": 81, "rule": "4n+1"},
            "fps": {"fixed": 16},
            "steps": {"minimum": 1, "maximum": 50, "default": 10},
            "scheduler": {"fixed": "unipc"},
            "tiling": {"allowed": ["auto", "default"], "default": "auto"},
            "seed": {"minimum": 0, "maximum": 2**32 - 1, "optional": True},
        },
        "output": {"container": "mp4", "audio": "none"},
        "isolation": "separate-subprocess",
        "concurrency": "serialized-with-media-queue",
        "cancel_mode": "process-group-with-stage-checks",
    }
    return {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "default_capability_id": VIDEO_CAPABILITY_ID,
        "capabilities": [capability],
    }


def get_video_runtime_status(
    active_media_job: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report honest readiness from local files plus a pinned successful smoke proof."""
    diagnostics = _runtime_diagnostics()
    state = "ready" if diagnostics["ready"] else "setup-required"
    active = _sanitize_active_job(active_media_job)
    if diagnostics["ready"] and active is not None:
        state = "busy"

    result: dict[str, Any] = {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "state": state,
        "ready": diagnostics["ready"],
        "apple_silicon": diagnostics["apple_silicon"],
        "isolation": "separate-subprocess",
        "concurrency": "serialized-with-media-queue",
        "cancel_mode": "process-group-with-stage-checks",
        "network_mode": "offline-hugging-face-cache",
        "engine": {
            "name": "mlx-video",
            "source": VIDEO_ENGINE_SOURCE,
            "revision": VIDEO_ENGINE_REVISION,
            "license": VIDEO_ENGINE_LICENSE,
            "configured": diagnostics["runner_configured"],
            "available": diagnostics["runner_available"],
            "tested": diagnostics["smoke_valid"],
        },
        "model": {
            "source": VIDEO_MODEL_SOURCE,
            "revision": VIDEO_MODEL_REVISION,
            "license": VIDEO_MODEL_LICENSE,
            "configured": diagnostics["model_configured"],
            "cached": diagnostics["model_cached"],
            "converted": diagnostics["model_complete"],
            "smoke_tested": diagnostics["smoke_valid"],
        },
        "reasons": diagnostics["reasons"],
    }
    if active is not None:
        result["active_media_job"] = active
    return result


def prepare_video_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a wire request and bind it to the ready server-side runtime."""
    public_request, seed_source = _normalize_video_payload(payload)
    diagnostics = _runtime_diagnostics()
    if not diagnostics["ready"]:
        reason_codes = ", ".join(item["code"] for item in diagnostics["reasons"])
        raise VideoValidationError(
            f"Video runtime setup is required before submission ({reason_codes})."
        )

    server = {
        "runtime_python": str(diagnostics["runtime_python"]),
        "model_dir": str(diagnostics["model_dir"]),
        "smoke_provenance": str(diagnostics["smoke_provenance"]),
        "hf_home": str(_configured_hf_home()),
        "hf_hub_cache": str(_configured_hf_hub_cache()),
        "output_root": str(_configured_output_base()),
        "runtime_root": str(_configured_runtime_root()),
        "engine_revision": VIDEO_ENGINE_REVISION,
        "model_revision": VIDEO_MODEL_REVISION,
    }
    request_hash = _json_hash(public_request)
    plan = {
        **public_request,
        "request_hash": request_hash,
        "seed_source": seed_source,
        "_server": server,
    }
    plan["_plan_integrity"] = _plan_integrity(plan)
    return plan


def public_video_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reviewable request portion of a server-created plan."""
    _validate_prepared_plan(plan, require_runtime=False)
    return {
        key: value
        for key, value in plan.items()
        if key not in {"_server", "_plan_integrity", "seed_source"}
    }


def record_video_smoke_success(
    artifact_path: str | os.PathLike[str],
    *,
    generation_seconds: float,
    raw_num_frames: int | None = None,
) -> dict[str, Any]:
    """Record a successful setup-only smoke after exact local media validation."""
    if not isinstance(generation_seconds, (int, float)) or generation_seconds <= 0:
        raise VideoValidationError("generation_seconds must be positive.")
    runtime_root = _configured_runtime_root()
    artifact = Path(artifact_path).expanduser().resolve(strict=True)
    try:
        artifact_relative = artifact.relative_to(runtime_root).as_posix()
    except ValueError as exc:
        raise VideoValidationError("Smoke artifacts must remain inside the video runtime root.") from exc

    runtime_python = (
        _path_from_environment(VIDEO_RUNTIME_PYTHON_ENV, preserve_symlink=True)
        or _default_runtime_python()
    )
    model_dir = _path_from_environment(VIDEO_MODEL_DIR_ENV) or _default_model_dir()
    engine_valid, engine_reason = _validate_engine_checkout(runtime_python)
    model_complete, model_reason = _validate_converted_model(model_dir, verify_hashes=True)
    tokenizer_valid, tokenizer_reason = _validate_tokenizer(runtime_root)
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise VideoValidationError("The isolated video runtime is unavailable.")
    if not engine_valid:
        raise VideoValidationError(engine_reason)
    if not model_complete:
        raise VideoValidationError(model_reason)
    if not tokenizer_valid:
        raise VideoValidationError(tokenizer_reason)
    attestation = _current_runtime_attestation(runtime_python, model_dir, runtime_root)

    plan = {
        "output": {
            "width": 832,
            "height": 480,
            "num_frames": 5,
            "fps": 16,
            "container": "mp4",
        },
        "_server": {
            "runtime_python": str(runtime_python),
            "runtime_root": str(runtime_root),
            "hf_home": str(_configured_hf_home()),
            "hf_hub_cache": str(_configured_hf_hub_cache()),
        },
    }
    verification = _inspect_mp4(artifact, plan, require_exact_frames=True)
    if raw_num_frames is not None:
        if (
            not isinstance(raw_num_frames, int)
            or isinstance(raw_num_frames, bool)
            or raw_num_frames < verification["num_frames"]
        ):
            raise VideoValidationError("raw_num_frames is invalid.")
        verification["raw_num_frames"] = raw_num_frames
        verification["tail_frames_removed"] = raw_num_frames - verification["num_frames"]
        verification["raw_frame_count"] = raw_num_frames
        verification["final_frame_count"] = verification["num_frames"]

    proof = {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "purpose": "smoke-test",
        "status": "completed",
        "capability_id": VIDEO_CAPABILITY_ID,
        "engine": {
            "source": VIDEO_ENGINE_SOURCE,
            "revision": VIDEO_ENGINE_REVISION,
            "license": VIDEO_ENGINE_LICENSE,
        },
        "model": {
            "source": VIDEO_MODEL_SOURCE,
            "revision": VIDEO_MODEL_REVISION,
            "license": VIDEO_MODEL_LICENSE,
        },
        "attestation": attestation,
        "runtime": {
            "isolation": "separate-subprocess",
            "network_mode": "hugging-face-offline",
        },
        "profile": {
            "width": 832,
            "height": 480,
            "num_frames": 5,
            "fps": 16,
            "steps": 10,
            "scheduler": "unipc",
            "seed": 42,
        },
        "generation_seconds": round(float(generation_seconds), 3),
        "completed_at": _now_iso(),
        "output": {
            "artifact": artifact_relative,
            "sha256": _sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "verification": verification,
        },
    }
    state_path = _path_from_environment(VIDEO_SMOKE_PROVENANCE_ENV) or _default_smoke_provenance()
    _atomic_write_json(state_path, proof)
    valid, reason = _validate_smoke_provenance(
        state_path,
        expected_attestation=attestation,
        runtime_root=runtime_root,
    )
    if not valid:
        raise VideoRuntimeError(f"The written smoke proof failed validation: {reason}")
    return proof


def run_video_job(
    plan: Mapping[str, Any],
    *,
    job_id: str,
    progress_callback: Callable[[str, Any], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one prepared request in the isolated environment and verify its MP4."""
    _validate_job_id(job_id)
    _validate_prepared_plan(plan, require_runtime=True)
    progress = progress_callback or (lambda _event, _data=None: None)
    is_cancelled = cancel_check or (lambda: False)
    _emit_progress(progress, "validating", 2.0)

    server = plan["_server"]
    output_root = Path(server["output_root"]) / _OUTPUT_DIRECTORY_NAME
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_root = output_root.resolve(strict=True)
    job_dir = output_root / job_id
    try:
        job_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise VideoValidationError(
            "This video job output already exists; existing artifacts will not be overwritten."
        ) from exc

    work_dir = job_dir / ".work"
    work_dir.mkdir(mode=0o700)
    working_video = work_dir / "generated.raw.mp4"
    normalized_video = work_dir / "generated.mp4"
    final_video = job_dir / "video.mp4"
    request_artifact = f"{_OUTPUT_DIRECTORY_NAME}/{job_id}/request.json"
    provenance_artifact = f"{_OUTPUT_DIRECTORY_NAME}/{job_id}/provenance.json"
    video_artifact = f"{_OUTPUT_DIRECTORY_NAME}/{job_id}/video.mp4"

    public_request = public_video_plan(plan)
    _atomic_write_json(job_dir / "request.json", public_request)
    started_at = _now_iso()
    provenance: dict[str, Any] = {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "purpose": "video-generation",
        "status": "running",
        "job_id": job_id,
        "capability_id": VIDEO_CAPABILITY_ID,
        "request_hash": plan["request_hash"],
        "request": public_request,
        "engine": {
            "source": VIDEO_ENGINE_SOURCE,
            "revision": VIDEO_ENGINE_REVISION,
            "license": VIDEO_ENGINE_LICENSE,
            "entrypoint": _ENTRYPOINT,
        },
        "model": {
            "source": VIDEO_MODEL_SOURCE,
            "revision": VIDEO_MODEL_REVISION,
            "license": VIDEO_MODEL_LICENSE,
            "converted_local": True,
        },
        "runtime": {
            "isolation": "separate-subprocess",
            "network_mode": "hugging-face-offline",
            "scheduler": "unipc",
        },
        "started_at": started_at,
        "artifacts": {
            "request": request_artifact,
            "provenance": provenance_artifact,
        },
        "artifact_urls": {
            "request": f"/api/v1/video/artifacts/{job_id}/request.json",
            "provenance": f"/api/v1/video/artifacts/{job_id}/provenance.json",
        },
        "runner_log": [],
    }
    _atomic_write_json(job_dir / "provenance.json", provenance)

    process = None
    running = None
    try:
        if is_cancelled():
            return _finish_cancelled(
                job_dir, work_dir, provenance, request_artifact, provenance_artifact
            )

        argv = _build_runner_argv(plan, working_video)
        environment = _offline_environment(plan)
        process = subprocess.Popen(
            argv,
            cwd=server["runtime_root"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        running = _RunningProcess(process=process, cancelled=threading.Event())
        with _RUNNING_PROCESSES_LOCK:
            if job_id in _RUNNING_PROCESSES:
                _terminate_process(process)
                raise VideoRuntimeError("A video process is already registered for this job.")
            _RUNNING_PROCESSES[job_id] = running

        return_code, log_lines = _observe_process(
            process,
            plan=plan,
            progress_callback=progress,
            cancel_check=is_cancelled,
            cancel_event=running.cancelled,
        )
        provenance["runner_log"] = log_lines

        if running.cancelled.is_set() or is_cancelled():
            return _finish_cancelled(
                job_dir,
                work_dir,
                provenance,
                request_artifact,
                provenance_artifact,
            )
        if return_code != 0:
            raise VideoRuntimeError(
                f"The isolated video runner exited unsuccessfully (code {return_code})."
            )
        if not working_video.is_file() or working_video.stat().st_size <= 0:
            raise VideoRuntimeError("The isolated runner did not produce an MP4 artifact.")

        _emit_progress(progress, "muxing", 95.0)
        verification = _normalize_and_verify_mp4(working_video, normalized_video, plan)
        if running.cancelled.is_set() or is_cancelled():
            return _finish_cancelled(
                job_dir,
                work_dir,
                provenance,
                request_artifact,
                provenance_artifact,
            )
        os.replace(normalized_video, final_video)
        _cleanup_work_directory(work_dir)
        if running.cancelled.is_set() or is_cancelled():
            return _finish_cancelled(
                job_dir,
                work_dir,
                provenance,
                request_artifact,
                provenance_artifact,
            )
        output_hash = _sha256_file(final_video)
        completed_at = _now_iso()
        provenance.update(
            {
                "status": "completed",
                "completed_at": completed_at,
                "output": {
                    "artifact": video_artifact,
                    "sha256": output_hash,
                    "size_bytes": final_video.stat().st_size,
                    "verification": verification,
                },
            }
        )
        provenance["artifacts"]["video"] = video_artifact
        provenance["artifact_urls"]["video"] = (
            f"/api/v1/video/artifacts/{job_id}/video.mp4"
        )
        _atomic_write_json(job_dir / "provenance.json", provenance)
        if running.cancelled.is_set() or is_cancelled():
            return _finish_cancelled(
                job_dir,
                work_dir,
                provenance,
                request_artifact,
                provenance_artifact,
            )
        _emit_progress(progress, "completed", 100.0)
        return {
            "schema_version": VIDEO_SCHEMA_VERSION,
            "status": "completed",
            "job_id": job_id,
            "capability_id": VIDEO_CAPABILITY_ID,
            "request_hash": plan["request_hash"],
            "artifacts": {
                "video": video_artifact,
                "provenance": provenance_artifact,
                "request": request_artifact,
            },
            "artifact_urls": {
                "video": f"/api/v1/video/artifacts/{job_id}/video.mp4",
                "provenance": f"/api/v1/video/artifacts/{job_id}/provenance.json",
                "request": f"/api/v1/video/artifacts/{job_id}/request.json",
            },
            "output": {
                "container": "mp4",
                "sha256": output_hash,
                "size_bytes": final_video.stat().st_size,
                "verification": verification,
            },
        }
    except Exception as exc:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        _remove_partial_video(working_video)
        _remove_partial_video(normalized_video)
        _remove_partial_video(final_video)
        _cleanup_work_directory(work_dir)
        provenance.update(
            {
                "status": "failed",
                "completed_at": _now_iso(),
                "failure": {"type": type(exc).__name__, "message": _safe_error_message(exc)},
            }
        )
        _atomic_write_json(job_dir / "provenance.json", provenance)
        raise
    finally:
        with _RUNNING_PROCESSES_LOCK:
            current = _RUNNING_PROCESSES.get(job_id)
            if running is not None and current is running:
                _RUNNING_PROCESSES.pop(job_id, None)


def request_video_cancel(job_id: str) -> bool:
    """Request cancellation of an active isolated subprocess."""
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        return False
    with _RUNNING_PROCESSES_LOCK:
        running = _RUNNING_PROCESSES.get(job_id)
    if running is None:
        return False
    running.cancelled.set()
    _terminate_process(running.process)
    return True


def discard_video_job_output(job_id: str) -> dict[str, Any]:
    """Remove a just-published video when cancellation won the final state race."""
    _validate_job_id(job_id)
    output_root = (_configured_output_base() / _OUTPUT_DIRECTORY_NAME).resolve(strict=True)
    job_dir = (output_root / job_id).resolve(strict=True)
    try:
        job_dir.relative_to(output_root)
    except ValueError as exc:
        raise VideoValidationError("Video job path escapes the output root.") from exc
    video = job_dir / "video.mp4"
    _remove_partial_video(video)
    _cleanup_work_directory(job_dir / ".work")
    provenance_path = job_dir / "provenance.json"
    provenance: dict[str, Any] = {}
    try:
        if provenance_path.stat().st_size <= 1_000_000:
            loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                provenance = loaded
    except (OSError, UnicodeError, json.JSONDecodeError):
        provenance = {}
    provenance.update(
        {
            "schema_version": VIDEO_SCHEMA_VERSION,
            "status": "cancelled",
            "job_id": job_id,
            "capability_id": VIDEO_CAPABILITY_ID,
            "completed_at": _now_iso(),
        }
    )
    provenance.pop("output", None)
    artifacts = provenance.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("video", None)
    artifact_urls = provenance.get("artifact_urls")
    if isinstance(artifact_urls, dict):
        artifact_urls.pop("video", None)
    _atomic_write_json(provenance_path, provenance)
    return {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "status": "cancelled",
        "job_id": job_id,
        "capability_id": VIDEO_CAPABILITY_ID,
        "artifact_urls": {
            "provenance": f"/api/v1/video/artifacts/{job_id}/provenance.json",
            "request": f"/api/v1/video/artifacts/{job_id}/request.json",
        },
    }


def resolve_video_artifact(job_id: str, filename: str) -> Path:
    """Resolve one allowlisted artifact while containing symlinks and traversal."""
    _validate_job_id(job_id)
    if filename not in _PUBLIC_ARTIFACT_FILENAMES:
        raise VideoValidationError("Video artifact name is invalid.")
    output_root = (_configured_output_base() / _OUTPUT_DIRECTORY_NAME).resolve(strict=True)
    job_dir = (output_root / job_id).resolve(strict=True)
    try:
        job_dir.relative_to(output_root)
    except ValueError as exc:
        raise VideoValidationError("Video artifact path escapes the output root.") from exc
    candidate = (job_dir / filename).resolve(strict=True)
    try:
        candidate.relative_to(job_dir)
    except ValueError as exc:
        raise VideoValidationError("Video artifact path escapes its job directory.") from exc
    if not candidate.is_file():
        raise VideoValidationError("Video artifact does not exist.")
    return candidate


def _normalize_video_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise VideoValidationError("Video request must be an object.")
    _require_exact_keys(
        payload,
        required={
            "schema_version",
            "type",
            "operation",
            "capability_id",
            "prompt",
            "output",
            "sampling",
        },
        optional=set(),
        label="video request",
    )
    if payload["schema_version"] != VIDEO_SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise VideoValidationError("schema_version must be 1.")
    if payload["type"] != VIDEO_JOB_TYPE_ID:
        raise VideoValidationError("type must be video.")
    if payload["operation"] != VIDEO_OPERATION_ID:
        raise VideoValidationError("Only text-to-video is supported.")
    if payload["capability_id"] != VIDEO_CAPABILITY_ID:
        raise VideoValidationError("The requested video capability is unsupported.")

    prompt = payload["prompt"]
    if not isinstance(prompt, str):
        raise VideoValidationError("prompt must be text.")
    prompt = prompt.strip()
    if not prompt:
        raise VideoValidationError("prompt must not be empty.")
    if len(prompt) > _PROMPT_MAX_LENGTH:
        raise VideoValidationError(f"prompt must be at most {_PROMPT_MAX_LENGTH} characters.")
    if "\x00" in prompt:
        raise VideoValidationError("prompt contains an unsupported null character.")

    output = payload["output"]
    if not isinstance(output, Mapping):
        raise VideoValidationError("output must be an object.")
    _require_exact_keys(
        output,
        required={"width", "height", "num_frames", "fps", "container"},
        optional=set(),
        label="output",
    )
    if not _is_exact_int(output["width"], 832):
        raise VideoValidationError("width must be 832.")
    if not _is_exact_int(output["height"], 480):
        raise VideoValidationError("height must be 480.")
    frames = _bounded_int(output["num_frames"], "num_frames", 5, 81)
    if (frames - 1) % 4 != 0:
        raise VideoValidationError("num_frames must follow the 4n+1 rule.")
    if not _is_exact_int(output["fps"], 16):
        raise VideoValidationError("fps must be 16.")
    if output["container"] != "mp4":
        raise VideoValidationError("container must be mp4.")

    sampling = payload["sampling"]
    if not isinstance(sampling, Mapping):
        raise VideoValidationError("sampling must be an object.")
    _require_exact_keys(
        sampling,
        required={"steps", "tiling"},
        optional={"seed", "scheduler", "wan_scheduler"},
        label="sampling",
    )
    if "scheduler" in sampling and "wan_scheduler" in sampling:
        raise VideoValidationError("Provide scheduler only once.")
    scheduler = sampling.get("scheduler", sampling.get("wan_scheduler", "unipc"))
    if scheduler != "unipc":
        raise VideoValidationError("scheduler must be unipc.")
    tiling = sampling["tiling"]
    if tiling not in _ALLOWED_TILING:
        raise VideoValidationError("tiling must be auto or default.")
    steps = _bounded_int(sampling["steps"], "steps", 1, 50)
    if "seed" in sampling:
        seed = _bounded_int(sampling["seed"], "seed", 0, 2**32 - 1)
        seed_source = "request"
    else:
        seed = secrets.randbelow(2**32)
        seed_source = "server-random"

    return (
        {
            "schema_version": VIDEO_SCHEMA_VERSION,
            "type": VIDEO_JOB_TYPE_ID,
            "operation": VIDEO_OPERATION_ID,
            "capability_id": VIDEO_CAPABILITY_ID,
            "prompt": prompt,
            "output": {
                "width": 832,
                "height": 480,
                "num_frames": frames,
                "fps": 16,
                "container": "mp4",
            },
            "sampling": {
                "steps": steps,
                "scheduler": "unipc",
                "tiling": tiling,
                "seed": seed,
            },
        },
        seed_source,
    )


def _validate_prepared_plan(plan: Mapping[str, Any], *, require_runtime: bool) -> None:
    if not isinstance(plan, Mapping):
        raise VideoValidationError("Prepared video plan is invalid.")
    required = {
        "schema_version",
        "type",
        "operation",
        "capability_id",
        "prompt",
        "output",
        "sampling",
        "request_hash",
        "seed_source",
        "_server",
        "_plan_integrity",
    }
    if set(plan) != required:
        raise VideoValidationError("Prepared video plan is incomplete or contains extra fields.")
    normalized, _ = _normalize_video_payload(
        {
            key: plan[key]
            for key in (
                "schema_version",
                "type",
                "operation",
                "capability_id",
                "prompt",
                "output",
                "sampling",
            )
        }
    )
    public = {
        key: plan[key]
        for key in (
            "schema_version",
            "type",
            "operation",
            "capability_id",
            "prompt",
            "output",
            "sampling",
        )
    }
    if normalized != public or plan["request_hash"] != _json_hash(public):
        raise VideoValidationError("Prepared video plan has been modified.")
    if plan["seed_source"] not in {"request", "server-random"}:
        raise VideoValidationError("Prepared video plan seed source is invalid.")
    server = plan["_server"]
    expected_server_keys = {
        "runtime_python",
        "model_dir",
        "smoke_provenance",
        "hf_home",
        "hf_hub_cache",
        "output_root",
        "runtime_root",
        "engine_revision",
        "model_revision",
    }
    if not isinstance(server, Mapping) or set(server) != expected_server_keys:
        raise VideoValidationError("Prepared video server configuration is invalid.")
    if server["engine_revision"] != VIDEO_ENGINE_REVISION:
        raise VideoValidationError("Prepared video engine pin is invalid.")
    if server["model_revision"] != VIDEO_MODEL_REVISION:
        raise VideoValidationError("Prepared video model pin is invalid.")
    for key in (
        "runtime_python",
        "model_dir",
        "smoke_provenance",
        "hf_home",
        "hf_hub_cache",
        "output_root",
        "runtime_root",
    ):
        value = server[key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise VideoValidationError("Prepared video server paths are invalid.")
    if plan["_plan_integrity"] != _plan_integrity(plan):
        raise VideoValidationError("Prepared video plan integrity check failed.")
    if require_runtime:
        current_paths = {
            "runtime_python": str(
                _path_from_environment(VIDEO_RUNTIME_PYTHON_ENV, preserve_symlink=True)
                or _default_runtime_python()
            ),
            "model_dir": str(
                _path_from_environment(VIDEO_MODEL_DIR_ENV) or _default_model_dir()
            ),
            "smoke_provenance": str(
                _path_from_environment(VIDEO_SMOKE_PROVENANCE_ENV)
                or _default_smoke_provenance()
            ),
            "hf_home": str(_configured_hf_home()),
            "hf_hub_cache": str(_configured_hf_hub_cache()),
            "output_root": str(_configured_output_base()),
            "runtime_root": str(_configured_runtime_root()),
        }
        if any(server[key] != value for key, value in current_paths.items()):
            raise VideoValidationError(
                "The configured video runtime changed while this job was queued."
            )
        runtime = _runtime_diagnostics(
            runtime_python=Path(server["runtime_python"]),
            model_dir=Path(server["model_dir"]),
            smoke_provenance=Path(server["smoke_provenance"]),
            runtime_root=Path(server["runtime_root"]),
        )
        if not runtime["ready"]:
            raise VideoValidationError("The prepared video runtime is no longer ready.")


def _runtime_diagnostics(
    *,
    runtime_python: Path | None = None,
    model_dir: Path | None = None,
    smoke_provenance: Path | None = None,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    reasons: list[dict[str, str]] = []
    apple_silicon = _is_apple_silicon()
    if not apple_silicon:
        reasons.append(
            {
                "code": "unsupported-platform",
                "message": "The audited runner requires Apple silicon on macOS.",
            }
        )

    if runtime_python is None:
        runtime_python = (
            _path_from_environment(VIDEO_RUNTIME_PYTHON_ENV, preserve_symlink=True)
            or _default_runtime_python()
        )
    runtime_configured = runtime_python is not None
    runner_available = bool(
        runtime_python
        and runtime_python.is_file()
        and os.access(runtime_python, os.X_OK)
    )
    if not runtime_configured:
        reasons.append(
            {
                "code": "runner-python-not-configured",
                "message": "The isolated video Python interpreter is not configured.",
            }
        )
    elif runtime_python is None or not runner_available:
        reasons.append(
            {
                "code": "runner-python-unavailable",
                "message": "The configured isolated video Python interpreter is unavailable.",
            }
        )
    engine_valid = False
    if runner_available and runtime_python is not None:
        engine_valid, engine_reason = _validate_engine_checkout(runtime_python)
        if not engine_valid:
            reasons.append(
                {
                    "code": "engine-pin-invalid",
                    "message": engine_reason,
                }
            )

    if model_dir is None:
        model_dir = _path_from_environment(VIDEO_MODEL_DIR_ENV) or _default_model_dir()
    model_configured = model_dir is not None
    model_cached = bool(model_dir and model_dir.is_dir())
    model_complete = False
    if not model_configured:
        reasons.append(
            {
                "code": "model-not-configured",
                "message": "The converted Wan model directory is not configured.",
            }
        )
    elif not model_cached:
        reasons.append(
            {
                "code": "model-unavailable",
                "message": "The configured converted Wan model directory is unavailable.",
            }
        )
    else:
        model_complete, model_reason = _validate_converted_model(model_dir)
        if not model_complete:
            reasons.append(
                {
                    "code": "model-incomplete",
                    "message": model_reason,
                }
            )

    if runtime_root is None:
        runtime_root = _configured_runtime_root()
    tokenizer_available, tokenizer_reason = _validate_tokenizer(runtime_root)
    if not tokenizer_available:
        reasons.append(
            {
                "code": "tokenizer-unavailable",
                "message": tokenizer_reason,
            }
        )

    if smoke_provenance is None:
        configured_smoke = _path_from_environment(VIDEO_SMOKE_PROVENANCE_ENV)
        smoke_provenance = configured_smoke or _default_smoke_provenance()
    smoke_valid = False
    if smoke_provenance is None or not smoke_provenance.is_file():
        reasons.append(
            {
                "code": "smoke-provenance-missing",
                "message": "A successful pinned local smoke-test proof is required.",
            }
        )
    elif (
        engine_valid
        and model_complete
        and tokenizer_available
        and runtime_python is not None
        and model_dir is not None
    ):
        attestation = _current_runtime_attestation(runtime_python, model_dir, runtime_root)
        smoke_valid, smoke_reason = _validate_smoke_provenance(
            smoke_provenance,
            expected_attestation=attestation,
            runtime_root=runtime_root,
        )
        if not smoke_valid:
            reasons.append(
                {
                    "code": "smoke-provenance-invalid",
                    "message": smoke_reason,
                }
            )
    else:
        reasons.append(
            {
                "code": "smoke-provenance-invalid",
                "message": "The smoke proof cannot match an invalid runtime.",
            }
        )

    ready = bool(
        apple_silicon
        and runner_available
        and engine_valid
        and model_complete
        and smoke_valid
        and tokenizer_available
        and not reasons
    )
    return {
        "ready": ready,
        "apple_silicon": apple_silicon,
        "runner_configured": runtime_configured,
        "runner_available": runner_available,
        "engine_valid": engine_valid,
        "model_configured": model_configured,
        "model_cached": model_cached,
        "model_complete": model_complete,
        "smoke_valid": smoke_valid,
        "tokenizer_available": tokenizer_available,
        "runtime_python": runtime_python,
        "model_dir": model_dir,
        "smoke_provenance": smoke_provenance,
        "reasons": reasons,
    }


def _validate_engine_checkout(runtime_python: Path) -> tuple[bool, str]:
    """Require the isolated interpreter to belong to the clean audited checkout."""
    try:
        engine_root = runtime_python.parents[2]
        if engine_root.name != f"mlx-video-{VIDEO_ENGINE_REVISION[:12]}":
            return False, "The isolated engine directory does not match the audited revision."
        completed = subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            cwd=engine_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        if completed.stdout.strip() != VIDEO_ENGINE_REVISION:
            return False, "The isolated engine revision does not match the audited pin."
        status = subprocess.run(
            ["/usr/bin/git", "status", "--porcelain", "--untracked-files=no"],
            cwd=engine_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        if status.stdout.strip():
            return False, "The isolated engine has modified tracked files."
    except (OSError, IndexError, subprocess.SubprocessError):
        return False, "The isolated engine checkout cannot be verified."
    return True, ""


def _validate_tokenizer(runtime_root: Path) -> tuple[bool, str]:
    """Bind the hard-coded upstream tokenizer name to the exact cached snapshot."""
    tokenizer = runtime_root / "google" / "umt5-xxl"
    try:
        if not tokenizer.is_dir():
            return False, "The pinned local UMT5 tokenizer is unavailable."
        resolved = tokenizer.resolve(strict=True)
        if resolved.name != VIDEO_TOKENIZER_REVISION:
            return False, "The local UMT5 tokenizer revision does not match the audited pin."
        expected_fragment = (
            "models--google--umt5-xxl",
            "snapshots",
            VIDEO_TOKENIZER_REVISION,
        )
        parts = resolved.parts
        if not any(
            tuple(parts[index : index + 3]) == expected_fragment
            for index in range(max(0, len(parts) - 2))
        ):
            return False, "The local UMT5 tokenizer is outside the pinned shared cache snapshot."
        for filename in ("config.json", "tokenizer_config.json"):
            artifact = resolved / filename
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                return False, "The pinned local UMT5 tokenizer is incomplete."
    except OSError:
        return False, "The pinned local UMT5 tokenizer cannot be verified."
    return True, ""


def _validate_converted_model(
    model_dir: Path,
    *,
    verify_hashes: bool = False,
) -> tuple[bool, str]:
    try:
        for filename in _REQUIRED_MODEL_FILES:
            path = model_dir / filename
            if not path.is_file() or path.stat().st_size <= 0:
                return False, "The converted Wan model is missing required local files."
        config_path = model_dir / "config.json"
        if config_path.stat().st_size > 1_000_000:
            return False, "The converted Wan model configuration is invalid."
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return False, "The converted Wan model configuration is invalid."
        if config.get("dim") != 1536 or config.get("num_layers") != 30:
            return False, "The converted model is not the audited Wan 2.1 1.3B checkpoint."
        if config.get("model_type", "t2v") != "t2v":
            return False, "The converted model is not text-to-video."
        if config.get("sample_fps", 16) != 16:
            return False, "The converted model does not use the audited 16 fps configuration."
        manifest_path = model_dir / _MODEL_MANIFEST_FILENAME
        if not manifest_path.is_file() or manifest_path.stat().st_size > 1_000_000:
            return False, "The converted model integrity manifest is missing."
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False, "The converted model integrity manifest is invalid."
        if (
            manifest.get("schema_version") != 1
            or manifest.get("purpose") != "mlx-video-conversion-manifest"
            or manifest.get("engine_revision") != VIDEO_ENGINE_REVISION
            or manifest.get("source")
            != {"repository": VIDEO_MODEL_SOURCE, "revision": VIDEO_MODEL_REVISION}
            or manifest.get("conversion")
            != {"model_version": "2.1", "dtype": "bfloat16"}
        ):
            return False, "The converted model integrity manifest does not match the audited pins."
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(_REQUIRED_MODEL_FILES):
            return False, "The converted model integrity manifest is incomplete."
        for filename in _REQUIRED_MODEL_FILES:
            record = files.get(filename)
            artifact = model_dir / filename
            if not isinstance(record, dict):
                return False, "The converted model integrity manifest is invalid."
            expected_size = record.get("size_bytes")
            expected_hash = record.get("sha256")
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size <= 0
                or artifact.stat().st_size != expected_size
                or not _SHA256.fullmatch(str(expected_hash or ""))
            ):
                return False, "A converted model artifact does not match its integrity manifest."
            if verify_hashes and _sha256_file(artifact) != expected_hash:
                return False, "A converted model artifact hash does not match its integrity manifest."
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, "The converted Wan model configuration cannot be verified."
    return True, ""


def _current_runtime_attestation(
    runtime_python: Path,
    model_dir: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    manifest = model_dir / _MODEL_MANIFEST_FILENAME
    tokenizer = (runtime_root / "google" / "umt5-xxl").resolve(strict=True)
    file_state = {}
    for filename in (*_REQUIRED_MODEL_FILES, _MODEL_MANIFEST_FILENAME):
        stat = (model_dir / filename).stat()
        file_state[filename] = {
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
    return {
        "engine_revision": VIDEO_ENGINE_REVISION,
        "engine_checkout": runtime_python.parents[2].name,
        "model_manifest_sha256": _sha256_file(manifest),
        "model_files": file_state,
        "tokenizer": {
            "source": VIDEO_TOKENIZER_SOURCE,
            "revision": tokenizer.name,
        },
    }


def _validate_smoke_provenance(
    path: Path,
    *,
    expected_attestation: Mapping[str, Any] | None = None,
    runtime_root: Path | None = None,
) -> tuple[bool, str]:
    try:
        if path.stat().st_size > 1_000_000:
            return False, "The smoke-test proof is invalid."
        proof = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, "The smoke-test proof cannot be read."
    if not isinstance(proof, dict):
        return False, "The smoke-test proof is invalid."
    if proof.get("schema_version") != VIDEO_SCHEMA_VERSION:
        return False, "The smoke-test proof version is unsupported."
    if proof.get("purpose") != "smoke-test" or proof.get("status") != "completed":
        return False, "The local smoke test has not completed successfully."
    if proof.get("capability_id") != VIDEO_CAPABILITY_ID:
        return False, "The smoke-test capability does not match."
    if expected_attestation is not None and proof.get("attestation") != expected_attestation:
        return False, "The smoke-test proof does not match the current pinned runtime."
    engine = proof.get("engine")
    model = proof.get("model")
    if not isinstance(engine, dict) or not isinstance(model, dict):
        return False, "The smoke-test pins are missing."
    if (
        engine.get("source") != VIDEO_ENGINE_SOURCE
        or engine.get("revision") != VIDEO_ENGINE_REVISION
        or engine.get("license") != VIDEO_ENGINE_LICENSE
    ):
        return False, "The smoke-test engine pin does not match."
    if (
        model.get("source") != VIDEO_MODEL_SOURCE
        or model.get("revision") != VIDEO_MODEL_REVISION
        or model.get("license") != VIDEO_MODEL_LICENSE
    ):
        return False, "The smoke-test model pin does not match."
    profile = proof.get("profile")
    expected_profile = {
        "width": 832,
        "height": 480,
        "num_frames": 5,
        "fps": 16,
        "steps": 10,
        "scheduler": "unipc",
        "seed": 42,
    }
    if profile != expected_profile:
        return False, "The smoke-test profile does not match the audited mode."
    generation_seconds = proof.get("generation_seconds")
    if (
        not isinstance(generation_seconds, (int, float))
        or isinstance(generation_seconds, bool)
        or generation_seconds <= 0
    ):
        return False, "The smoke-test timing is invalid."
    output = proof.get("output")
    if not isinstance(output, dict):
        return False, "The smoke-test output proof is missing."
    verification = output.get("verification")
    if not isinstance(verification, dict):
        return False, "The smoke-test media verification is missing."
    output_size = output.get("size_bytes")
    output_hash = str(output.get("sha256", ""))
    if (
        not isinstance(output_size, int)
        or isinstance(output_size, bool)
        or output_size <= 0
        or not _SHA256.fullmatch(output_hash)
    ):
        return False, "The smoke-test artifact digest is invalid."
    if runtime_root is not None:
        artifact_name = output.get("artifact")
        if not isinstance(artifact_name, str) or not artifact_name:
            return False, "The smoke-test artifact path is missing."
        artifact = (runtime_root / artifact_name).resolve(strict=False)
        try:
            artifact.relative_to(runtime_root.resolve(strict=True))
            if (
                not artifact.is_file()
                or artifact.stat().st_size != output_size
                or _sha256_file(artifact) != output_hash
            ):
                return False, "The smoke-test artifact no longer matches its proof."
        except (OSError, ValueError):
            return False, "The smoke-test artifact cannot be verified."
    if (
        verification.get("container") != "mp4"
        or verification.get("width") != 832
        or verification.get("height") != 480
        or verification.get("fps") != 16
    ):
        return False, "The smoke-test artifact does not match the audited media mode."
    frames = verification.get("num_frames")
    if not isinstance(frames, int) or isinstance(frames, bool):
        return False, "The smoke-test frame count is invalid."
    if frames != 5:
        return False, "The smoke-test frame count is invalid."
    return True, ""


def _build_runner_argv(plan: Mapping[str, Any], output_path: Path) -> list[str]:
    server = plan["_server"]
    output = plan["output"]
    sampling = plan["sampling"]
    return [
        server["runtime_python"],
        "-I",
        "-u",
        "-m",
        _ENTRYPOINT,
        "--model-dir",
        server["model_dir"],
        f"--prompt={plan['prompt']}",
        "--width",
        str(output["width"]),
        "--height",
        str(output["height"]),
        "--num-frames",
        str(output["num_frames"]),
        "--steps",
        str(sampling["steps"]),
        "--seed",
        str(sampling["seed"]),
        "--scheduler",
        "unipc",
        "--tiling",
        sampling["tiling"],
        "--output-path",
        str(output_path),
    ]


def _offline_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    server = plan["_server"]
    runtime_bin = str(Path(server["runtime_python"]).parent)
    environment = {
        "PATH": f"{runtime_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        "HF_HOME": server["hf_home"],
        "HF_HUB_CACHE": server["hf_hub_cache"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return environment


def _observe_process(
    process: Any,
    *,
    plan: Mapping[str, Any],
    progress_callback: Callable[[str, Any], None],
    cancel_check: Callable[[], bool],
    cancel_event: threading.Event,
) -> tuple[int, list[str]]:
    output_queue: queue.Queue[Any] = queue.Queue()
    stream_complete = object()

    def read_stream() -> None:
        stream = process.stdout
        try:
            if stream is not None:
                for raw_line in iter(stream.readline, ""):
                    output_queue.put(raw_line)
        finally:
            output_queue.put(stream_complete)

    reader = threading.Thread(target=read_stream, name="mlx-video-output", daemon=True)
    reader.start()
    stream_done = False
    log_lines: list[str] = []
    termination_requested = False
    while True:
        if (cancel_event.is_set() or cancel_check()) and not termination_requested:
            cancel_event.set()
            _terminate_process(process)
            termination_requested = True
        try:
            item = output_queue.get(timeout=0.1)
        except queue.Empty:
            item = None
        if item is stream_complete:
            stream_done = True
        elif isinstance(item, str):
            for fragment in item.replace("\r", "\n").splitlines():
                sanitized = _sanitize_runner_line(fragment, plan)
                if not sanitized:
                    continue
                if len(log_lines) < 2_000:
                    log_lines.append(sanitized[:1_000])
                print(f"[mlx-video] {sanitized[:1_000]}")
                _parse_runner_stage(sanitized, progress_callback)
        if stream_done and process.poll() is not None and output_queue.empty():
            break
    reader.join(timeout=1.0)
    return process.wait(), log_lines


def _parse_runner_stage(
    line: str,
    progress_callback: Callable[[str, Any], None],
) -> None:
    lowered = line.lower()
    if "loading t5 encoder" in lowered or "loading transformer" in lowered or "encoding text" in lowered:
        _emit_progress(progress_callback, "loading", 10.0)
    denoising = _DENOISING_STEPS.search(line)
    if denoising:
        total = int(denoising.group(1))
        _emit_progress(
            progress_callback,
            "denoising",
            30.0,
            current_step=0,
            total_steps=total,
        )
    matches = _DIFFUSION_PROGRESS.findall(line)
    if matches:
        current, total = (int(value) for value in matches[-1])
        if total > 0 and 0 <= current <= total:
            percent = round(30.0 + (current / total) * 50.0, 1)
            _emit_progress(
                progress_callback,
                "denoising",
                percent,
                current_step=current,
                total_steps=total,
            )
    if "decoding with vae" in lowered:
        _emit_progress(progress_callback, "decoding", 85.0)
    if "video saved to" in lowered:
        _emit_progress(progress_callback, "muxing", 95.0)


def _emit_progress(
    callback: Callable[[str, Any], None],
    stage: str,
    percent: float,
    *,
    current_step: int | None = None,
    total_steps: int | None = None,
) -> None:
    payload: dict[str, Any] = {"stage": stage, "percent": percent}
    if current_step is not None:
        payload["current_step"] = current_step
    if total_steps is not None:
        payload["total_steps"] = total_steps
    try:
        callback("video_progress", payload)
    except Exception:
        # UI progress must never control the model process.
        pass


def _terminate_process(process: Any) -> None:
    if process.poll() is not None:
        return
    terminated = False
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            terminated = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if not terminated:
        try:
            process.terminate()
        except (ProcessLookupError, OSError, AttributeError):
            pass
    try:
        process.wait(timeout=5.0)
        return
    except (subprocess.TimeoutExpired, TimeoutError):
        pass
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.kill()
    except (ProcessLookupError, OSError, AttributeError):
        pass
    try:
        process.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, TimeoutError):
        pass


def _finish_cancelled(
    job_dir: Path,
    work_dir: Path,
    provenance: dict[str, Any],
    request_artifact: str,
    provenance_artifact: str,
) -> dict[str, Any]:
    _remove_partial_video(work_dir / "generated.raw.mp4")
    _remove_partial_video(work_dir / "generated.mp4")
    _remove_partial_video(job_dir / "video.mp4")
    _cleanup_work_directory(work_dir)
    provenance.update({"status": "cancelled", "completed_at": _now_iso()})
    provenance.pop("output", None)
    artifacts = provenance.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("video", None)
    artifact_urls = provenance.get("artifact_urls")
    if isinstance(artifact_urls, dict):
        artifact_urls.pop("video", None)
    _atomic_write_json(job_dir / "provenance.json", provenance)
    return {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "status": "cancelled",
        "job_id": provenance["job_id"],
        "capability_id": VIDEO_CAPABILITY_ID,
        "request_hash": provenance["request_hash"],
        "artifacts": {
            "provenance": provenance_artifact,
            "request": request_artifact,
        },
    }


_VIDEO_INSPECT_SCRIPT = """
import cv2
import json
import sys

capture = cv2.VideoCapture(sys.argv[1])
if not capture.isOpened():
    raise SystemExit("video-open-failed")
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(capture.get(cv2.CAP_PROP_FPS))
frames = 0
first_stddev = None
while True:
    ok, frame = capture.read()
    if not ok:
        break
    if frames == 0:
        first_stddev = float(frame.std())
    frames += 1
capture.release()
print(json.dumps({
    "width": width,
    "height": height,
    "fps": fps,
    "num_frames": frames,
    "first_frame_stddev": first_stddev,
}))
"""


def _normalize_and_verify_mp4(
    raw_path: Path,
    normalized_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Correct Wan's VAE tail frames and require an exact readable MP4."""
    raw = _inspect_mp4(raw_path, plan, require_exact_frames=False)
    requested_frames = int(plan["output"]["num_frames"])
    raw_frames = int(raw["num_frames"])
    if raw_frames < requested_frames:
        raise VideoRuntimeError("The generated video contains fewer frames than requested.")

    if raw_frames == requested_frames:
        os.replace(raw_path, normalized_path)
    else:
        ffmpeg = _isolated_ffmpeg(plan)
        fps = int(plan["output"]["fps"])
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(raw_path),
                    "-vf",
                    f"select='lt(n,{requested_frames})',setpts=N/({fps}*TB)",
                    "-frames:v",
                    str(requested_frames),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(fps),
                    str(normalized_path),
                ],
                cwd=plan["_server"]["runtime_root"],
                env=_offline_environment(plan),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VideoRuntimeError("The generated video tail could not be normalized.") from exc

    verified = _inspect_mp4(normalized_path, plan, require_exact_frames=True)
    verified["raw_num_frames"] = raw_frames
    verified["tail_frames_removed"] = raw_frames - requested_frames
    verified["raw_frame_count"] = raw_frames
    verified["final_frame_count"] = verified["num_frames"]
    return verified


def _inspect_mp4(
    path: Path,
    plan: Mapping[str, Any],
    *,
    require_exact_frames: bool = True,
) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(64)
    if len(header) < 12 or header[4:8] != b"ftyp":
        raise VideoRuntimeError("The generated artifact is not a valid MP4 container.")

    server = plan["_server"]
    try:
        completed = subprocess.run(
            [server["runtime_python"], "-I", "-c", _VIDEO_INSPECT_SCRIPT, str(path)],
            cwd=server["runtime_root"],
            env=_offline_environment(plan),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=True,
        )
        metadata = json.loads(completed.stdout)
        width = int(metadata["width"])
        height = int(metadata["height"])
        fps = float(metadata["fps"])
        frames = int(metadata["num_frames"])
        first_stddev = float(metadata["first_frame_stddev"])
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        raise VideoRuntimeError("The isolated runtime could not verify the generated MP4.") from exc

    expected = plan["output"]
    if width != expected["width"] or height != expected["height"]:
        raise VideoRuntimeError("The generated video dimensions do not match the request.")
    if abs(fps - expected["fps"]) > 0.05:
        raise VideoRuntimeError("The generated video frame rate does not match the request.")
    if frames <= 0 or first_stddev <= 0:
        raise VideoRuntimeError("The generated video has no readable image content.")
    if require_exact_frames and frames != expected["num_frames"]:
        raise VideoRuntimeError("The generated video frame count does not match the request.")
    return {
        "container": "mp4",
        "container_verified": True,
        "inspector": "isolated-opencv",
        "width": width,
        "height": height,
        "fps": int(round(fps)),
        "num_frames": frames,
        "duration_seconds": round(frames / fps, 6),
        "first_frame_stddev": round(first_stddev, 6),
    }


def _isolated_ffmpeg(plan: Mapping[str, Any]) -> str:
    server = plan["_server"]
    try:
        completed = subprocess.run(
            [
                server["runtime_python"],
                "-I",
                "-c",
                "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
            ],
            cwd=server["runtime_root"],
            env=_offline_environment(plan),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=True,
        )
        executable = Path(completed.stdout.strip()).resolve(strict=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoRuntimeError("The isolated video encoder is unavailable.") from exc
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise VideoRuntimeError("The isolated video encoder is unavailable.")
    return str(executable)


def _configured_output_base() -> Path:
    configured = os.getenv(VIDEO_OUTPUT_ROOT_ENV)
    path = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[1] / "output"
    )
    if not path.is_absolute():
        raise VideoValidationError(f"{VIDEO_OUTPUT_ROOT_ENV} must be an absolute path.")
    return path.resolve(strict=False)


def _configured_hf_home() -> Path:
    configured = os.getenv("HF_HOME")
    path = Path(configured).expanduser() if configured else Path.home() / ".cache" / "huggingface"
    if not path.is_absolute():
        raise VideoValidationError("HF_HOME must be an absolute path.")
    return path.resolve(strict=False)


def _configured_hf_hub_cache() -> Path:
    configured = os.getenv("HF_HUB_CACHE")
    path = Path(configured).expanduser() if configured else _configured_hf_home() / "hub"
    if not path.is_absolute():
        raise VideoValidationError("HF_HUB_CACHE must be an absolute path.")
    return path.resolve(strict=False)


def _configured_runtime_root() -> Path:
    configured = os.getenv(VIDEO_RUNTIME_ROOT_ENV)
    path = (
        Path(configured).expanduser()
        if configured
        else _configured_hf_home() / "mlx-media" / "video-runner"
    )
    if not path.is_absolute():
        raise VideoValidationError(f"{VIDEO_RUNTIME_ROOT_ENV} must be an absolute path.")
    return path.resolve(strict=False)


def _default_runtime_python() -> Path:
    return (
        _configured_runtime_root()
        / "engines"
        / f"mlx-video-{VIDEO_ENGINE_REVISION[:12]}"
        / ".venv"
        / "bin"
        / "python"
    )


def _default_model_dir() -> Path:
    return (
        _configured_runtime_root()
        / "models"
        / f"wan2.1-t2v-1.3b-bf16-{VIDEO_MODEL_REVISION[:12]}"
    )


def _default_smoke_provenance() -> Path:
    return _configured_runtime_root() / "state" / "wan-2.1-t2v-1.3b.json"


def _path_from_environment(name: str, *, preserve_symlink: bool = False) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return None
    if preserve_symlink:
        return Path(os.path.abspath(path))
    return path.resolve(strict=False)


def _sanitize_active_job(active: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(active, Mapping):
        return None
    job_id = active.get("id") or active.get("job_id")
    job_type = active.get("type")
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        return None
    if job_type not in {"video", "photo_batch"}:
        return None
    return {"id": job_id, "type": job_type}


def _sanitize_runner_line(line: str, plan: Mapping[str, Any]) -> str:
    sanitized = _ANSI_ESCAPE.sub("", line).strip()
    if not sanitized:
        return ""
    if re.search(r"\bprompt\s*:", sanitized, re.I):
        prefix = sanitized.split(":", 1)[0]
        return f"{prefix}: [redacted]"
    server = plan["_server"]
    known_paths = [
        server["runtime_python"],
        server["model_dir"],
        server["smoke_provenance"],
        server["hf_home"],
        server["hf_hub_cache"],
        server["output_root"],
        server["runtime_root"],
        str(Path.home()),
    ]
    for value in sorted(set(known_paths), key=len, reverse=True):
        if value:
            sanitized = sanitized.replace(value, "[local-path]")
    return sanitized


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise VideoValidationError(f"{label} is missing required fields.")
    if extra:
        raise VideoValidationError(f"{label} contains unsupported fields.")


def _is_exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VideoValidationError(f"{name} must be an integer.")
    if value < minimum or value > maximum:
        raise VideoValidationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _validate_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        raise VideoValidationError("Video job id is invalid.")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _plan_integrity(plan: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in plan.items()
        if key != "_plan_integrity"
    }
    return _json_hash(payload)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _cleanup_work_directory(work_dir: Path) -> None:
    if work_dir.is_dir():
        shutil.rmtree(work_dir)


def _remove_partial_video(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _parse_fraction(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    except (ValueError, ZeroDivisionError):
        return None


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    home = str(Path.home())
    if home:
        message = message.replace(home, "[local-path]")
    return message[:500] or type(exc).__name__


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
