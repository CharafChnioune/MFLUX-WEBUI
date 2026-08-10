"""Safe, local-only SeedVR2 batch preparation and output pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from backend.photo_imports import (
    PhotoImportValidationError,
    get_import_root,
    inventory_photos,
    open_import_source,
    photo_source_id,
)


BATCH_SCHEMA_VERSION = 1
OUTPUT_ROOT_ENV = "MFLUX_PHOTO_BATCH_OUTPUT_ROOT"
SUPPORTED_BATCH_MODELS = {"seedvr2-3b"}
SUPPORTED_OUTPUT_FORMATS = {"png", "jpeg"}


class PhotoBatchValidationError(ValueError):
    """Raised before a batch can touch the output directory."""


class PhotoBatchRuntimeError(RuntimeError):
    """Raised when a prepared local batch cannot safely run."""


class PhotoBatchCancelled(Exception):
    """Internal control flow used to stop between model calls."""


def prepare_photo_batch(params: Mapping[str, Any]) -> dict[str, Any]:
    """Build an immutable source snapshot and deterministic output plan."""
    if not isinstance(params, Mapping):
        raise PhotoBatchValidationError("Photo batch parameters must be an object.")

    directory = params.get("directory")
    recursive = params.get("recursive", True)
    try:
        inventory = inventory_photos(
            directory,
            recursive=recursive,
            gps_mode=params.get("gps_mode", "suggest"),
            location_overrides=params.get("location_overrides"),
        )
    except PhotoImportValidationError as exc:
        raise PhotoBatchValidationError(str(exc)) from exc

    selected_ids = _normalize_selected_ids(params.get("file_ids"))
    inventory_by_id = {item["id"]: item for item in inventory["items"]}
    if selected_ids is None:
        selected = list(inventory["items"])
    else:
        missing = set(selected_ids) - set(inventory_by_id)
        if missing:
            raise PhotoBatchValidationError(
                "file_ids contains photos outside the current source snapshot."
            )
        selected = [inventory_by_id[file_id] for file_id in selected_ids]

    if not selected:
        raise PhotoBatchValidationError("No supported photos were selected for the batch.")

    settings = _normalize_settings(params)
    files: list[dict[str, Any]] = []
    extension = "jpg" if settings["output_format"] == "jpeg" else "png"
    for index, item in enumerate(selected, start=1):
        output_stem = f"{index:05d}-{_safe_stem(Path(item['name']).stem)}-{item['id'][:8]}"
        files.append(
            {
                "id": item["id"],
                "name": item["name"],
                "root_relative_path": item["root_relative_path"],
                "size_bytes": item["size_bytes"],
                "modified_at": item["modified_at"],
                "image": item["image"],
                "captured_at": item["captured_at"],
                "captured_at_source": item["captured_at_source"],
                "timezone_known": item["timezone_known"],
                "device": item["device"],
                "orientation": item["orientation"],
                "gps_detected": item["gps_detected"],
                "location_candidate": item["location_candidate"],
                "output_relative_path": f"images/{output_stem}.{extension}",
                "sidecar_relative_path": f"sidecars/{output_stem}.json",
                "raw_exif_relative_path": f"sidecars/{output_stem}.exif.bin",
            }
        )

    fingerprint_payload = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "source_directory": inventory["directory"],
        "settings": settings,
        "files": files,
    }
    plan_hash = _json_hash(fingerprint_payload)
    batch_id = plan_hash[:20]
    return {
        **fingerprint_payload,
        "plan_hash": plan_hash,
        "batch_id": batch_id,
        "num_images": len(files),
        "output_relative_directory": f"seedvr2-{batch_id}",
        "originals_preserved": True,
        "external_services_used": False,
    }


def public_photo_batch_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reviewable, non-absolute part of a server-created plan."""
    validate_photo_batch_plan(plan)
    return {
        "schema_version": plan["schema_version"],
        "batch_id": plan["batch_id"],
        "source_directory": plan["source_directory"],
        "num_images": plan["num_images"],
        "output_relative_directory": plan["output_relative_directory"],
        "settings": plan["settings"],
        "files": [
            {
                "id": item["id"],
                "name": item["name"],
                "root_relative_path": item["root_relative_path"],
                "captured_at": item["captured_at"],
                "orientation": item["orientation"],
                "output_relative_path": item["output_relative_path"],
            }
            for item in plan["files"]
        ],
        "originals_preserved": True,
        "external_services_used": False,
        "requires_explicit_job_submission": True,
    }


def validate_photo_batch_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping):
        raise PhotoBatchValidationError("Photo batch plan is invalid.")
    required = {"schema_version", "source_directory", "settings", "files", "plan_hash", "batch_id"}
    if not required.issubset(plan):
        raise PhotoBatchValidationError("Photo batch plan is incomplete.")
    if plan["schema_version"] != BATCH_SCHEMA_VERSION:
        raise PhotoBatchValidationError("Photo batch plan version is unsupported.")
    payload = {
        "schema_version": plan["schema_version"],
        "source_directory": plan["source_directory"],
        "settings": plan["settings"],
        "files": plan["files"],
    }
    expected = _json_hash(payload)
    if plan["plan_hash"] != expected or plan["batch_id"] != expected[:20]:
        raise PhotoBatchValidationError("Photo batch plan no longer matches its source snapshot.")
    if plan.get("num_images") != len(plan["files"]):
        raise PhotoBatchValidationError("Photo batch image count is invalid.")
    if plan.get("output_relative_directory") != f"seedvr2-{plan['batch_id']}":
        raise PhotoBatchValidationError("Photo batch output directory is invalid.")


def get_photo_batch_output_root(*, create: bool = False) -> Path:
    configured = os.getenv(OUTPUT_ROOT_ENV)
    root = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[1] / "output" / "photo-batches"
    )
    if not root.is_absolute():
        raise PhotoBatchValidationError(f"{OUTPUT_ROOT_ENV} must be an absolute path.")
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise PhotoBatchValidationError("Photo batch output root is not a directory.")
    else:
        root = root.resolve(strict=False)
    return root


def run_photo_batch(
    plan: Mapping[str, Any],
    *,
    progress_callback: Callable[[str, Any], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    enhance_one: Callable[[Path, Path, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a prepared plan serially, reusing one local SeedVR2 model."""
    validate_photo_batch_plan(plan)
    progress = progress_callback or (lambda event, data=None: None)
    is_cancelled = cancel_check or (lambda: False)

    import_root = get_import_root(require_exists=True)
    output_root = get_photo_batch_output_root(create=False)
    # Validate before mkdir: a bad output setting must never create anything
    # inside the read-only import tree.
    _ensure_disjoint_roots(import_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_root = output_root.resolve(strict=True)
    if not output_root.is_dir():
        raise PhotoBatchValidationError("Photo batch output root is not a directory.")
    _ensure_disjoint_roots(import_root, output_root)

    batch_dir = output_root / plan["output_relative_directory"]
    try:
        batch_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise PhotoBatchValidationError(
            "This deterministic batch output already exists; existing outputs will not be overwritten."
        ) from exc

    images_dir = batch_dir / "images"
    sidecars_dir = batch_dir / "sidecars"
    work_dir = batch_dir / ".work"
    for directory in (images_dir, sidecars_dir, work_dir):
        directory.mkdir(mode=0o700)

    manifest = _new_manifest(plan)
    _write_manifest(batch_dir, manifest)
    progress("stage", "preparing_batch")

    if is_cancelled():
        manifest["status"] = "cancelled"
        manifest["completed_at"] = _now_iso()
        _write_manifest(batch_dir, manifest)
        _cleanup_work_dir(work_dir)
        return _batch_result(plan, manifest)

    enhancer_context = _huggingface_offline() if enhance_one is None else nullcontext()
    try:
        with enhancer_context:
            progress("stage", "loading_model")
            enhancer = enhance_one or _create_seedvr2_enhancer(plan["settings"])
            for index, file_plan in enumerate(plan["files"], start=1):
                if is_cancelled():
                    manifest["status"] = "cancelled"
                    break

                progress(
                    "image_start",
                    {"current_image": index, "total_images": plan["num_images"], "seed": None},
                )
                entry = manifest["files"][index - 1]
                entry["status"] = "running"
                entry["started_at"] = _now_iso()
                _write_manifest(batch_dir, manifest)

                try:
                    completed = _process_photo(
                        import_root=import_root,
                        batch_dir=batch_dir,
                        work_dir=work_dir,
                        file_plan=file_plan,
                        settings=plan["settings"],
                        enhancer=enhancer,
                        cancel_check=is_cancelled,
                    )
                except PhotoBatchCancelled:
                    entry["status"] = "cancelled"
                    manifest["status"] = "cancelled"
                    _write_manifest(batch_dir, manifest)
                    break
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["completed_at"] = _now_iso()
                    entry["error"] = _safe_processing_error(exc)
                    manifest["failed_count"] += 1
                    progress("image_error", {"current_image": index, "error": entry["error"]})
                else:
                    entry.update(completed)
                    entry["status"] = "completed"
                    entry["completed_at"] = _now_iso()
                    manifest["completed_count"] += 1

                progress("image_complete", {"current_image": index, "total_images": plan["num_images"]})
                _write_manifest(batch_dir, manifest)

        if manifest["status"] == "running":
            manifest["status"] = (
                "completed_with_errors" if manifest["failed_count"] else "completed"
            )
    except Exception:
        manifest["status"] = "failed"
        manifest["completed_at"] = _now_iso()
        _write_manifest(batch_dir, manifest)
        _cleanup_work_dir(work_dir)
        raise

    manifest["completed_at"] = _now_iso()
    _write_manifest(batch_dir, manifest)
    _cleanup_work_dir(work_dir)
    progress("stage", manifest["status"])
    return _batch_result(plan, manifest)


def preprocess_photo_source(
    source: Any,
    *,
    filename: str,
    destination: Path,
) -> dict[str, Any]:
    """Decode locally, apply EXIF orientation to pixels, and write a metadata-free PNG."""
    suffix = Path(filename).suffix.lower()
    is_heif = suffix in {".heic", ".heif"}
    if is_heif:
        _register_heif_opener_required()

    from PIL import Image, ImageOps

    source.seek(0)
    source_sha256 = _sha256_stream(source)
    source.seek(0)
    with Image.open(source) as image:
        image.load()
        exif = image.getexif()
        orientation_raw = exif.get(274) if exif else None
        orientation = int(orientation_raw) if isinstance(orientation_raw, (int, float)) else None
        raw_exif = image.info.get("exif")
        raw_exif = bytes(raw_exif) if isinstance(raw_exif, (bytes, bytearray)) else None
        source_dimensions = {"width": image.width, "height": image.height}

        normalized = ImageOps.exif_transpose(image)
        if normalized.mode != "RGB":
            normalized = normalized.convert("RGB")
        else:
            normalized = normalized.copy()
        normalized_dimensions = {"width": normalized.width, "height": normalized.height}
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open("xb") as output:
            normalized.save(output, format="PNG", compress_level=1)

    return {
        "decoder": "pillow-heif" if is_heif else "Pillow",
        "source_sha256": source_sha256,
        "source_dimensions": source_dimensions,
        "normalized_dimensions": normalized_dimensions,
        "orientation_before": orientation,
        "exif_transposed": bool(orientation not in (None, 1)),
        "container_transform_applied_by_decoder": is_heif,
        "raw_exif": raw_exif,
    }


def _process_photo(
    *,
    import_root: Path,
    batch_dir: Path,
    work_dir: Path,
    file_plan: Mapping[str, Any],
    settings: Mapping[str, Any],
    enhancer: Callable[[Path, Path, int, Mapping[str, Any]], None],
    cancel_check: Callable[[], bool],
) -> dict[str, Any]:
    root_relative_path = file_plan["root_relative_path"]
    preprocessed_path = work_dir / f"{file_plan['id']}.png"
    generated_path = work_dir / f"{file_plan['id']}.generated.{_output_extension(settings)}"
    output_path = batch_dir / file_plan["output_relative_path"]
    sidecar_path = batch_dir / file_plan["sidecar_relative_path"]
    raw_exif_path = batch_dir / file_plan["raw_exif_relative_path"]

    try:
        with open_import_source(import_root, root_relative_path) as (source, source_stat):
            if photo_source_id(root_relative_path, source_stat) != file_plan["id"]:
                raise PhotoBatchValidationError(
                    "A source photo changed after the batch plan was created."
                )
            preprocessing = preprocess_photo_source(
                source,
                filename=file_plan["name"],
                destination=preprocessed_path,
            )

        if cancel_check():
            raise PhotoBatchCancelled()

        seed = _file_seed(settings["batch_seed"], file_plan["id"])
        enhancer(preprocessed_path, generated_path, seed, settings)
        if cancel_check():
            raise PhotoBatchCancelled()
        output_dimensions = _verify_generated_image(generated_path)
        _publish_new(generated_path, output_path)
        os.utime(output_path, ns=(source_stat.st_mtime_ns, source_stat.st_mtime_ns))
        output_sha256 = _sha256_path(output_path)

        raw_exif = preprocessing.pop("raw_exif")
        raw_exif_sidecar = None
        raw_exif_sha256 = None
        if raw_exif:
            _write_new_bytes(raw_exif_path, raw_exif)
            raw_exif_sidecar = file_plan["raw_exif_relative_path"]
            raw_exif_sha256 = hashlib.sha256(raw_exif).hexdigest()

        sidecar = {
            "schema_version": BATCH_SCHEMA_VERSION,
            "source": {
                "id": file_plan["id"],
                "root_relative_path": root_relative_path,
                "name": file_plan["name"],
                "size_bytes": source_stat.st_size,
                "modified_at": file_plan["modified_at"],
                "sha256": preprocessing["source_sha256"],
                "captured_at": file_plan["captured_at"],
                "captured_at_source": file_plan["captured_at_source"],
                "timezone_known": file_plan["timezone_known"],
                "device": file_plan["device"],
                "orientation": file_plan["orientation"],
                "gps_detected": file_plan["gps_detected"],
                "location_candidate": file_plan["location_candidate"],
                "raw_exif_sidecar": raw_exif_sidecar,
                "raw_exif_sha256": raw_exif_sha256,
            },
            "preprocessing": preprocessing,
            "enhancement": {
                "engine": "SeedVR2",
                "model": settings["model"],
                "resolution": settings["resolution"],
                "softness": settings["softness"],
                "seed": seed,
                "huggingface_cache": "shared_default",
                "network_access": "disabled",
            },
            "output": {
                "relative_path": file_plan["output_relative_path"],
                "format": settings["output_format"],
                "dimensions": output_dimensions,
                "sha256": output_sha256,
                "source_mtime_copied": True,
                "metadata_embedded": False,
            },
            "privacy": {
                "external_requests_performed": 0,
                "gps_embedded_in_output": False,
                "gps_preserved_in_local_sidecar": bool(file_plan["gps_detected"]),
            },
        }
        _write_new_json(sidecar_path, sidecar)
        return {
            "output_relative_path": file_plan["output_relative_path"],
            "sidecar_relative_path": file_plan["sidecar_relative_path"],
            "raw_exif_relative_path": raw_exif_sidecar,
            "output_sha256": output_sha256,
            "seed": seed,
        }
    finally:
        for temporary in (preprocessed_path, generated_path):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _create_seedvr2_enhancer(settings: Mapping[str, Any]):
    from backend.seedvr2_manager import coerce_seedvr2_resolution, load_seedvr2_model

    model = load_seedvr2_model(settings["model"])
    resolution = coerce_seedvr2_resolution(settings["resolution"])

    def enhance(input_path: Path, output_path: Path, seed: int, current_settings: Mapping[str, Any]):
        generated = model.generate_image(
            seed=seed,
            image_path=input_path,
            resolution=resolution,
            softness=current_settings["softness"],
        )
        image = getattr(generated, "image", None)
        if image is None:
            generated.save(output_path)
            return
        if current_settings["output_format"] == "jpeg":
            image.convert("RGB").save(
                output_path,
                format="JPEG",
                quality=current_settings["jpeg_quality"],
                subsampling=0,
                optimize=True,
            )
        else:
            image.save(output_path, format="PNG")

    return enhance


def _normalize_selected_ids(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PhotoBatchValidationError("file_ids must be an array of inventory IDs.")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(r"[a-f0-9]{20}", item):
            raise PhotoBatchValidationError("file_ids contains an invalid inventory ID.")
        if item in seen:
            raise PhotoBatchValidationError("file_ids contains duplicates.")
        seen.add(item)
        normalized.append(item)
    return normalized


def _normalize_settings(params: Mapping[str, Any]) -> dict[str, Any]:
    raw_model = str(params.get("model", "seedvr2-3b")).strip().lower()
    model = "seedvr2-3b" if raw_model in {"3b", "seedvr2-3b"} else raw_model
    if model not in SUPPORTED_BATCH_MODELS:
        raise PhotoBatchValidationError("Only the verified SeedVR2 3B batch model is supported.")

    resolution = _normalize_resolution(params.get("resolution", "2x"))
    try:
        softness = float(params.get("softness", 0.0))
    except (TypeError, ValueError) as exc:
        raise PhotoBatchValidationError("softness must be between 0.0 and 1.0.") from exc
    if not 0.0 <= softness <= 1.0:
        raise PhotoBatchValidationError("softness must be between 0.0 and 1.0.")

    try:
        batch_seed = int(params.get("batch_seed", 0))
    except (TypeError, ValueError) as exc:
        raise PhotoBatchValidationError("batch_seed must be an unsigned 32-bit integer.") from exc
    if not 0 <= batch_seed <= 2**32 - 1:
        raise PhotoBatchValidationError("batch_seed must be an unsigned 32-bit integer.")

    output_format = str(params.get("output_format", "png")).strip().lower()
    if output_format == "jpg":
        output_format = "jpeg"
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise PhotoBatchValidationError("output_format must be 'png' or 'jpeg'.")
    try:
        jpeg_quality = int(params.get("jpeg_quality", 95))
    except (TypeError, ValueError) as exc:
        raise PhotoBatchValidationError("jpeg_quality must be between 80 and 100.") from exc
    if not 80 <= jpeg_quality <= 100:
        raise PhotoBatchValidationError("jpeg_quality must be between 80 and 100.")

    return {
        "model": model,
        "resolution": resolution,
        "softness": softness,
        "batch_seed": batch_seed,
        "output_format": output_format,
        "jpeg_quality": jpeg_quality,
        "orientation_policy": "exif_transpose_before_model",
        "metadata_policy": "local_sidecars_no_output_exif",
        "source_mtime_policy": "copy_to_output",
        "huggingface_cache": "shared_default_offline_only",
    }


def _normalize_resolution(value: Any) -> str | int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if re.fullmatch(r"[1-4]x", normalized):
            return normalized
        try:
            value = int(normalized)
        except ValueError as exc:
            raise PhotoBatchValidationError(
                "resolution must be 1x through 4x or a target shorter edge from 256 to 4096."
            ) from exc
    if isinstance(value, bool):
        raise PhotoBatchValidationError("resolution is invalid.")
    try:
        resolution = int(value)
    except (TypeError, ValueError) as exc:
        raise PhotoBatchValidationError("resolution is invalid.") from exc
    if not 256 <= resolution <= 4096:
        raise PhotoBatchValidationError(
            "resolution must be 1x through 4x or a target shorter edge from 256 to 4096."
        )
    return resolution


def _new_manifest(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": plan["batch_id"],
        "plan_hash": plan["plan_hash"],
        "status": "running",
        "created_at": _now_iso(),
        "completed_at": None,
        "source_directory": plan["source_directory"],
        "settings": plan["settings"],
        "total_count": plan["num_images"],
        "completed_count": 0,
        "failed_count": 0,
        "originals_preserved": True,
        "external_requests_performed": 0,
        "files": [
            {
                "id": item["id"],
                "root_relative_path": item["root_relative_path"],
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "output_relative_path": item["output_relative_path"],
                "sidecar_relative_path": item["sidecar_relative_path"],
            }
            for item in plan["files"]
        ],
    }


def _batch_result(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": plan["batch_id"],
        "status": manifest["status"],
        "output_relative_directory": plan["output_relative_directory"],
        "manifest_relative_path": f"{plan['output_relative_directory']}/manifest.json",
        "total_count": manifest["total_count"],
        "completed_count": manifest["completed_count"],
        "failed_count": manifest["failed_count"],
        "originals_preserved": True,
        "external_requests_performed": 0,
    }


def _ensure_disjoint_roots(import_root: Path, output_root: Path) -> None:
    try:
        output_root.relative_to(import_root)
    except ValueError:
        pass
    else:
        raise PhotoBatchValidationError("Photo batch output root must be outside the import root.")
    try:
        import_root.relative_to(output_root)
    except ValueError:
        return
    raise PhotoBatchValidationError("Photo import root must be outside the batch output root.")


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")[:80]
    return normalized or "photo"


def _file_seed(batch_seed: int, file_id: str) -> int:
    digest = hashlib.sha256(f"{batch_seed}:{file_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _output_extension(settings: Mapping[str, Any]) -> str:
    return "jpg" if settings["output_format"] == "jpeg" else "png"


def _verify_generated_image(path: Path) -> dict[str, int]:
    from PIL import Image

    if not path.is_file() or path.is_symlink():
        raise PhotoBatchRuntimeError("SeedVR2 did not create a regular output image.")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {"width": image.width, "height": image.height}


def _publish_new(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise PhotoBatchValidationError("A batch output already exists and will not be overwritten.") from exc
    temporary.unlink()


def _write_new_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    _write_new_bytes(path, data)


def _write_manifest(batch_dir: Path, payload: Mapping[str, Any]) -> None:
    manifest_path = batch_dir / "manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=batch_dir,
        prefix=".manifest-",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True, ensure_ascii=False)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, manifest_path)


def _sha256_stream(source: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as source:
        return _sha256_stream(source)


def _safe_processing_error(exc: Exception) -> str:
    if isinstance(exc, (PhotoBatchValidationError, PhotoBatchRuntimeError)):
        return str(exc)
    return f"{type(exc).__name__}: photo processing failed"


def _cleanup_work_dir(work_dir: Path) -> None:
    if not work_dir.exists():
        return
    for path in work_dir.iterdir():
        if path.is_file() and not path.is_symlink():
            path.unlink()
    try:
        work_dir.rmdir()
    except OSError:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _register_heif_opener_required() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        raise PhotoBatchRuntimeError(
            "HEIC/HEIF decoding requires the local pillow-heif dependency."
        ) from exc
    register_heif_opener(thumbnails=False)


@contextmanager
def _huggingface_offline():
    """Force the batch to use only the already-populated shared HF cache."""
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
