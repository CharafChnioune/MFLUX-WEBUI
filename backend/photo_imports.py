"""Local-only photo inventory and normalized EXIF metadata extraction."""

from __future__ import annotations

import hashlib
import math
import mimetypes
import os
import re
import stat as stat_module
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


IMPORT_ROOT_ENV = "MFLUX_PHOTO_IMPORT_ROOT"
MAX_FILES_ENV = "MFLUX_PHOTO_IMPORT_MAX_FILES"
DEFAULT_MAX_FILES = 5_000
HARD_MAX_FILES = 10_000
GPS_MODES = {"suggest", "disabled"}
SUPPORTED_EXTENSIONS = {
    ".avif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class PhotoImportValidationError(ValueError):
    """Raised when a requested import path or option is unsafe or invalid."""


def get_import_root(*, require_exists: bool = False) -> Path:
    """Return the configured import root without creating or modifying it."""
    configured = os.getenv(IMPORT_ROOT_ENV)
    if not configured:
        raise PhotoImportValidationError(
            f"Set {IMPORT_ROOT_ENV} to an absolute local directory before importing photos."
        )
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise PhotoImportValidationError(f"{IMPORT_ROOT_ENV} must be an absolute path.")

    try:
        root = root.resolve(strict=require_exists)
    except OSError as exc:
        raise PhotoImportValidationError("The configured photo import root is unavailable.") from exc

    if require_exists and not root.is_dir():
        raise PhotoImportValidationError("The configured photo import root is not a directory.")
    return root


def resolve_import_directory(directory: str, *, root: Path | None = None) -> Path:
    """Resolve a chosen directory and prove it remains inside the import root."""
    if not isinstance(directory, str) or not directory.strip():
        raise PhotoImportValidationError("directory must be a non-empty path string.")
    if "\x00" in directory:
        raise PhotoImportValidationError("directory contains an invalid character.")

    import_root = (root or get_import_root(require_exists=True)).resolve(strict=True)
    requested = Path(directory.strip()).expanduser()
    if requested.is_absolute():
        raise PhotoImportValidationError("directory must be relative to the configured import root.")
    candidate = import_root / requested

    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(import_root)
    except (OSError, ValueError) as exc:
        raise PhotoImportValidationError(
            "The chosen directory must exist inside the configured photo import root."
        ) from exc

    if not resolved.is_dir():
        raise PhotoImportValidationError("The chosen import path is not a directory.")
    return resolved


def get_photo_import_config() -> dict[str, Any]:
    try:
        root = get_import_root(require_exists=False)
        configured = True
        configuration_error = None
    except PhotoImportValidationError as exc:
        root = None
        configured = False
        configuration_error = str(exc)
    return {
        "configured": configured,
        "configuration_error": configuration_error,
        "import_root": str(root) if root is not None else None,
        "import_root_exists": bool(root and root.is_dir()),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "max_files": _configured_max_files(),
        "preserves_originals": True,
        "privacy": {
            "local_only": True,
            "external_services_used": False,
            "publishes_metadata": False,
            "default_gps_mode": "suggest",
            "gps_stays_local": True,
            "gps_can_be_disabled": True,
            "location_can_be_overridden": True,
            "location_candidates_are_coordinates_only": True,
        },
    }


def inventory_photos(
    directory: str,
    *,
    recursive: bool = True,
    gps_mode: str = "suggest",
    location_overrides: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Inventory photos without writing to, moving, or modifying source files."""
    if not isinstance(recursive, bool):
        raise PhotoImportValidationError("recursive must be true or false.")
    if gps_mode not in GPS_MODES:
        raise PhotoImportValidationError("gps_mode must be 'suggest' or 'disabled'.")

    import_root = (root or get_import_root(require_exists=True)).resolve(strict=True)
    chosen = resolve_import_directory(directory, root=import_root)
    overrides = _normalize_overrides(location_overrides or {})
    max_files = _configured_max_files()

    items: list[dict[str, Any]] = []
    used_overrides: set[str] = set()
    metadata_errors = 0
    skipped_symlinks = 0
    truncated = False

    for source_path in _iter_files(chosen, recursive=recursive):
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if source_path.is_symlink():
            skipped_symlinks += 1
            continue
        if len(items) >= max_files:
            truncated = True
            break

        try:
            relative_path = source_path.relative_to(chosen).as_posix()
            root_relative_path = source_path.relative_to(import_root).as_posix()
        except ValueError:
            skipped_symlinks += 1
            continue

        try:
            with _open_regular_source(import_root, root_relative_path) as (source, stat):
                try:
                    extracted = extract_photo_metadata(
                        source,
                        stat=stat,
                        filename=source_path.name,
                    )
                    metadata_status = "ok"
                except Exception:
                    extracted = _filesystem_metadata(stat)
                    metadata_status = "unavailable"
                    metadata_errors += 1
        except (OSError, PhotoImportValidationError):
            skipped_symlinks += 1
            continue

        gps = extracted.pop("_gps", None)
        override = overrides.get(root_relative_path)
        if override is not None:
            used_overrides.add(root_relative_path)
        location_candidate = _location_candidate(gps, override, gps_mode)

        identity = f"{root_relative_path}\0{stat.st_size}\0{stat.st_mtime_ns}"
        items.append(
            {
                "id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                "name": source_path.name,
                "relative_path": relative_path,
                "root_relative_path": root_relative_path,
                "extension": source_path.suffix.lower(),
                "mime_type": mimetypes.guess_type(source_path.name)[0] or "application/octet-stream",
                "size_bytes": stat.st_size,
                "modified_at": _timestamp_iso(stat.st_mtime),
                "metadata_status": metadata_status,
                "image": extracted["image"],
                "captured_at": extracted["captured_at"],
                "captured_at_source": extracted["captured_at_source"],
                "timezone_known": extracted["timezone_known"],
                "device": extracted["device"],
                "orientation": extracted["orientation"],
                "gps_detected": gps is not None,
                "location_candidate": location_candidate,
            }
        )

    items.sort(key=lambda item: item["relative_path"].casefold())
    selected_relative = chosen.relative_to(import_root).as_posix()
    unused_overrides = sorted(set(overrides) - used_overrides)
    if unused_overrides:
        raise PhotoImportValidationError(
            "location_overrides contains paths that are not present in the selected inventory."
        )
    return {
        "directory": selected_relative,
        "recursive": recursive,
        "items": items,
        "summary": {
            "image_count": len(items),
            "metadata_errors": metadata_errors,
            "skipped_symlinks": skipped_symlinks,
            "truncated": truncated,
            "max_files": max_files,
        },
        "originals": {
            "preserved": True,
            "write_operations_performed": 0,
        },
        "privacy": {
            "local_only": True,
            "external_requests_performed": 0,
            "gps_mode": gps_mode,
            "gps_stays_local": True,
            "gps_used_only_for_location_candidates": gps_mode == "suggest",
            "location_overrides_applied": len(used_overrides),
        },
    }


def extract_photo_metadata(
    source: Any,
    *,
    stat: os.stat_result,
    filename: str,
) -> dict[str, Any]:
    """Extract a small normalized metadata subset; never return raw EXIF blobs."""
    if Path(filename).suffix.lower() in {".heic", ".heif"}:
        _register_optional_heif_opener()

    from PIL import Image

    metadata = _filesystem_metadata(stat)
    with Image.open(source) as image:
        metadata["image"] = {
            "width": int(image.width),
            "height": int(image.height),
            "format": image.format,
            "mode": image.mode,
        }
        exif = image.getexif()
        if not exif:
            return metadata

        exif_ifd = _get_ifd(exif, 34665)
        gps_ifd = _get_ifd(exif, 34853)

        captured_raw = _first_value(exif_ifd, exif, keys=(36867, 36868, 306))
        offset_raw = _first_value(exif_ifd, exif, keys=(36881, 36882, 36880))
        subsecond_raw = _first_value(exif_ifd, exif, keys=(37521, 37522, 37520))
        captured = _normalize_capture_date(captured_raw, offset_raw, subsecond_raw)
        if captured is not None:
            metadata["captured_at"] = captured[0]
            metadata["captured_at_source"] = "exif"
            metadata["timezone_known"] = captured[1]

        metadata["device"] = {
            "make": _text_value(exif.get(271)),
            "model": _text_value(exif.get(272)),
            "lens_model": _text_value(_first_value(exif_ifd, exif, keys=(42036,))),
            "software": _text_value(exif.get(305)),
        }
        orientation = exif.get(274)
        metadata["orientation"] = int(orientation) if isinstance(orientation, (int, float)) else None
        metadata["_gps"] = _gps_from_ifd(gps_ifd)
    return metadata


def _filesystem_metadata(stat: os.stat_result) -> dict[str, Any]:
    return {
        "image": {"width": None, "height": None, "format": None, "mode": None},
        "captured_at": _timestamp_iso(stat.st_mtime),
        "captured_at_source": "filesystem_modified",
        "timezone_known": True,
        "device": {"make": None, "model": None, "lens_model": None, "software": None},
        "orientation": None,
        "_gps": None,
    }


def _iter_files(directory: Path, *, recursive: bool):
    if not recursive:
        for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() or path.is_symlink():
                yield path
        return

    for current, directories, filenames in os.walk(directory, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            [name for name in directories if not (current_path / name).is_symlink()],
            key=str.casefold,
        )
        for filename in sorted(filenames, key=str.casefold):
            yield current_path / filename


@contextmanager
def _open_regular_source(import_root: Path, root_relative_path: str):
    """Open one source read-only without following any path component symlink."""
    relative = PurePosixPath(_normalize_relative_path(root_relative_path))
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow_flags = getattr(os, "O_NOFOLLOW", 0)
    open_directories: list[int] = []
    source_fd: int | None = None
    try:
        current_fd = os.open(import_root, directory_flags | nofollow_flags)
        open_directories.append(current_fd)
        for component in relative.parts[:-1]:
            current_fd = os.open(
                component,
                directory_flags | nofollow_flags,
                dir_fd=current_fd,
            )
            open_directories.append(current_fd)

        source_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow_flags,
            dir_fd=current_fd,
        )
        source_stat = os.fstat(source_fd)
        if not stat_module.S_ISREG(source_stat.st_mode):
            raise PhotoImportValidationError("Only regular image files can be inventoried.")

        with os.fdopen(source_fd, "rb", closefd=True) as source:
            source_fd = None
            yield source, source_stat
    finally:
        if source_fd is not None:
            os.close(source_fd)
        for descriptor in reversed(open_directories):
            os.close(descriptor)


def _configured_max_files() -> int:
    raw = os.getenv(MAX_FILES_ENV, str(DEFAULT_MAX_FILES))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_FILES
    return min(max(value, 1), HARD_MAX_FILES)


def _normalize_overrides(raw_overrides: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_overrides, Mapping):
        raise PhotoImportValidationError("location_overrides must be an object.")
    normalized: dict[str, dict[str, Any]] = {}
    casefolded_paths: set[str] = set()
    for raw_path, raw_override in raw_overrides.items():
        path = _normalize_relative_path(raw_path)
        folded_path = path.casefold()
        if folded_path in casefolded_paths:
            raise PhotoImportValidationError("location_overrides contains colliding paths.")
        casefolded_paths.add(folded_path)
        if not isinstance(raw_override, Mapping):
            raise PhotoImportValidationError(f"Location override for {path} must be an object.")

        unknown = set(raw_override) - {"disabled", "label", "latitude", "longitude"}
        if unknown:
            raise PhotoImportValidationError(f"Location override for {path} has unsupported fields.")

        disabled = raw_override.get("disabled", False)
        if not isinstance(disabled, bool):
            raise PhotoImportValidationError(f"Location override disabled flag for {path} must be boolean.")

        label = raw_override.get("label")
        if label is not None:
            if not isinstance(label, str) or not label.strip() or len(label.strip()) > 200:
                raise PhotoImportValidationError(f"Location override label for {path} is invalid.")
            label = label.strip()

        latitude = raw_override.get("latitude")
        longitude = raw_override.get("longitude")
        if (latitude is None) != (longitude is None):
            raise PhotoImportValidationError(
                f"Location override for {path} must include both latitude and longitude."
            )
        if latitude is not None:
            latitude = _coordinate(latitude, minimum=-90, maximum=90, field="latitude")
            longitude = _coordinate(longitude, minimum=-180, maximum=180, field="longitude")

        if not disabled and label is None and latitude is None:
            raise PhotoImportValidationError(f"Location override for {path} is empty.")
        normalized[path] = {
            "disabled": disabled,
            "label": label,
            "latitude": latitude,
            "longitude": longitude,
        }
    return normalized


def _normalize_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value or "\x00" in value:
        raise PhotoImportValidationError("Location override paths must be normalized relative paths.")
    path = PurePosixPath(value.strip())
    normalized = path.as_posix()
    if path.is_absolute() or ".." in path.parts or normalized in {"", "."} or normalized != value.strip():
        raise PhotoImportValidationError("Location override paths must be normalized relative paths.")
    return normalized


def _location_candidate(
    gps: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
    gps_mode: str,
) -> dict[str, Any] | None:
    if override is not None:
        if override["disabled"]:
            return None
        return {
            "source": "user_override",
            "label": override["label"],
            "latitude": override["latitude"],
            "longitude": override["longitude"],
            "altitude_m": None,
            "external_lookup_performed": False,
        }
    if gps_mode == "disabled" or gps is None:
        return None
    return {
        "source": "exif_gps",
        "label": None,
        "latitude": gps["latitude"],
        "longitude": gps["longitude"],
        "altitude_m": gps.get("altitude_m"),
        "external_lookup_performed": False,
    }


def _gps_from_ifd(gps: Mapping[int, Any] | None) -> dict[str, float | None] | None:
    if not gps:
        return None
    latitude = _gps_to_decimal(gps.get(2), gps.get(1))
    longitude = _gps_to_decimal(gps.get(4), gps.get(3))
    if latitude is None or longitude is None:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None

    altitude = _ratio_to_float(gps.get(6))
    altitude_ref = gps.get(5)
    if altitude is not None and altitude_ref in (1, b"\x01"):
        altitude = -altitude
    return {
        "latitude": round(latitude, 7),
        "longitude": round(longitude, 7),
        "altitude_m": round(altitude, 2) if altitude is not None else None,
    }


def _gps_to_decimal(value: Any, reference: Any) -> float | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    parts = [_ratio_to_float(part) for part in value]
    if any(part is None for part in parts):
        return None
    degrees, minutes, seconds = parts
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    ref = _text_value(reference)
    if ref and ref.upper() in {"S", "W"}:
        decimal *= -1
    return decimal


def _ratio_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if numerator is not None and denominator is not None:
            return float(numerator) / float(denominator)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _coordinate(value: Any, *, minimum: float, maximum: float, field: str) -> float:
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as exc:
        raise PhotoImportValidationError(f"Location override {field} must be numeric.") from exc
    if not math.isfinite(coordinate) or not minimum <= coordinate <= maximum:
        raise PhotoImportValidationError(f"Location override {field} is outside its valid range.")
    return round(coordinate, 7)


def _normalize_capture_date(value: Any, offset: Any = None, subseconds: Any = None):
    text = _text_value(value)
    if not text:
        return None
    try:
        captured = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    subsecond_text = _text_value(subseconds)
    if subsecond_text:
        digits = "".join(character for character in subsecond_text if character.isdigit())[:6]
        if digits:
            captured = captured.replace(microsecond=int(digits.ljust(6, "0")))

    timezone_known = False
    offset_text = _text_value(offset)
    match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", offset_text or "")
    if match:
        sign = 1 if match.group(1) == "+" else -1
        delta = timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))
        captured = captured.replace(tzinfo=timezone(sign * delta))
        timezone_known = True
    return captured.isoformat(), timezone_known


def _timestamp_iso(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return text or None


def _get_ifd(exif: Any, tag: int) -> Mapping[int, Any]:
    try:
        value = exif.get_ifd(tag)
    except (AttributeError, KeyError, TypeError, ValueError):
        value = exif.get(tag, {})
    return value if isinstance(value, Mapping) else {}


def _first_value(*mappings: Mapping[int, Any], keys: tuple[int, ...]):
    for key in keys:
        for mapping in mappings:
            if key in mapping:
                return mapping[key]
    return None


def _register_optional_heif_opener() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()
