"""
Lightweight API server for MFLUX-WEBUI (SD WebUI-style + async job API).
Endpoints:
- POST /sdapi/v1/txt2img
- POST /sdapi/v1/img2img
- POST /sdapi/v1/controlnet
- POST /api/upscale             (simple factor-based upscaling)
- POST /api/v1/generate         (async job submission)
- GET  /api/v1/jobs             (list jobs)
- GET  /api/v1/jobs/{id}        (job status)
- GET  /api/v1/jobs/{id}/stream (SSE stream)
- DELETE /api/v1/jobs/{id}      (cancel job)
- GET  /api/v1/health
- GET  /api/v1/models
- GET  /api/v1/photo-imports/config
- POST /api/v1/photo-imports/inventory
- POST /api/v1/photo-batches/plan
- GET  /api/v1/system
- GET  /api/v1/queue
- GET  /api/v1/stats

Launch: python -m backend.api_server [host] [port]
Default: host=127.0.0.1, port=7861
"""

import base64
import json
import re
import sys
import tempfile
import time
import os
import mimetypes
import ipaddress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 7861
DEFAULT_MODEL = "flux2-klein-4b"
_CURRENT_MODEL = None
WEBUI_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

# Regex patterns for path-parameter routes
_RE_JOB_STREAM = re.compile(r"^/api/v1/jobs/([a-f0-9]+)/stream$")
_RE_JOB_DETAIL = re.compile(r"^/api/v1/jobs/([a-f0-9]+)$")


def _current_model():
    """Return the currently selected model, falling back to the first available."""
    global _CURRENT_MODEL
    if _CURRENT_MODEL:
        return _CURRENT_MODEL
    try:
        from backend.model_manager import get_updated_models

        available = get_updated_models()
        if DEFAULT_MODEL in available:
            _CURRENT_MODEL = DEFAULT_MODEL
        elif available:
            _CURRENT_MODEL = available[0]
        else:
            _CURRENT_MODEL = DEFAULT_MODEL
    except Exception:
        _CURRENT_MODEL = DEFAULT_MODEL
    return _CURRENT_MODEL


def _set_current_model(model_name: str | None):
    global _CURRENT_MODEL
    if not model_name:
        return
    _CURRENT_MODEL = model_name


def _resolve_model_from_payload(data: dict) -> str:
    """
    Open WebUI may send either `model` or the SD-WebUI compatible
    `sd_model_checkpoint`, sometimes nested in override_settings. Prefer
    explicit request and fall back to the selected model.
    """
    requested = data.get("model") or data.get("sd_model_checkpoint")
    override_settings = data.get("override_settings") or {}
    requested = requested or override_settings.get("sd_model_checkpoint")
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return _current_model()


def _list_models_payload():
    """
    Return SD WebUI style model descriptors for available aliases.
    """
    models = []
    try:
        from backend.model_manager import get_custom_model_config, get_updated_models

        aliases = get_updated_models()
    except Exception:
        aliases = [DEFAULT_MODEL]

    for alias in aliases:
        try:
            cfg = get_custom_model_config(alias)
            model_name = cfg.model_name
            base_arch = cfg.base_arch
        except Exception:
            model_name = alias
            base_arch = ""
        models.append(
            {
                "title": alias,
                "model_name": model_name,
                "hash": "",
                "sha256": "",
                "filename": alias,
                "config": base_arch,
            }
        )
    return models


def _bad_request(handler, message, status=400):
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps({"error": message}).encode("utf-8"))
    except BrokenPipeError:
        pass


def _json_response(handler, payload, status=200):
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
    except BrokenPipeError:
        pass


def _photo_import_access_allowed(handler) -> bool:
    """Keep private photo metadata on loopback and reject non-local web origins."""
    try:
        address = str(handler.client_address[0]).split("%", 1)[0]
        if not ipaddress.ip_address(address).is_loopback:
            return False
    except (ValueError, IndexError, TypeError):
        return False

    host = handler.headers.get("Host")
    if not host:
        return False
    parsed_host = urlparse(f"//{host}")
    if parsed_host.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return False

    origin = handler.headers.get("Origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _photo_json_response(handler, payload, status=200):
    """JSON response that never grants cross-origin access to non-local sites."""
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        origin = handler.headers.get("Origin")
        if origin and _photo_import_access_allowed(handler):
            handler.send_header("Access-Control-Allow-Origin", origin)
            handler.send_header("Vary", "Origin")
        handler.end_headers()
        handler.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
    except BrokenPipeError:
        pass


def _encode_pil_to_base64(image):
    buff = BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _decode_base64_image(data_b64):
    from PIL import Image

    if data_b64 is None:
        raise ValueError("No image provided")
    if "," in data_b64:
        data_b64 = data_b64.split(",", 1)[1]
    raw = base64.b64decode(data_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _save_temp_image(img) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        img.save(f, format="PNG")
    return path


class APIServer(BaseHTTPRequestHandler):

    def handle(self):
        """Override to suppress BrokenPipeError tracebacks from disconnected clients."""
        try:
            super().handle()
        except BrokenPipeError:
            pass

    # ── CORS ────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        path = urlparse(self.path).path
        if path.startswith(("/api/v1/photo-imports", "/api/v1/photo-batches")):
            if not _photo_import_access_allowed(self):
                return _photo_json_response(self, {"error": "Local access only."}, status=403)
            self.send_response(204)
            origin = self.headers.get("Origin")
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # ── GET routing ─────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if not path.startswith("/api/") and not path.startswith("/sdapi/"):
            return self.handle_webui(path)

        # Existing SD WebUI endpoints
        if path == "/sdapi/v1/options":
            return self.handle_options()
        if path == "/sdapi/v1/sd-models":
            return self.handle_models()

        # New v1 endpoints
        if path == "/api/v1/health":
            return self.handle_health()
        if path == "/api/v1/models":
            return self.handle_v1_models()
        if path == "/api/v1/photo-imports/config":
            return self.handle_photo_import_config()
        if path == "/api/v1/system":
            return self.handle_system()
        if path == "/api/v1/providers/nativ":
            return self.handle_nativ_status()
        if path == "/api/v1/queue":
            return self.handle_queue()
        if path == "/api/v1/stats":
            return self.handle_stats()
        if path == "/api/v1/jobs":
            return self.handle_list_jobs()

        # Path-parameter routes
        m = _RE_JOB_STREAM.match(path)
        if m:
            return self.handle_job_stream(m.group(1))
        m = _RE_JOB_DETAIL.match(path)
        if m:
            return self.handle_get_job(m.group(1))

        return _bad_request(self, "Unknown endpoint", status=404)

    def handle_webui(self, path: str):
        """Serve the built React studio, falling back to its SPA entrypoint."""
        if not WEBUI_DIST.exists():
            return _json_response(self, {
                "status": "studio-not-built",
                "message": "Build frontend with npm run build in frontend/."
            }, status=503)
        candidate = (WEBUI_DIST / path.lstrip("/")).resolve()
        if not str(candidate).startswith(str(WEBUI_DIST.resolve())) or not candidate.is_file():
            candidate = WEBUI_DIST / "index.html"
        content_type, _ = mimetypes.guess_type(candidate.name)
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Cache-Control", "no-cache" if candidate.name == "index.html" else "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(candidate.read_bytes())
        except (OSError, BrokenPipeError):
            return

    # ── POST routing ────────────────────────────────────────────────

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/sdapi/v1/txt2img":
            return self.handle_txt2img()
        if path == "/sdapi/v1/img2img":
            return self.handle_img2img()
        if path == "/sdapi/v1/controlnet":
            return self.handle_controlnet()
        if path == "/api/upscale":
            return self.handle_upscale()
        if path == "/sdapi/v1/options":
            return self.handle_options_update()
        if path == "/api/v1/generate":
            return self.handle_generate()
        if path == "/api/v1/photo-imports/inventory":
            return self.handle_photo_import_inventory()
        if path == "/api/v1/photo-batches/plan":
            return self.handle_photo_batch_plan()

        return _bad_request(self, "Unknown endpoint", status=404)

    # ── DELETE routing ──────────────────────────────────────────────

    def do_DELETE(self):
        parsed = urlparse(self.path)
        m = _RE_JOB_DETAIL.match(parsed.path)
        if m:
            return self.handle_cancel_job(m.group(1))
        return _bad_request(self, "Unknown endpoint", status=404)

    # ── Shared helpers ──────────────────────────────────────────────

    def _read_json(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid JSON: {exc}") from exc

    # ── Existing SD WebUI handlers (unchanged) ──────────────────────

    def handle_txt2img(self):
        from backend.flux_manager import generate_image_gradio

        try:
            data = self._read_json()
        except Exception as exc:
            return _bad_request(self, str(exc))

        prompt = data.get("prompt")
        if not prompt:
            return _bad_request(self, "prompt is required")

        model = _resolve_model_from_payload(data)
        seed = data.get("seed")
        width = int(data.get("width", 576))
        height = int(data.get("height", 1024))
        # Accept common aliases.
        steps = str(data.get("steps") or data.get("num_inference_steps") or "")  # blank => default per model
        guidance = float(data.get("guidance") or data.get("guidance_scale") or 3.5)
        num_images = int(data.get("num_images", 1))
        auto_seeds = bool(data.get("auto_seeds", False))
        lora_files = data.get("lora_files") or None
        lora_scales = data.get("lora_scales") or []
        if lora_scales is None:
            lora_scales = []
        if not isinstance(lora_scales, list):
            lora_scales = [lora_scales]
        low_ram = bool(data.get("low_ram", False))

        try:
            images, info, used_prompt = generate_image_gradio(
                prompt,
                model,
                None,
                seed,
                width,
                height,
                steps,
                guidance,
                lora_files,
                False,
                None,
                None,
                None,
                None,
                None,
                False,
                1,
                *lora_scales,
                num_images=num_images,
                low_ram=low_ram,
                auto_seeds=auto_seeds,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"Generation failed: {exc}", status=500)

        if not images:
            return _bad_request(self, info or "No images were generated successfully", status=500)

        encoded_images = [_encode_pil_to_base64(img) for img in images]
        response = {
            "images": encoded_images,
            "parameters": data,
            "info": info or "",
            "prompt": used_prompt,
        }
        return _json_response(self, response)

    def handle_img2img(self):
        from backend.flux_manager import generate_image_i2i_gradio

        try:
            data = self._read_json()
        except Exception as exc:
            return _bad_request(self, str(exc))

        prompt = data.get("prompt")
        init_images = data.get("init_images") or data.get("images") or []
        if not prompt or not init_images:
            return _bad_request(self, "prompt and init_images are required")
        try:
            init_img = _decode_base64_image(init_images[0])
        except Exception as exc:
            return _bad_request(self, f"Invalid init image: {exc}")

        model = _resolve_model_from_payload(data)
        seed = data.get("seed")
        width = int(data.get("width", init_img.width))
        height = int(data.get("height", init_img.height))
        steps = str(data.get("steps") or data.get("num_inference_steps") or "")
        guidance = float(data.get("guidance") or data.get("guidance_scale") or 3.5)
        num_images = int(data.get("num_images", 1))
        auto_seeds = bool(data.get("auto_seeds", False))
        image_strength = float(data.get("image_strength") or data.get("denoising_strength") or 0.4)
        lora_files = data.get("lora_files") or None
        lora_scales = data.get("lora_scales") or []
        if lora_scales is None:
            lora_scales = []
        if not isinstance(lora_scales, list):
            lora_scales = [lora_scales]
        low_ram = bool(data.get("low_ram", False))

        try:
            images, info, used_prompt = generate_image_i2i_gradio(
                prompt,
                init_img,
                model,
                None,
                seed,
                height,
                width,
                steps,
                guidance,
                image_strength,
                lora_files,
                False,
                None,
                None,
                None,
                False,
                1,
                *lora_scales,
                num_images=num_images,
                low_ram=low_ram,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"Img2Img failed: {exc}", status=500)

        if not images:
            return _bad_request(self, info or "No images were generated successfully", status=500)

        encoded_images = [_encode_pil_to_base64(img) for img in images]
        response = {
            "images": encoded_images,
            "parameters": data,
            "info": info or "",
            "prompt": used_prompt,
        }
        return _json_response(self, response)

    def handle_controlnet(self):
        from backend.flux_manager import generate_image_controlnet_gradio

        try:
            data = self._read_json()
        except Exception as exc:
            return _bad_request(self, str(exc))

        prompt = data.get("prompt")
        cn_images = data.get("controlnet_image") or data.get("controlnet_images") or data.get("init_images")
        if not prompt or not cn_images:
            return _bad_request(self, "prompt and controlnet_image are required")
        try:
            controlnet_img = _decode_base64_image(cn_images[0])
        except Exception as exc:
            return _bad_request(self, f"Invalid controlnet image: {exc}")

        model = _resolve_model_from_payload(data)
        seed = data.get("seed")
        width = int(data.get("width", controlnet_img.width))
        height = int(data.get("height", controlnet_img.height))
        steps = str(data.get("steps") or data.get("num_inference_steps") or "")
        guidance = float(data.get("guidance") or data.get("guidance_scale") or 3.5)
        controlnet_strength = float(data.get("controlnet_strength", 0.4))
        lora_files = data.get("lora_files") or None
        lora_scales = data.get("lora_scales") or []
        if lora_scales is None:
            lora_scales = []
        if not isinstance(lora_scales, list):
            lora_scales = [lora_scales]
        low_ram = bool(data.get("low_ram", False))

        try:
            images, info, used_prompt = generate_image_controlnet_gradio(
                prompt,
                controlnet_img,
                model,
                None,
                seed,
                height,
                width,
                steps,
                guidance,
                controlnet_strength,
                lora_files,
                False,  # metadata
                False,  # save_canny
                None,
                None,
                None,
                False,
                1,
                *lora_scales,
                num_images=1,
                low_ram=low_ram,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"ControlNet failed: {exc}", status=500)

        if not images:
            return _bad_request(self, info or "No images were generated successfully", status=500)

        encoded_images = [_encode_pil_to_base64(img) for img in images]
        response = {
            "images": encoded_images,
            "parameters": data,
            "info": info or "",
            "prompt": used_prompt,
        }
        return _json_response(self, response)

    def handle_upscale(self):
        from backend import upscale_manager

        try:
            data = self._read_json()
        except Exception as exc:
            return _bad_request(self, str(exc))

        image_b64 = data.get("image")
        if not image_b64:
            return _bad_request(self, "image is required (base64)")
        try:
            img = _decode_base64_image(image_b64)
        except Exception as exc:
            return _bad_request(self, f"Invalid image: {exc}")

        factor = data.get("upscale_factor", 2)
        output_format = data.get("output_format", "PNG").upper()
        metadata = bool(data.get("metadata", False))

        try:
            temp_path = _save_temp_image(img)
            upscaled, status = upscale_manager.upscale_image_gradio(
                input_image=temp_path,
                upscale_factor=factor,
                output_format=output_format,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"Upscale failed: {exc}", status=500)

        if upscaled is None:
            return _bad_request(self, status or "Upscale failed", status=500)

        encoded_image = _encode_pil_to_base64(upscaled)
        response = {
            "images": [encoded_image],
            "info": status or "",
            "parameters": data,
        }
        return _json_response(self, response)

    def handle_options(self):
        """
        Minimal SD WebUI-compatible options endpoint so clients like Open WebUI
        can validate connectivity.
        """
        options = {
            "sd_model_checkpoint": _current_model(),
            "sd_model_checkpoint_hash": "",
            "sd_vae": "auto",
            "CLIP_stop_at_last_layers": 2,
            "inpainting_fill": 1,
        }
        return _json_response(self, options)

    def handle_nativ_status(self):
        """Expose only a local provider health/model summary to the studio."""
        from backend.nativ_provider import status
        return _json_response(self, status())

    def handle_options_update(self):
        """
        Accept SD WebUI-style options updates (primarily model selection).
        """
        try:
            data = self._read_json()
        except Exception as exc:
            return _bad_request(self, str(exc))

        requested_model = data.get("sd_model_checkpoint")
        if requested_model:
            _set_current_model(str(requested_model))

        # Return updated options snapshot
        return self.handle_options()

    def handle_models(self):
        """
        Minimal model list endpoint for compatibility.
        """
        return _json_response(self, _list_models_payload())

    # ── New async job endpoints ─────────────────────────────────────

    def handle_generate(self):
        """POST /api/v1/generate - submit an async generation job."""
        from backend.api_models import APIError, JobType
        from backend.job_manager import get_job_manager

        try:
            data = self._read_json()
        except Exception as exc:
            return _json_response(self, {
                "error": APIError(
                    code=APIError.INVALID_JSON,
                    message=str(exc),
                ).to_dict()
            }, status=400)

        if not isinstance(data, dict):
            return _json_response(self, {
                "error": APIError(
                    code=APIError.INVALID_JSON,
                    message="JSON body must be an object.",
                ).to_dict()
            }, status=400)

        raw_type = data.get("type", "txt2img")
        try:
            job_type = JobType(raw_type)
        except ValueError:
            return _json_response(self, {
                "error": APIError(
                    code=APIError.INVALID_PARAM,
                    message=f"Unknown job type: {raw_type}",
                ).to_dict()
            }, status=400)

        prompt = data.get("prompt", "")
        if job_type in (JobType.txt2img, JobType.img2img, JobType.controlnet) and not prompt:
            return _json_response(self, {
                "error": APIError(
                    code=APIError.MISSING_PARAM,
                    message="prompt is required",
                ).to_dict()
            }, status=400)

        response = _json_response
        if job_type == JobType.photo_batch:
            if not _photo_import_access_allowed(self):
                return _photo_json_response(self, {"error": "Local access only."}, status=403)
            from backend.photo_batch import PhotoBatchValidationError, prepare_photo_batch

            try:
                plan = prepare_photo_batch(data)
            except (PhotoBatchValidationError, ValueError) as exc:
                return _photo_json_response(self, {"error": str(exc)}, status=400)
            data = dict(data)
            data["_photo_batch_plan"] = plan
            data["num_images"] = plan["num_images"]
            response = _photo_json_response

        mgr = get_job_manager()
        job = mgr.submit_job(job_type, data)
        payload = {
            "job_id": job.id,
            "status": job.status.value,
            "type": job.job_type.value,
        }
        if job_type == JobType.photo_batch:
            payload.update({
                "batch_id": plan["batch_id"],
                "output_relative_directory": plan["output_relative_directory"],
            })
        return response(self, payload, status=202)

    def handle_get_job(self, job_id: str):
        """GET /api/v1/jobs/{id} - get job status."""
        from backend.api_models import APIError
        from backend.job_manager import get_job_manager

        mgr = get_job_manager()
        job = mgr.get_job(job_id)
        if job is None:
            return _json_response(self, {
                "error": APIError(
                    code=APIError.JOB_NOT_FOUND,
                    message=f"Job {job_id} not found",
                ).to_dict()
            }, status=404)
        if job.job_type.value == "photo_batch" and not _photo_import_access_allowed(self):
            return _photo_json_response(self, {"error": "Local access only."}, status=403)
        responder = _photo_json_response if job.job_type.value == "photo_batch" else _json_response
        return responder(self, job.to_dict())

    def handle_job_stream(self, job_id: str):
        """GET /api/v1/jobs/{id}/stream - SSE event stream."""
        from backend.api_models import APIError
        from backend.job_manager import get_job_manager
        from backend.sse_handler import stream_job_events

        mgr = get_job_manager()
        job = mgr.get_job(job_id)
        if job is None:
            return _json_response(self, {
                "error": APIError(
                    code=APIError.JOB_NOT_FOUND,
                    message=f"Job {job_id} not found",
                ).to_dict()
            }, status=404)
        if job.job_type.value == "photo_batch" and not _photo_import_access_allowed(self):
            return _photo_json_response(self, {"error": "Local access only."}, status=403)
        stream_job_events(self, job)

    def handle_cancel_job(self, job_id: str):
        """DELETE /api/v1/jobs/{id} - cancel a job."""
        from backend.api_models import APIError
        from backend.job_manager import get_job_manager

        mgr = get_job_manager()
        existing = mgr.get_job(job_id)
        if existing is not None and existing.job_type.value == "photo_batch":
            if not _photo_import_access_allowed(self):
                return _photo_json_response(self, {"error": "Local access only."}, status=403)
        job = mgr.cancel_job(job_id)
        if job is None:
            return _json_response(self, {
                "error": APIError(
                    code=APIError.JOB_NOT_FOUND,
                    message=f"Job {job_id} not found",
                ).to_dict()
            }, status=404)
        responder = _photo_json_response if job.job_type.value == "photo_batch" else _json_response
        return responder(self, {
            "job_id": job.id,
            "status": job.status.value,
        })

    def handle_list_jobs(self):
        """GET /api/v1/jobs - list all jobs."""
        from backend.job_manager import get_job_manager

        mgr = get_job_manager()
        jobs = mgr.list_jobs()
        if not _photo_import_access_allowed(self):
            jobs = [job for job in jobs if job.job_type.value != "photo_batch"]
        return _json_response(self, {
            "jobs": [j.to_dict() for j in jobs],
        })

    # ── System info endpoints ───────────────────────────────────────

    def handle_health(self):
        """GET /api/v1/health"""
        return _json_response(self, {
            "status": "ok",
            "timestamp": time.time(),
        })

    def handle_photo_import_config(self):
        """GET local photo-import limits and privacy defaults."""
        if not _photo_import_access_allowed(self):
            return _photo_json_response(self, {"error": "Local access only."}, status=403)

        from backend.photo_imports import get_photo_import_config

        return _photo_json_response(self, get_photo_import_config())

    def handle_photo_import_inventory(self):
        """POST a local directory inventory without altering source photos."""
        if not _photo_import_access_allowed(self):
            return _photo_json_response(self, {"error": "Local access only."}, status=403)

        from backend.photo_imports import PhotoImportValidationError, inventory_photos

        try:
            data = self._read_json()
            if not isinstance(data, dict):
                raise PhotoImportValidationError("JSON body must be an object.")
            result = inventory_photos(
                data.get("directory"),
                recursive=data.get("recursive", True),
                gps_mode=data.get("gps_mode", "suggest"),
                location_overrides=data.get("location_overrides"),
            )
        except (PhotoImportValidationError, ValueError) as exc:
            return _photo_json_response(self, {"error": str(exc)}, status=400)
        return _photo_json_response(self, result)

    def handle_photo_batch_plan(self):
        """Validate and preview a deterministic local SeedVR2 batch."""
        if not _photo_import_access_allowed(self):
            return _photo_json_response(self, {"error": "Local access only."}, status=403)

        from backend.photo_batch import (
            PhotoBatchValidationError,
            prepare_photo_batch,
            public_photo_batch_plan,
        )

        try:
            data = self._read_json()
            if not isinstance(data, dict):
                raise PhotoBatchValidationError("JSON body must be an object.")
            result = public_photo_batch_plan(prepare_photo_batch(data))
        except (PhotoBatchValidationError, ValueError) as exc:
            return _photo_json_response(self, {"error": str(exc)}, status=400)
        return _photo_json_response(self, result)

    def handle_v1_models(self):
        """GET /api/v1/models - models with capabilities."""
        from backend.model_manager import get_custom_model_config, get_updated_models

        models = []
        try:
            aliases = get_updated_models()
        except Exception:
            aliases = [DEFAULT_MODEL]

        for alias in aliases:
            entry = {"name": alias, "capabilities": ["txt2img"]}
            try:
                cfg = get_custom_model_config(alias)
                entry["base_arch"] = cfg.base_arch
                entry["hf_name"] = cfg.model_name
                if cfg.base_arch not in ("flux2",):
                    entry["capabilities"].append("img2img")
                    entry["capabilities"].append("controlnet")
            except Exception:
                pass
            models.append(entry)
        return _json_response(self, {"models": models})

    def handle_system(self):
        """GET /api/v1/system - memory, active model, queue depth."""
        from backend.job_manager import get_job_manager

        info = {
            "active_model": _current_model(),
            "queue_depth": get_job_manager().queue_depth(),
        }
        try:
            import mlx.core as mx
            get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
            info["memory"] = {
                "active_mb": round(get_active() / 1e6, 2),
                "peak_mb": round(get_peak() / 1e6, 2),
            }
        except Exception:
            info["memory"] = None
        return _json_response(self, info)

    def handle_queue(self):
        """GET /api/v1/queue"""
        from backend.job_manager import get_job_manager
        from backend.api_models import JobStatus

        mgr = get_job_manager()
        jobs = mgr.list_jobs()
        if not _photo_import_access_allowed(self):
            jobs = [job for job in jobs if job.job_type.value != "photo_batch"]
        pending = [j.to_dict() for j in jobs if j.status == JobStatus.queued]
        running = [j.to_dict() for j in jobs if j.status == JobStatus.running]
        return _json_response(self, {
            "pending": pending,
            "pending_count": len(pending),
            "running": running,
            "running_count": len(running),
        })

    def handle_stats(self):
        """GET /api/v1/stats"""
        from backend.job_manager import get_job_manager

        return _json_response(self, get_job_manager().get_stats())


def run_server(host: str = HOST, port: int = PORT):
    server = ThreadingHTTPServer((host, port), APIServer)
    print(f"API server running on http://{host}:{port} (txt2img/img2img/controlnet/upscale + async /api/v1)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down API server...")
        server.server_close()


if __name__ == "__main__":
    host = HOST
    port = PORT
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    run_server(host, port)
