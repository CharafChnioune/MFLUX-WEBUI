"""
Lightweight API server for MFLUX-WEBUI (SD WebUI-style).
Endpoints:
- POST /sdapi/v1/txt2img
- POST /sdapi/v1/img2img
- POST /sdapi/v1/controlnet
- POST /api/upscale             (simple factor-based upscaling)

Launch: python -m backend.api_server [host] [port]
Default: host=0.0.0.0, port=7861
"""

import base64
import json
import sys
import tempfile
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

from PIL import Image

from backend.flux_manager import (
    generate_image_gradio,
    generate_image_i2i_gradio,
    generate_image_controlnet_gradio,
)
from backend import upscale_manager
from backend.model_manager import get_updated_models, get_custom_model_config

HOST = "0.0.0.0"
PORT = 7861
DEFAULT_MODEL = "schnell-4-bit"
_CURRENT_MODEL = None


def _current_model():
    """Return the currently selected model, falling back to the first available."""
    global _CURRENT_MODEL
    if _CURRENT_MODEL:
        return _CURRENT_MODEL
    try:
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
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps({"error": message}).encode("utf-8"))


def _json_response(handler, payload, status=200):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps(payload).encode("utf-8"))


def _encode_pil_to_base64(image):
    buff = BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _decode_base64_image(data_b64) -> Image.Image:
    if data_b64 is None:
        raise ValueError("No image provided")
    if "," in data_b64:
        data_b64 = data_b64.split(",", 1)[1]
    raw = base64.b64decode(data_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _save_temp_image(img: Image.Image) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        img.save(f, format="PNG")
    return path


class APIServer(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/sdapi/v1/options":
            return self.handle_options()
        if parsed.path == "/sdapi/v1/sd-models":
            return self.handle_models()
        return _bad_request(self, "Unknown endpoint", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/sdapi/v1/txt2img":
            return self.handle_txt2img()
        if parsed.path == "/sdapi/v1/img2img":
            return self.handle_img2img()
        if parsed.path == "/sdapi/v1/controlnet":
            return self.handle_controlnet()
        if parsed.path == "/api/upscale":
            return self.handle_upscale()
        if parsed.path == "/sdapi/v1/options":
            return self.handle_options_update()
        return _bad_request(self, "Unknown endpoint", status=404)

    def _read_json(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid JSON: {exc}") from exc

    def handle_txt2img(self):
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
        steps = str(data.get("steps", ""))  # blank => default per model
        guidance = float(data.get("guidance", 3.5))
        num_images = int(data.get("num_images", 1))
        auto_seeds = bool(data.get("auto_seeds", False))
        lora_files = data.get("lora_files") or None
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
                None,
                num_images=num_images,
                low_ram=low_ram,
                auto_seeds=auto_seeds,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"Generation failed: {exc}", status=500)

        encoded_images = [_encode_pil_to_base64(img) for img in images]
        response = {
            "images": encoded_images,
            "parameters": data,
            "info": info or "",
            "prompt": used_prompt,
        }
        return _json_response(self, response)

    def handle_img2img(self):
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
        steps = str(data.get("steps", ""))
        guidance = float(data.get("guidance", 3.5))
        num_images = int(data.get("num_images", 1))
        auto_seeds = bool(data.get("auto_seeds", False))
        image_strength = float(data.get("image_strength", 0.4))
        lora_files = data.get("lora_files") or None
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
                None,
                num_images=num_images,
                low_ram=low_ram,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"Img2Img failed: {exc}", status=500)

        encoded_images = [_encode_pil_to_base64(img) for img in images]
        response = {
            "images": encoded_images,
            "parameters": data,
            "info": info or "",
            "prompt": used_prompt,
        }
        return _json_response(self, response)

    def handle_controlnet(self):
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
        steps = str(data.get("steps", ""))
        guidance = float(data.get("guidance", 3.5))
        controlnet_strength = float(data.get("controlnet_strength", 0.4))
        lora_files = data.get("lora_files") or None
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
                "horizontal",
                num_images=1,
                low_ram=low_ram,
            )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"ControlNet failed: {exc}", status=500)

        encoded_images = [_encode_pil_to_base64(img) for img in images]
        response = {
            "images": encoded_images,
            "parameters": data,
            "info": info or "",
            "prompt": used_prompt,
        }
        return _json_response(self, response)

    def handle_upscale(self):
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


def run_server(host: str = HOST, port: int = PORT):
    server = ThreadingHTTPServer((host, port), APIServer)
    print(f"API server running on http://{host}:{port} (txt2img/img2img/controlnet/upscale)")
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
