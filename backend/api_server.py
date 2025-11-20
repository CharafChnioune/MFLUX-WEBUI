"""
Minimal Stable Diffusion WebUI-compatible API (txt2img) for MFLUX-WEBUI.
Endpoint: POST /sdapi/v1/txt2img
Request body (JSON): prompt (required), optional: seed, width, height, steps, guidance, num_images, model, auto_seeds, lora_files, low_ram.
Response: {"images": [base64_png_strings], "parameters": <input_json>, "info": <text_info>}
Launch: python -m backend.api_server [host] [port]
Default: host=0.0.0.0, port=7861
"""

import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

from backend.flux_manager import generate_image_gradio

HOST = "0.0.0.0"
PORT = 7861


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


class APIServer(BaseHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/sdapi/v1/txt2img":
            return self.handle_txt2img()
        return _bad_request(self, "Unknown endpoint", status=404)

    def handle_txt2img(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _bad_request(self, f"Invalid JSON: {exc}")

        prompt = data.get("prompt")
        if not prompt:
            return _bad_request(self, "prompt is required")

        # Map incoming params to flux_manager.generate_image_gradio
        model = data.get("model") or "schnell-4-bit"
        seed = data.get("seed")
        width = int(data.get("width", 576))
        height = int(data.get("height", 1024))
        steps = str(data.get("steps", ""))  # keep optional textbox semantics
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
                False,  # metadata export
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


def run_server(host: str = HOST, port: int = PORT):
    server = ThreadingHTTPServer((host, port), APIServer)
    print(f"API server running on http://{host}:{port} (endpoint: /sdapi/v1/txt2img)")
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
