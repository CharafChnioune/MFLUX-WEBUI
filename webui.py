import sys
import threading
import os
import subprocess
import importlib
from pathlib import Path

# Ensure local mflux sources are available before importing any UI modules.
repo_root = Path(__file__).resolve().parent
local_mflux = repo_root / "mflux-main" / "src"
if local_mflux.exists():
    local_path = str(local_mflux)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)


def _ensure_gradio():
    """
    Ensure gradio is available. If missing, attempt to install from requirements.txt,
    otherwise install the pinned minimum version.
    """
    try:
        return importlib.import_module("gradio")
    except ImportError:
        print("[Setup] gradio not found. Attempting installation...")
        req_path = repo_root / "requirements.txt"
        try:
            if req_path.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=5.35.0"])
        except Exception as exc:
            print(f"[Setup] Automatic gradio install failed: {exc}")
            print("[Setup] Please run: pip install -r requirements.txt")
            raise
        return importlib.import_module("gradio")

# Ensure gradio is ready before the UI import
_ensure_gradio()

from frontend.gradioui import create_ui
from backend.api_server import run_server as run_api_server

if __name__ == "__main__":
    # Start lightweight API server (SD WebUI-compatible txt2img) alongside the UI
    api_host = os.environ.get("MFLUX_API_HOST", "0.0.0.0")
    try:
        api_port = int(os.environ.get("MFLUX_API_PORT", "7861"))
    except ValueError:
        api_port = 7861

    def _start_api():
        try:
            run_api_server(api_host, api_port)
        except Exception as e:
            # Don't break the UI if API fails; just log
            print(f"[API] Failed to start API server: {e}")

    threading.Thread(target=_start_api, name="api-server", daemon=True).start()
    print(f"[API] Starting API server on http://{api_host}:{api_port} (endpoint: /sdapi/v1/txt2img)")

    demo = create_ui()
    demo.queue().launch(
        server_port=None,
        show_error=True
    )
