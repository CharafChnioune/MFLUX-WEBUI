import sys
import threading
import os
from pathlib import Path

# Ensure local mflux sources are available before importing any UI modules.
repo_root = Path(__file__).resolve().parent
local_mflux = repo_root / "mflux-main" / "src"
if local_mflux.exists():
    local_path = str(local_mflux)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

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
