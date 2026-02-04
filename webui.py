import sys
import threading
import os
import subprocess
import importlib
import platform
from pathlib import Path

# Ensure local mflux sources are available before importing any UI modules.
repo_root = Path(__file__).resolve().parent
local_mflux = repo_root / "mflux-main" / "src"
if local_mflux.exists():
    local_path = str(local_mflux)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
    print("[Setup] MFLUX WebUI requires macOS on Apple Silicon (MLX). Windows/Linux are not supported.")
    sys.exit(1)


def _ensure_gradio():
    """
    Ensure gradio is available. If missing, attempt to install from requirements.txt,
    otherwise install the pinned minimum version.
    """
    def _pip_install(args):
        """
        Run pip with PEP 668-friendly settings. If direct install fails, create a
        local venv (.venv) and install there, then add it to sys.path.
        """
        base_cmd = [sys.executable, "-m", "pip"] + args
        env = os.environ.copy()
        env.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
        try:
            subprocess.check_call(base_cmd, env=env)
            return
        except subprocess.CalledProcessError:
            pass

        # Fallback: create a repo-local virtualenv and install there
        venv_dir = repo_root / ".venv"
        venv_python = venv_dir / "bin" / "python3"
        if not venv_python.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call([str(venv_python), "-m", "pip", "install"] + args[1:])

        site_packages = venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if site_packages.exists():
            sys.path.insert(0, str(site_packages))

    try:
        return importlib.import_module("gradio")
    except ImportError:
        print("[Setup] gradio not found. Attempting installation...")
        req_path = repo_root / "requirements.txt"
        try:
            if req_path.exists():
                _pip_install(["install", "-r", str(req_path)])
            else:
                _pip_install(["install", "gradio>=5.35.0"])
        except Exception as exc:
            print(f"[Setup] Automatic gradio install failed: {exc}")
            print("[Setup] Please run: python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt")
            raise
        return importlib.import_module("gradio")

# Ensure gradio is ready before the UI import
_ensure_gradio()

from frontend.gradioui import create_ui
from backend.api_server import run_server as run_api_server
from backend.job_manager import get_job_manager

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

    # Start async job manager (worker + cleanup threads)
    get_job_manager()
    print("[API] Job manager started (async /api/v1/generate endpoint ready)")

    demo, theme, custom_css = create_ui()
    try:
        queue_concurrency = int(os.environ.get("MFLUX_QUEUE_CONCURRENCY", "4"))
    except ValueError:
        queue_concurrency = 4
    queue_status = os.environ.get("MFLUX_QUEUE_STATUS", "true").lower() in {"1", "true", "yes", "on"}

    def _parse_env_port(name: str):
        value = os.environ.get(name)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            print(f"[Setup] Invalid {name} value: {value!r}. Using default.")
            return None

    server_name = os.environ.get("MFLUX_SERVER_NAME", "127.0.0.1")
    server_port = _parse_env_port("MFLUX_SERVER_PORT")
    if server_port is None:
        server_port = _parse_env_port("PORT")

    open_browser = os.environ.get("MFLUX_OPEN_BROWSER", "true").lower() in {"1", "true", "yes", "on"}
    if server_port:
        print(f"[UI] Using server port {server_port} (override).")
    print("[UI] Launching WebUI. Open the URL Gradio prints below. Models download on first generation.")

    demo.queue(default_concurrency_limit=queue_concurrency).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        inbrowser=open_browser,
        theme=theme,
        css=custom_css,
    )
