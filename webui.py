import sys
from pathlib import Path

# Ensure local mflux sources are available before importing any UI modules.
repo_root = Path(__file__).resolve().parent
local_mflux = repo_root / "mflux-main" / "src"
if local_mflux.exists():
    local_path = str(local_mflux)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from frontend.gradioui import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch(
        server_port=None,
        show_error=True
    )
