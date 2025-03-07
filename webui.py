# Main entry point for MFLUX-WEBUI
from frontend.gradioui import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch(
        server_port=None,  # Let Gradio find an available port
        server_name="0.0.0.0",  # Make accessible from any network interface
        show_error=True
    )