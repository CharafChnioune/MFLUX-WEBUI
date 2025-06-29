from frontend.gradioui import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch(
        server_port=None,
        show_error=True
    )