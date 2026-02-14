#!/usr/bin/env python3
"""
MFLUX API Server - Flux2 Image Generation REST API
Runs API-only mode without Gradio UI.

Usage:
    python api_main.py [--host HOST] [--port PORT]

Default:
    host=0.0.0.0, port=7861

Environment Variables:
    MFLUX_API_HOST - Override default host
    MFLUX_API_PORT - Override default port
    MFLUX_DEFAULT_MODEL - Set default model (default: flux2-klein-4b)
    MFLUX_OUTPUT_DIR - Set output directory
    MFLUX_MODELS_DIR - Set models cache directory

Examples:
    # Start with defaults
    python api_main.py

    # Custom host and port
    python api_main.py --host 127.0.0.1 --port 8080

    # Using environment variables
    MFLUX_API_PORT=9000 python api_main.py
"""

import os
import sys
import argparse
from backend.api_server import run_server

def main():
    parser = argparse.ArgumentParser(
        description="MFLUX API Server - Flux2 Klein Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  MFLUX_API_HOST       Override default host (default: 0.0.0.0)
  MFLUX_API_PORT       Override default port (default: 7861)
  MFLUX_DEFAULT_MODEL  Set default model (default: flux2-klein-4b)
  MFLUX_OUTPUT_DIR     Set output directory
  MFLUX_MODELS_DIR     Set models cache directory

Supported Models:
  - flux2-klein-4b (default)
  - flux2-klein-4b-mlx-4bit (pre-quantized)
  - flux2-klein-4b-mlx-8bit (pre-quantized)
  - flux2-klein-9b
  - flux2-klein-9b-mlx-4bit (pre-quantized)
  - flux2-klein-9b-mlx-8bit (pre-quantized)

API Documentation:
  See API.md for complete endpoint documentation and examples.

  Main endpoints:
    POST /sdapi/v1/txt2img     - Generate images (SD-WebUI compatible)
    POST /api/v1/generate      - Async generation with job tracking
    GET  /api/v1/models        - List available models
    GET  /api/v1/health        - Health check
        """
    )

    parser.add_argument(
        "--host",
        default=os.getenv("MFLUX_API_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MFLUX_API_PORT", "7861")),
        help="Port to bind to (default: 7861)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MFLUX API Server - Flux2 Klein Image Generation")
    print("=" * 70)
    print(f"Server Address: http://{args.host}:{args.port}")
    print(f"API Version: v1")
    print(f"Supported Models: Flux2 Klein (4B and 9B variants only)")
    print(f"Documentation: See API.md for endpoint details")
    print("-" * 70)
    print(f"Default Model: {os.getenv('MFLUX_DEFAULT_MODEL', 'flux2-klein-4b')}")
    print(f"Output Directory: {os.getenv('MFLUX_OUTPUT_DIR', './output')}")
    print(f"Models Directory: {os.getenv('MFLUX_MODELS_DIR', './models')}")
    print("=" * 70)
    print("\nStarting server...")
    print("Press Ctrl+C to stop\n")

    try:
        run_server(args.host, args.port)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
