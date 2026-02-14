# MFLUX API Server (v2.0.0 - API-Only)

> **🚀 IMPORTANT: This is now an API-only version!**
>
> - ❌ **Gradio UI has been removed**
> - ✅ **REST API only** - All functionality via HTTP endpoints
> - ✅ **Flux2 Klein models only** (Flux1 removed)
> - 📖 **See [API.md](API.md) for complete API documentation**

![MFLUX WebUI Logo](logo.png)

A REST API server for Flux2 Klein image generation on Apple Silicon using MLX.

[![Install with Pinokio](https://img.shields.io/badge/Install%20with-Pinokio-blue)](https://pinokio.computer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![MFLUX](https://img.shields.io/badge/MFLUX-v0.15.4-green)](https://github.com/filipstrand/mflux)

## Quick Start (API-Only)

### Installation
```bash
pip install -r requirements.txt
```

### Run the API Server
```bash
# Default (0.0.0.0:7861)
python api_main.py

# Custom host/port
python api_main.py --host 127.0.0.1 --port 8080

# Using environment variables
MFLUX_API_PORT=9000 python api_main.py
```

### Make Your First Request
```bash
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "model": "flux2-klein-4b",
    "steps": 4
  }'
```

### API Documentation
- **Complete API Docs**: [API.md](API.md)
- **Base URL**: `http://localhost:7861`
- **Supported Models**: Flux2 Klein 4B/9B (Flux1 removed)

---

## Migration from UI Version

If you were using the UI version:

1. **UI is removed** - Use API endpoints instead
2. **Flux1 models removed** - Use `flux2-klein-4b` or `flux2-klein-9b`
3. **See [API.md](API.md)** for endpoint mapping (e.g., UI "Generate" button → `POST /sdapi/v1/txt2img`)
4. **Old UI code** archived in `legacy/` directory

---

## Supported Models (Flux2 Klein Only)

- `flux2-klein-4b` (default) - 4B base model
- `flux2-klein-4b-mlx-4bit` - Pre-quantized 4-bit
- `flux2-klein-4b-mlx-8bit` - Pre-quantized 8-bit
- `flux2-klein-9b` - 9B base model (highest quality)
- `flux2-klein-9b-mlx-4bit` - Pre-quantized 4-bit
- `flux2-klein-9b-mlx-8bit` - Pre-quantized 8-bit
- All models support runtime quantization (3/4/6/8-bit)

**Note**: Flux2 uses **fixed guidance=1.0** (not adjustable).

---

## Introduction (Legacy UI Information)

> **Note**: The information below describes the legacy UI version.
> This functionality is now available via REST API endpoints.
> See [API.md](API.md) for current usage.

MFLUX WebUI was a comprehensive interface for the **MFLUX 0.15.x** image generation library. It provided an intuitive way to interact with MFLUX models, from one-click "easy" generation to specialized tools with advanced workflow management and intelligent prompt processing.

## Features

### Core Generation Features
- 🖼️ Simple and advanced **text-to-image** generation
- 🎨 **Image-to-Image** transformation
- 🖌️ **Fill Tool (Inpaint/Outpaint)**
- 🌊 **Depth Tool** with depth-guided generation
- ⚡ **Flux2 Klein** text-to-image + multi-image edit
- **Qwen Image & Qwen Edit** for multilingual generation + edits
- **FIBO** structured prompt generation
- **Z-Image Turbo** fast text-to-image
- 🔁 **Redux** image variation generator
- ⬆️ **Upscale** high-resolution upscaling (ControlNet-aware)
- 👕 **CatVTON** virtual try-on
- ✏️ **IC-Edit** in-context editing
- 🧩 **Concept Attention** fine-grained prompt control
- 🎛️ ControlNet support
- 🎯 **Dreambooth Fine-Tuning**

### New v0.15.4 Highlights
- **Flux2 Klein (4B/9B)** support with dedicated generate + edit tabs (guidance fixed at 1.0)
- **SeedVR2 Upscale** tab integration with softness control for faithful 1-step upscaling
- **FIBO** tab with JSON prompts + optional VLM expansion
- **Z-Image Turbo** tab with LoRA + img2img support
- 🎲 **Dynamic Prompts** - Wildcard support and prompt variations (applied across all generation workflows)
- 🎯 **Auto Seeds** - Intelligent seed management and selection (shared workflow for Easy, Advanced, Canvas, ControlNet, Image-to-Image, In-Context LoRA)
- ⚙️ **Configuration Manager** - Advanced config handling with presets
- 📊 **Generation Workflow** - Comprehensive progress tracking and statistics
- 🧠 Enhanced **Multi-LoRA** support with library path management
- ⚡ Advanced **Quantization** (3-,4-,6-,8-bit) & Low RAM mode
- 📤 Enhanced **Metadata export** with workflow information
- 🔄 **Stepwise output** for generation progress visualization
- 🤖 **Third-party HuggingFace model** support
- 🌈 Modern tabbed UI for beginners and experts
- 🤖 **Ollama integration** for prompt enhancement

## Installation

### System Requirements

- macOS on Apple Silicon (MLX). Windows and Linux are not supported by MLX yet.
- Pinokio support targets macOS for this project.
- Enough disk space for models (tens of GB for multiple checkpoints).
- Sufficient RAM for the selected model/quantization level.

### Quick Start with Pinokio

Get started in seconds! Install and run MFLUX WebUI with just one click using [Pinokio](https://pinokio.computer).

### Manual Installation

If you prefer to install manually, follow these steps:

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mflux-webui.git
   cd mflux-webui
   ```

2. Create and activate a conda environment:
   ```
   conda create -n mflux python=3.12
   conda activate mflux
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the WebUI

To run the WebUI:

```
python webui.py   # starts UI + built-in API server
```

Access the interface in your web browser at `http://localhost:7860`.

#### WebUI launch options

- `MFLUX_SERVER_NAME` (default `127.0.0.1`), set to `0.0.0.0` to expose on your LAN
- `MFLUX_SERVER_PORT` or `PORT` to force a specific port (otherwise Gradio picks an open one)
- `MFLUX_OPEN_BROWSER` set to `false` to disable auto-opening the browser

#### Troubleshooting

If your browser shows a folder listing instead of the UI, the WebUI is not running or you are on the wrong port.
Run `python webui.py` in a terminal and open the exact URL that Gradio prints.
Model downloads start on the first Generate click (or via the Model & LoRA Management tab).

### Queueing

The UI runs with Gradio queueing enabled by default. To control how many jobs run in parallel:

```
export MFLUX_QUEUE_CONCURRENCY=4
export MFLUX_QUEUE_STATUS=true
```

The built-in API server also queues requests and defaults to sequential processing:

```
export MFLUX_API_QUEUE_CONCURRENCY=1
```

#### Built-in API server
- Auto-starts with the UI.
- Defaults: `http://<MFLUX_API_HOST>:<MFLUX_API_PORT>` with endpoint `/sdapi/v1/txt2img` (Stable Diffusion WebUI-style).
- Environment overrides:
  - `MFLUX_API_HOST` (default `0.0.0.0`)
  - `MFLUX_API_PORT` (default `7861`)
- Example request:
```bash
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing guitar",
    "width": 768,
    "height": 1024,
    "num_images": 1
  }'
```

## API (Stable Diffusion WebUI style)

The built-in API mirrors SD WebUI-style endpoints for automation.

- **Base URL:** `http://<MFLUX_API_HOST>:<MFLUX_API_PORT>` (defaults: `0.0.0.0:7861`)
- **Endpoints:**
  - `POST /sdapi/v1/txt2img`
    - Fields: `prompt` (required), `seed`, `width`, `height`, `steps`, `guidance`, `num_images`, `model`, `auto_seeds`, `lora_files`, `low_ram`
  - `POST /sdapi/v1/img2img`
    - Fields: `prompt` (required), `init_images` (array base64, required), `seed`, `width`, `height`, `steps`, `guidance`, `image_strength`, `num_images`, `model`, `auto_seeds`, `lora_files`, `low_ram`
  - `POST /sdapi/v1/controlnet`
    - Fields: `prompt` (required), `controlnet_image` / `controlnet_images` / `init_images` (array base64, required), `seed`, `width`, `height`, `steps`, `guidance`, `controlnet_strength`, `model`, `lora_files`, `low_ram`
  - `POST /api/upscale`
    - Fields: `image` (base64, required), `upscale_factor` (default 2), `output_format` (PNG/JPEG/WebP), `metadata` (bool)
- Model selection: pass `model` or `sd_model_checkpoint` with an alias from `GET /sdapi/v1/sd-models`.
- **Response JSON (generation endpoints):**
  - `images`: array of base64-encoded PNGs
  - `parameters`: echo of the request payload
  - `info`: text summary from the generation call
  - `prompt`: the prompt actually used (where applicable)

Run only the API (skip UI) by importing and calling `backend.api_server.run_server(host, port)` in your own launcher if needed.

### v0.13.3 Configuration

After installation, you may want to configure the new v0.13.3 features:

#### Environment Variables

Set up optional environment variables for enhanced functionality:

```bash
export LORA_LIBRARY_PATH="/path/to/your/lora/models"  # Custom LoRA library location
export MFLUX_CONFIG_PATH="/path/to/config/files"     # Custom config directory
```

#### Initial Setup

1. **First Run**: Launch the WebUI and navigate to the **Configuration** tab
2. **Dynamic Prompts**: Set up your prompt categories in the **Dynamic Prompts** tab
3. **Auto Seeds**: Configure seed management in the **Auto Seeds** tab
4. **Presets**: Create and save your preferred configuration presets

#### Optional Dependencies

For full functionality, you may want to install optional dependencies:

```bash
# For advanced image processing
pip install opencv-python scipy matplotlib

# For development
pip install black flake8 pytest
```

### Interface Overview

The MFLUX WebUI v0.13.3 contains the following tabs:

#### Core Generation Tabs
1. **MFLUX Easy** – quick text-to-image generation  
2. **Qwen Image** – text-to-image generation with the Qwen Image model (multilingual prompts, negative prompts, LoRA, quantization and optional init image)  
3. **Qwen Image Edit** – image editing based on one or more reference images using Qwen-Image-Edit (semantic + appearance editing)  
4. **FIBO** – structured prompt-based generation  
5. **Z-Image Turbo** – fast text-to-image generation  
6. **🎨 Canvas** – node-like workflow canvas for chaining tools  
7. **Advanced Generate** – full control over generation parameters  
8. **ControlNet** – guided generation with control images  
9. **Image-to-Image** – transform existing images  
10. **Fill Tool (Inpaint/Outpaint)** – remove or extend content  
11. **Depth Tool** – depth-guided generation  
12. **Redux** – create image variations  
13. **Upscale** – intelligent upscaling  
14. **CatVTON** – virtual try-on  
15. **IC-Edit** – in-context editing  
16. **Concept Attention** – weighted prompt control  
17. **In-Context LoRA** – apply reference styles  
18. **Dreambooth Fine-Tuning** – train custom models  
19. **Kontext** – context-aware generation  
20. **Model & LoRA Management** – download, quantize & manage models

#### New v0.13.3 Management & Workflow Tabs
21. **Auto Seeds** – intelligent seed management and auto-generation (global workflow for all generation tabs)
22. **Dynamic Prompts** – wildcard prompts, categories, and variations (applied in Easy, Advanced, Canvas, ControlNet, Image-to-Image and In-Context LoRA)
23. **Configuration** – advanced config management with presets

## Project Structure

```
frontend/
├── components/     # UI components for each tab
├── gradioui.py    # Main Gradio UI implementation
└── __init__.py
```

## Detailed Feature Guide

### MFLUX Easy

The MFLUX Easy tab provides a simplified interface for quick image generation:

- **Prompt**: Enter your text prompt describing the image you want to generate.
- **Enhance prompt with Ollama**: Option to improve the prompt using Ollama.
- **Model**: Choose between "schnell" (fast, lower quality) and "dev" (slow, higher quality).
- **Image Format**: Select common image dimensions.
- **LoRA Files**: Select LoRA files to use in generation (if available).

### Advanced Generate

The Advanced Generate tab offers more control over the image generation process:

- All features from MFLUX Easy, plus:
- **Seed**: Set a specific seed for reproducible results.
- **Width/Height**: Set custom dimensions for the generated image.
- **Inference Steps**: Control the number of denoising steps.
- **Guidance Scale**: Adjust how closely the image follows the text prompt.
- **Export Metadata**: Option to export generation parameters as JSON.

### ControlNet

The ControlNet tab allows for guided image generation:

- All features from Advanced Generate, plus:
- **Control Image**: Upload an image to guide the generation process.
- **ControlNet Strength**: Adjust the influence of the control image.
- **Save Canny Edge Detection Image**: Option to save the edge detection result.

Note: ControlNet requires [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny), which was trained for the `dev` model. It can work well with `schnell`, but performance is not guaranteed.

### Image-to-Image

The Image-to-Image tab allows you to transform existing images using new prompts:

- **Prompt**: Enter your text prompt describing how you want to transform the image.
- **Enhance prompt with Ollama**: Option to improve the prompt using Ollama.
- **Initial Image**: Upload the image you want to transform.
- **Init Image Strength**: Control how much the initial image influences the final result (0.0 - 1.0).
- **Model**: Choose the model to use for image transformation.
- **Seed**: Set a specific seed for reproducible results.
- **Width/Height**: Set dimensions for the generated image.
- **Inference Steps**: Control the number of denoising steps.
- **Guidance Scale**: Adjust how closely the image follows the text prompt.
- **LoRA Files**: Select LoRA files to use in transformation (if available).
- **LoRA Scale**: Adjust the influence of the LoRA files.
- **Export Metadata**: Option to export generation parameters as JSON.

## New v0.13.3 Features

### New Model Tabs

- **FIBO**: JSON-native prompting with optional VLM expansion for structured prompts.
- **Z-Image Turbo**: fast generation in ~9 steps with LoRA and img2img support.

### Auto Seeds Management

The Auto Seeds tab provides intelligent seed management:

- **Enable Auto Seeds**: Toggle automatic seed generation
- **Current Pool**: View and manage the current seed collection
- **Pool Size**: Configure how many seeds to maintain
- **Generation Strategy**: Choose between random, sequential, or weighted selection
- **Seed History**: Track previously used seeds and their results
- **Export/Import**: Save and load seed collections
- **Statistics**: View seed usage analytics and performance metrics

### Dynamic Prompts

The Dynamic Prompts tab enables advanced prompt processing:

- **Enable Dynamic Prompts**: Toggle dynamic prompt processing
- **Wildcard Support**: Use `{category}` syntax for random selections
- **Prompt Variations**: Generate multiple variations from templates
- **Categories Management**: Create and manage prompt categories
- **Template Library**: Save and reuse prompt templates
- **File Import/Export**: Load prompts from txt/json files
- **Random Selection**: Configure randomization settings
- **Prompt Testing**: Preview variations before generation

**Example wildcards:**
```
A {adjective} {animal} in a {environment}, {style}
```
Where categories contain:
- `adjective`: [beautiful, majestic, mysterious, elegant]
- `animal`: [cat, wolf, eagle, dragon]
- `environment`: [forest, mountain, ocean, city]
- `style`: [digital art, oil painting, photograph]

### Configuration Manager

The Configuration Manager tab offers comprehensive settings control:

- **Current Config**: View all active configuration settings
- **Quick Settings**: Apply common configuration presets
- **Config Sections**: Manage settings by category:
  - Generation parameters
  - System settings
  - LoRA configuration
  - Auto seeds settings
  - Battery management
  - Quantization options
- **Presets**: Save and load custom configuration presets
- **Import/Export**: Share configurations via JSON/YAML files
- **Template Export**: Create configuration templates
- **Validation**: Automatic config validation and error checking
- **Reset Options**: Restore default settings

### Enhanced Generation Workflow

v0.13.3 introduces a comprehensive generation workflow system:

- **Pre-generation Checks**: Validate settings before starting
- **Progress Monitoring**: Real-time generation progress tracking
- **Smart Pause/Resume**: Intelligent generation control
- **Enhanced Metadata**: Comprehensive generation information
- **Statistics Tracking**: Monitor generation performance
- **Error Recovery**: Robust error handling and recovery
- **Resource Management**: Optimized memory and GPU usage

### Dreambooth Fine-Tuning

The Dreambooth Fine-Tuning tab allows you to train and customize models:

- Train new models using your own images
- Fine-tune existing models for specific styles or subjects
- Customize training parameters and configurations
- Monitor training progress

### Models

The Models tab enables you to manage models:

- **Download LoRA**: Download LoRA models directly from within the interface.
- **Download and Add Model**: Download models from Hugging Face and add them to available models.
- **Quantize Model**: Create optimized versions of the models:
  - Choose which model to quantize ("dev" or "schnell").
  - Select the quantization level (4-bit or 8-bit).
  - View the quantization output in a dedicated textbox.

### Fill Tool (Inpaint/Outpaint)

The Fill Tool lets you remove objects or extend the canvas of an existing image.

- **Input Image** with optional **Mask Image** indicating the region to fill/outpaint  
- **Prompt** describing what should appear in the masked / empty area  
- **Width / Height**, **Steps**, **Guidance Scale** sliders for fine-tuning  
- Supports **Low-RAM mode**, **3/4/6/8-bit quantization**, and **metadata export**

### Depth Tool

Generate images guided by the depth map extracted from a reference photo.

- **Reference Image** and text **Prompt**  
- **ControlNet Strength** slider controls depth influence  
- Option to save the extracted depth map alongside the result

### Redux

Create multiple stylistic variations of a reference image in a single click.

- **Redux Strength** slider determines how different the variations are  
- Specify **number of variations**, seed, steps, and guidance

### Upscale

Increase resolution up to 4× while preserving detail.

- Accepts any **Input Image** (optionally with a descriptive **Prompt**)  
- Intelligent patch-based upscaling, ControlNet-aware for advanced guidance  
- Adjustable **Upscale Factor**, **Steps**, **Sharpen Strength**, and more

### CatVTON (Virtual Try-On)

Virtually dress a person in a garment image.

- **Person Image + Mask** identifying clothing region  
- **Garment Image** to apply  
- Auto-generated prompt if none provided  
- Outputs both a diptych preview and the final try-on image

### IC-Edit (In-Context Editing)

Edit an image using a natural-language instruction.

- **Reference Image** and an **Instruction Prompt** (e.g. "make it snowy")  
- Generate one or multiple edited variants  
- Full control over seed, steps, guidance, and quantization

### Concept Attention

Apply weighted concepts directly in your prompt using the `{concept:weight}` syntax.

- Supports multiple weighted tags per prompt  
- Displays parsed concepts and weights before generation  
- Works with any model or LoRA combo

### In-Context LoRA

Apply the style of a reference image on-the-fly without external LoRA files.

- **Reference Image** is encoded into a temporary LoRA  
- Adjustable strength slider; combine with standard LoRAs for unique results

## LoRA Integration

LoRA (Low-Rank Adaptation) allows for fine-tuned models to be used in image generation:

1. Place your LoRA files (`.safetensors`) in the `lora` directory.
2. Select the desired LoRA file(s) from the dropdown menu in the WebUI.
3. Use the LoRA download functionality in the Models tab to easily add new LoRA models.

## Ollama Integration

MFLUX WebUI integrates Ollama for prompt enhancement:

- Enable the "Enhance prompt with Ollama" option to automatically improve your prompts.
- Adjust Ollama settings, including the model and system prompt, in the Ollama Settings section.

## Changelog

### Qwen v3 Enhancements (Tag: `qwen3`)

This tag introduces a richer Qwen workflow on top of the existing MFLUX WebUI:

- **Qwen Image tab**
  - Text-to-image generation with the Qwen Image model
  - Supports multilingual prompts (Chinese and English), negative prompts, LoRA, quantization, seeds and multiple images per batch
  - Optional init image input for Qwen-based image-to-image generation

- **Qwen Image Edit tab**
  - Uses `Qwen-Image-Edit` for semantic and appearance editing:
    - Semantic edits: IP creation, viewpoint changes (90°/180° rotations), style transfer, background/clothing changes
    - Appearance edits: add/remove elements while preserving the rest of the image
    - Precise text editing in English and Chinese (posters, signs, etc.)
  - Supports:
    - Multiple reference images
    - LoRA integration and quantization
    - Seed control and metadata export

- **Chained editing workflow**
  - After running Qwen Image Edit, the **Use last output as input** button reuses the latest edited images as new reference images
  - Enables step-by-step refinement workflows (e.g. progressively correcting calligraphy or fine-grained local changes)

- **Region-based editing (crop helper)**
  - A dedicated **Preview / Crop (optional)** image area in the Qwen Edit tab allows selecting a region with the selection tool
  - The selected crop is saved and used as the sole reference image for the next Qwen-Image-Edit call
  - This makes it easy to target specific regions (e.g. a single character, logo, or object) for precise edits

These enhancements sit on top of the standard MFLUX Qwen integration and do not change the underlying Qwen models; they add a more powerful and user-friendly workflow layer around Qwen-Image and Qwen-Image-Edit.

### v0.13.3 (Latest)

**Highlights:**
- Updated to MFLUX v0.13.3 (FIBO + Z-Image Turbo support)
- Added FIBO and Z-Image Turbo tabs
- Refreshed ControlNet/Depth/Upscale integrations for the latest MFLUX APIs
- Improved API model selection for Open WebUI clients
- Updated Dreambooth trainer to the new DreamBooth modules

### v0.9.1 - Dependency & UI Improvements

**🛠️ Dependency Fixes**
- Restricted MLX dependency upper bound to 0.26.1 (mlx>=0.22.0,<=0.26.1) to prevent incompatibility issues

**🎨 Inpaint Mask Tool Improvements**
- Enhanced interactive inpaint masking tool with additional shape options (ellipse, rectangle, and free-hand drawing)
- Added eraser mode for precise mask corrections
- Implemented undo/redo history for non-destructive editing when crafting masks

**👩‍💻 Developer Experience**
- Introduced initial mypy static-type checking configuration
- Upgraded pre-commit hooks and addressed lint warnings

### v0.9.0 - Major Feature Update

#### New Features
- 🎲 **Dynamic Prompts**: Wildcard support with categories and templates
- 🎯 **Auto Seeds**: Intelligent seed management and selection
- ⚙️ **Configuration Manager**: Advanced config handling with presets
- 📊 **Generation Workflow**: Comprehensive progress tracking
- 🔄 **Stepwise Output**: Real-time generation progress visualization
- 🤖 **Third-party HuggingFace Model Support**: Expanded model compatibility
- 📤 **Enhanced Metadata**: Workflow information in exports

#### Improvements
- Enhanced Multi-LoRA support with library path management
- Advanced quantization options (3-,4-,6-,8-bit)
- Improved error handling and recovery
- Better resource management and memory optimization
- Modern tabbed UI with better organization
- Comprehensive statistics and analytics

#### Technical Changes
- Updated to MFLUX v0.9.1
- New modular backend architecture
- Enhanced workflow management system
- Improved configuration validation
- Better integration with external dependencies

#### Bug Fixes
- Fixed memory leaks in generation workflow
- Improved stability with multiple simultaneous generations
- Better error messages and user feedback
- Enhanced cleanup and resource management

### v0.8.0 - Previous Release
- Base MFLUX WebUI functionality
- Core generation features
- LoRA and ControlNet support
- Basic model management

## Contributing

We welcome contributions to MFLUX WebUI! If you have suggestions for improvements or encounter any issues, please feel free to:

1. Open an issue
2. Submit a pull request

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License, Version 2.0.

### FLUX Models License

The FLUX models (black-forest-labs/FLUX.1) used in this project are subject to their own license terms. Please refer to the [black-forest-labs/FLUX.1 license](https://huggingface.co/black-forest-labs/FLUX.1/blob/main/LICENSE.md) for more information on the usage and distribution of these models.

---

Get started with MFLUX WebUI in seconds using [Pinokio](https://pinokio.computer) - the easiest way to install and run AI applications!
