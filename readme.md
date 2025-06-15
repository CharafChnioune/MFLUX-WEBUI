# MFLUX WebUI

![MFLUX WebUI Logo](logo.png)

A powerful and user-friendly web interface for MFLUX, powered by Gradio.

[![Install with Pinokio](https://img.shields.io/badge/Install%20with-Pinokio-blue)](https://pinokio.computer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

## Introduction

MFLUX WebUI is a comprehensive interface for the **MFLUX 0.8.0** image generation library. It provides an intuitive way to interact with MFLUX models, from one-click "easy" generation to specialized tools such as Fill, Depth, Redux, Upscale, CatVTON, IC-Edit, and Concept Attention.

## Features

- 🖼️ Simple and advanced **text-to-image** generation
- 🎨 **Image-to-Image** transformation
- 🖌️ **Fill Tool (Inpaint/Outpaint)**
- 🌊 **Depth Tool** with depth-guided generation
- 🔁 **Redux** image variation generator
- ⬆️ **Upscale** high-resolution upscaling (ControlNet-aware)
- 👕 **CatVTON** virtual try-on
- ✏️ **IC-Edit** in-context editing
- 🧩 **Concept Attention** fine-grained prompt control
- 🎛️ ControlNet support
- 🧠 Multi-LoRA support & dynamic prompt files
- ⚡ Quantization (3-,4-,6-,8-bit) & Low RAM mode
- 📤 Metadata export & auto-seed
- 🌈 User-friendly UI for beginners and experts
- 🤖 Ollama integration for prompt enhancement
- 🎯 **Dreambooth Fine-Tuning**

## Installation

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
python webui.py
```

Access the interface in your web browser at `http://localhost:7860`.

### Interface Overview

The MFLUX WebUI now contains the following tabs:

1. **MFLUX Easy** – quick text-to-image generation  
2. **Advanced Generate** – full control over generation parameters  
3. **ControlNet** – guided generation with control images  
4. **Image-to-Image** – transform existing images  
5. **Fill Tool (Inpaint/Outpaint)** – remove or extend content  
6. **Depth Tool** – depth-guided generation  
7. **Redux** – create image variations  
8. **Upscale** – intelligent upscaling  
9. **CatVTON** – virtual try-on  
10. **IC-Edit** – in-context editing  
11. **Concept Attention** – weighted prompt control  
12. **In-Context LoRA** – apply reference styles  
13. **Dreambooth Fine-Tuning** – train custom models  
14. **Model & LoRA Management** – download, quantize & manage models

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