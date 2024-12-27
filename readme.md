# MFLUX WebUI

![MFLUX WebUI Logo](logo.png)

A powerful and user-friendly web interface for MFLUX, powered by Gradio.

[![Install with Pinokio](https://img.shields.io/badge/Install%20with-Pinokio-blue)](https://pinokio.computer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## Introduction

MFLUX WebUI is a comprehensive interface for the MFLUX image generation system. It provides an intuitive way to interact with MFLUX models, offering both simple and advanced options for image generation, as well as support for ControlNet, LoRA integration, and image-to-image transformation.

## Features

- 🖼️ Simple and advanced image generation interfaces
- 🎨 **Image-to-Image Transformation** for enhancing and modifying existing images
- 🎛️ ControlNet support for guided image generation
- 🧠 LoRA (Low-Rank Adaptation) integration for fine-tuned models
- 🎯 **Dreambooth Fine-Tuning** for training custom models
- ⚙️ Model quantization options for optimized performance
- 🌈 User-friendly UI suitable for both beginners and advanced users
- 🔧 Customizable settings for precise control over image generation
- 🤖 Ollama integration for prompt enhancement
- 📥 LoRA download functionality

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
   conda create -n mflux python=3.11
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

The MFLUX WebUI consists of six main tabs:

1. **MFLUX Easy**: A simplified interface for quick image generation.
2. **Advanced Generate**: Provides full control over image generation parameters.
3. **ControlNet**: Enables guided image generation using a control image.
4. **Image-to-Image**: Transform existing images with new prompts.
5. **Dreambooth Fine-Tuning**: Train and fine-tune your own models.
6. **Models**: Allows for model management, including downloading, adding, and quantizing models.

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