# MFLUX WebUI

![MFLUX Logo](https://raw.githubusercontent.com/CharafChnioune/mflux/main/src/mflux/assets/logo.png)

A powerful and user-friendly web interface for MFLUX, powered by Gradio.

[![Install with Pinokio](https://img.shields.io/badge/Install%20with-Pinokio-blue)](https://pinokio.computer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## Introduction

MFLUX WebUI is a comprehensive interface for the MFLUX image generation system. It provides an intuitive way to interact with MFLUX models, offering both simple and advanced options for image generation, as well as support for ControlNet and LoRA integration.

## Features

- 🖼️ Simple and advanced image generation interfaces
- 🎛️ ControlNet support for guided image generation
- 🧠 LoRA (Low-Rank Adaptation) integration for fine-tuned models
- ⚙️ Model quantization options for optimized performance
- 🌈 User-friendly UI suitable for both beginners and advanced users
- 🔧 Customizable settings for precise control over image generation
- 🤖 Ollama integration for prompt enhancement

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

The MFLUX WebUI consists of four main tabs:

1. **MFLUX Easy**: A simplified interface for quick image generation.
2. **Advanced Generate**: Provides full control over image generation parameters.
3. **ControlNet**: Enables guided image generation using a control image.
4. **Models**: Allows for model management, including downloading, adding, and quantizing models.

## Detailed Feature Guide

### MFLUX Easy

The MFLUX Easy tab provides a simplified interface for quick image generation:

- **Prompt**: Enter your text prompt describing the image you want to generate.
- **Enhance prompt with Ollama**: Option to improve the prompt using Ollama.
- **Model**: Choose between "schnell" (fast, lower quality) and "dev" (slow, higher quality).
- **Height/Width**: Set the dimensions of the generated image.
- **LoRA Files**: Select LoRA files to use in generation (if available).

### Advanced Generate

The Advanced Generate tab offers more control over the image generation process:

- All features from MFLUX Easy, plus:
- **Seed**: Set a specific seed for reproducible results.
- **Inference Steps**: Control the number of denoising steps.
- **Guidance Scale**: Adjust how closely the image follows the text prompt.
- **Export Metadata**: Option to export generation parameters as JSON.

### ControlNet

The ControlNet tab allows for guided image generation:

- All features from Advanced Generate, plus:
- **Control Image**: Upload an image to guide the generation process.
- **ControlNet Strength**: Adjust the influence of the control image.
- **Save Canny Edge Detection**: Option to save the edge detection result.

Note: ControlNet requires a one-time download of ~3.58GB of weights from Huggingface.

### Models

The Models tab enables you to manage models:

- **Download and Add Model**: Download models from Hugging Face and add them to available models.
- **Quantize Model**: Create optimized versions of the models:
  - Choose which model to quantize ("dev" or "schnell").
  - Select the quantization level (4-bit or 8-bit).
  - Specify where to save the quantized model.

## LoRA Integration

LoRA (Low-Rank Adaptation) allows for fine-tuned models to be used in image generation:

1. Place your LoRA files (`.safetensors`) in the `lora` directory.
2. Select the desired LoRA file(s) from the dropdown menu in the WebUI.

## Ollama Integration

MFLUX WebUI now integrates Ollama for prompt enhancement:

- Enable the "Enhance prompt with Ollama" option to automatically improve your prompts.
- The default Ollama model is set to 'qwen2.5:3b', but this can be adjusted in the code.

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