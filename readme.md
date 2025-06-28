# MFLUX WebUI v0.9.0

![MFLUX WebUI Logo](logo.png)

A powerful and user-friendly web interface for MFLUX, powered by Gradio. Now with advanced workflow management, battery monitoring, dynamic prompts, and comprehensive configuration support!

[![Install with Pinokio](https://img.shields.io/badge/Install%20with-Pinokio-blue)](https://pinokio.computer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![MFLUX](https://img.shields.io/badge/MFLUX-v0.9.0-green)](https://github.com/filipstrand/mflux)

## Introduction

MFLUX WebUI is a comprehensive interface for the **MFLUX 0.9.0** image generation library. It provides an intuitive way to interact with MFLUX models, from one-click "easy" generation to specialized tools with advanced workflow management, battery monitoring, and intelligent prompt processing.

## Features

### Core Generation Features
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
- 🎯 **Dreambooth Fine-Tuning**

### New v0.9.0 Features
- 🔋 **Battery Monitor** - Smart generation control based on battery level
- 🎲 **Dynamic Prompts** - Wildcard support and prompt variations
- 🎯 **Auto Seeds** - Intelligent seed management and selection
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

### v0.9.0 Configuration

After installation, you may want to configure the new v0.9.0 features:

#### Environment Variables

Set up optional environment variables for enhanced functionality:

```bash
export LORA_LIBRARY_PATH="/path/to/your/lora/models"  # Custom LoRA library location
export MFLUX_CONFIG_PATH="/path/to/config/files"     # Custom config directory
export MFLUX_BATTERY_MONITOR=true                    # Enable battery monitoring by default
```

#### Initial Setup

1. **First Run**: Launch the WebUI and navigate to the **Configuration** tab
2. **Battery Setup**: If using a laptop, configure the **Battery Monitor** tab with your preferred thresholds
3. **Dynamic Prompts**: Set up your prompt categories in the **Dynamic Prompts** tab
4. **Auto Seeds**: Configure seed management in the **Auto Seeds** tab
5. **Presets**: Create and save your preferred configuration presets

#### Optional Dependencies

For full functionality, you may want to install optional dependencies:

```bash
# For advanced image processing
pip install opencv-python scipy matplotlib

# For development
pip install black flake8 pytest
```

### Interface Overview

The MFLUX WebUI v0.9.0 contains the following tabs:

#### Core Generation Tabs
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
15. **Kontext** – context-aware generation

#### New v0.9.0 Management Tabs
16. **Auto Seeds** – intelligent seed management and auto-generation
17. **Dynamic Prompts** – wildcard prompts, categories, and variations
18. **Battery Monitor** – battery-aware generation control
19. **Configuration** – advanced config management with presets

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

## New v0.9.0 Features

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

### Battery Monitor

The Battery Monitor tab provides intelligent power management:

- **Enable Monitoring**: Toggle battery-aware generation control
- **Stop Threshold**: Set battery percentage to stop generation
- **Pause Threshold**: Set battery percentage to pause generation
- **Resume on AC**: Automatically resume when plugged in
- **Notifications**: Enable desktop notifications for battery events
- **Status Display**: Real-time battery level and charging status
- **Generation History**: Track power consumption during generation
- **Power Profile**: Optimize settings based on power source

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

v0.9.0 introduces a comprehensive generation workflow system:

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

### v0.9.0 (Latest) - Major Feature Update

#### New Features
- 🔋 **Battery Monitor**: Smart power management for laptop users
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
- Updated to MFLUX v0.9.0
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