# MFLUX API Documentation

## Overview

REST API for **Flux2 Klein** image generation on Apple Silicon using MLX.

This is an **API-only** version - the Gradio UI has been removed. All functionality is accessed via REST endpoints.

### Key Features
- ✅ Flux2 Klein 4B and 9B models (Flux1 removed)
- ✅ Automatic model downloading
- ✅ Quantization support (3/4/6/8-bit)
- ✅ LoRA support
- ✅ Async job management with SSE streaming
- ✅ SD-WebUI compatible endpoints
- ✅ Apple Silicon optimized (MLX)

### Base URL
```
http://localhost:7861
```

### CORS
All origins allowed (`Access-Control-Allow-Origin: *`)

---

## Quick Start

### Installation
```bash
cd MFLUX-WEBUI
pip install -r requirements.txt
```

### Start the Server
```bash
# Default (0.0.0.0:7861)
python api_main.py

# Custom host/port
python api_main.py --host 127.0.0.1 --port 8080

# Using environment variables
MFLUX_API_PORT=9000 python api_main.py
```

### First Request
```bash
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "model": "flux2-klein-4b",
    "steps": 4
  }'
```

---

## Supported Models

### Flux2 Klein Models (Flux1 removed)

| Model Alias | Description | Size | Recommended Use |
|------------|-------------|------|-----------------|
| `flux2-klein-4b` | Distilled 4B model (default) | ~4GB | Fast generation, good quality |
| `flux2-klein-4b-mlx-4bit` | Pre-quantized 4-bit | ~1GB | Fastest loading |
| `flux2-klein-4b-mlx-8bit` | Pre-quantized 8-bit | ~2GB | Good balance |
| `flux2-klein-4b-3-bit` | Runtime quantized | ~1GB | Maximum compression |
| `flux2-klein-4b-4-bit` | Runtime quantized | ~1GB | Very fast |
| `flux2-klein-4b-6-bit` | Runtime quantized | ~1.5GB | Good quality/speed |
| `flux2-klein-4b-8-bit` | Runtime quantized | ~2GB | Better quality |
| `flux2-klein-9b` | Distilled 9B model | ~9GB | Highest quality |
| `flux2-klein-9b-mlx-4bit` | Pre-quantized 4-bit | ~2.5GB | Fast, high quality |
| `flux2-klein-9b-mlx-8bit` | Pre-quantized 8-bit | ~4.5GB | Best balance |
| `flux2-klein-9b-3-bit` | Runtime quantized | ~2.5GB | Compressed high quality |
| `flux2-klein-9b-4-bit` | Runtime quantized | ~2.5GB | Fast high quality |
| `flux2-klein-9b-6-bit` | Runtime quantized | ~3.5GB | Premium quality/speed |
| `flux2-klein-9b-8-bit` | Runtime quantized | ~4.5GB | Maximum quality |
| `flux2-klein-base-4b` | Base 4B model | Varies | Allows guidance > 1.0 |
| `flux2-klein-base-4b-mlx-4bit` | Pre-quantized base 4-bit | Varies | Fastest loading (base) |
| `flux2-klein-base-4b-mlx-8bit` | Pre-quantized base 8-bit | Varies | Good balance (base) |
| `flux2-klein-base-4b-3-bit` | Runtime quantized (base) | Varies | Maximum compression (base) |
| `flux2-klein-base-4b-4-bit` | Runtime quantized (base) | Varies | Very fast (base) |
| `flux2-klein-base-4b-6-bit` | Runtime quantized (base) | Varies | Good quality/speed (base) |
| `flux2-klein-base-4b-8-bit` | Runtime quantized (base) | Varies | Better quality (base) |
| `flux2-klein-base-9b` | Base 9B model | Varies | Allows guidance > 1.0 |
| `flux2-klein-base-9b-3-bit` | Runtime quantized (base) | Varies | Compressed high quality (base) |
| `flux2-klein-base-9b-4-bit` | Runtime quantized (base) | Varies | Fast high quality (base) |
| `flux2-klein-base-9b-6-bit` | Runtime quantized (base) | Varies | Premium quality/speed (base) |
| `flux2-klein-base-9b-8-bit` | Runtime quantized (base) | Varies | Maximum quality (base) |
| `flux2-dev` | **Experimental** FLUX.2-dev (very large) | ~100GB+ | Only for high-RAM machines; experimental loader |
| `flux2-dev-4-bit` | **Experimental** runtime-quantized | Varies | Requires initial full download; reduces RAM after quantization |
| `flux2-dev-8-bit` | **Experimental** runtime-quantized | Varies | Requires initial full download; reduces RAM after quantization |
| `seedvr2` | Video/image model | Varies | Experimental |

**Note**: Distilled Flux2 Klein models use fixed guidance=1.0. Base models (and `flux2-dev`) allow guidance > 1.0.

---

## API Endpoints

### 1. Image Generation (SD-WebUI Compatible)

#### POST /sdapi/v1/txt2img

Generate images from text prompt (synchronous, blocks until complete).

**Request Body:**
```json
{
  "prompt": "a beautiful landscape with mountains and lakes",
  "model": "flux2-klein-4b",
  "seed": 42,
  "width": 512,
  "height": 512,
  "steps": 4,
  "guidance": 1.0,
  "num_images": 1,
  "lora_files": [],
  "lora_scales": [],
  "low_ram": false
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | ✅ Yes | - | Text description of desired image |
| `model` | string | No | `flux2-klein-4b` | Model to use (see Supported Models) |
| `sd_model_checkpoint` | string | No | - | Alternative model field (SD-WebUI compat) |
| `seed` | int/null | No | null | Random seed (null = random) |
| `width` | int | No | 512 | Image width (multiple of 16) |
| `height` | int | No | 512 | Image height (multiple of 16) |
| `steps` | int | No | 4 | Number of inference steps (4-20 typical) |
| `guidance` | float | No | 1.0 | Distilled models override to 1.0; base models allow guidance > 1.0 |
| `num_images` | int | No | 1 | Number of images to generate (1-4) |
| `lora_files` | array | No | [] | List of LoRA file paths |
| `lora_scales` | array | No | [] | LoRA scaling factors (0.0-2.0) |
| `low_ram` | bool | No | false | Enable low-RAM mode |

**Response (200):**
```json
{
  "images": [
    "iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "parameters": {
    "prompt": "a beautiful landscape...",
    "seed": 42,
    "steps": 4,
    "model": "flux2-klein-4b"
  },
  "info": "Generated 1 image(s) in 2.5s"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "astronaut riding a horse on mars",
    "model": "flux2-klein-4b",
    "steps": 4,
    "seed": 123
  }' | jq -r '.images[0]' | base64 -d > output.png
```

**Example (Python):**
```python
import requests
import base64
from io import BytesIO
from PIL import Image

response = requests.post(
    "http://localhost:7861/sdapi/v1/txt2img",
    json={
        "prompt": "a serene japanese garden with cherry blossoms",
        "model": "flux2-klein-4b",
        "steps": 4,
        "width": 768,
        "height": 512
    }
)

data = response.json()
image_b64 = data["images"][0]
image = Image.open(BytesIO(base64.b64decode(image_b64)))
image.save("generated.png")
print(f"Generated image: {data['info']}")
```

---

#### POST /sdapi/v1/img2img

Image-to-image generation (Flux2 Edit mode).

**Request Body:**
```json
{
  "prompt": "turn this into a watercolor painting",
  "init_images": ["base64_encoded_image"],
  "model": "flux2-klein-4b",
  "image_strength": 0.75,
  "steps": 4
}
```

**Additional Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `init_images` | array | *(required)* | Base64-encoded input images |
| `image_strength` | float | 0.75 | How much to transform (0.0-1.0) |

---

### 2. Async Job API

For long-running generations, use async endpoints with job tracking.

#### POST /api/v1/generate

Submit async generation job (returns immediately).

**Request Body:**
```json
{
  "prompt": "a futuristic city at night",
  "model": "flux2-klein-9b",
  "steps": 8,
  "num_images": 4
}
```

**Response (200):**
```json
{
  "job_id": "a3f2b8c9d4e5",
  "status": "queued",
  "position": 1
}
```

---

#### GET /api/v1/jobs

List all jobs.

**Response (200):**
```json
{
  "jobs": [
    {
      "id": "a3f2b8c9d4e5",
      "status": "generating",
      "progress": 0.75,
      "prompt": "a futuristic city..."
    }
  ]
}
```

---

#### GET /api/v1/jobs/{job_id}

Get job status.

**Response (200):**
```json
{
  "id": "a3f2b8c9d4e5",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "images": ["base64..."],
    "seeds": [42, 43, 44, 45]
  },
  "created_at": "2026-02-08T10:30:00Z",
  "completed_at": "2026-02-08T10:32:15Z"
}
```

**Status Values:**
- `queued` - Waiting in queue
- `generating` - Currently generating
- `completed` - Finished successfully
- `failed` - Error occurred
- `cancelled` - User cancelled

---

#### GET /api/v1/jobs/{job_id}/stream

Server-Sent Events (SSE) stream for real-time progress.

**Example (JavaScript):**
```javascript
const eventSource = new EventSource('/api/v1/jobs/a3f2b8c9d4e5/stream');

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Progress: ${data.progress * 100}%`);
});

eventSource.addEventListener('complete', (e) => {
  const data = JSON.parse(e.data);
  console.log('Generation complete!', data.result);
  eventSource.close();
});
```

---

#### DELETE /api/v1/jobs/{job_id}

Cancel a running job.

**Response (200):**
```json
{
  "status": "cancelled"
}
```

---

### 3. Model Management

#### GET /api/v1/models

List all available models.

**Response (200):**
```json
{
  "models": [
    {
      "title": "flux2-klein-4b",
      "model_name": "black-forest-labs/FLUX.2-klein-4B",
      "hash": "abc123...",
      "sha256": "def456...",
      "config": {
        "base_arch": "flux2",
        "supports_guidance": false,
        "max_sequence_length": 512
      },
      "downloaded": true,
      "size_gb": 4.2
    }
  ]
}
```

---

#### GET /api/v1/models/{model_id}/status

Get model download/load status.

**Response (200):**
```json
{
  "model": "flux2-klein-9b",
  "status": "ready",
  "downloaded": true,
  "loaded": false,
  "size_gb": 9.1,
  "path": "/path/to/models/flux2-klein-9b"
}
```

---

### 4. System Information

#### GET /api/v1/health

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "models_available": 14,
  "active_jobs": 1,
  "queue_length": 2
}
```

---

#### GET /api/v1/system

System status and memory info.

**Response (200):**
```json
{
  "active_model": "flux2-klein-4b",
  "mlx_memory_gb": 4.2,
  "system_memory_gb": {
    "total": 32,
    "used": 16,
    "free": 16
  },
  "device": "Apple M2 Max"
}
```

---

#### GET /api/v1/queue

Get current generation queue status.

**Response (200):**
```json
{
  "active": {
    "job_id": "a3f2b8c9d4e5",
    "prompt": "a futuristic city...",
    "progress": 0.5,
    "eta_seconds": 45
  },
  "queued": [
    {
      "job_id": "b4g3c0d1e6f7",
      "position": 1,
      "prompt": "abstract art..."
    }
  ]
}
```

---

#### GET /api/v1/stats

Performance statistics.

**Response (200):**
```json
{
  "total_generations": 156,
  "total_images": 412,
  "average_time_seconds": 3.2,
  "uptime_hours": 48.5
}
```

---

## Complete Workflow Examples

### Example 1: Generate and Download Image

```bash
# 1. Generate image
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "sunset over mountains, vibrant colors",
    "model": "flux2-klein-4b",
    "steps": 4,
    "width": 1024,
    "height": 576
  }' | jq -r '.images[0]' | base64 -d > sunset.png

echo "Image saved to sunset.png"
```

### Example 2: Async Generation with Progress Monitoring

```python
import requests
import time

# Submit job
response = requests.post(
    "http://localhost:7861/api/v1/generate",
    json={
        "prompt": "abstract geometric patterns, colorful",
        "model": "flux2-klein-4b",
        "steps": 8,
        "num_images": 4
    }
)
job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status = requests.get(f"http://localhost:7861/api/v1/jobs/{job_id}").json()

    if status["status"] == "completed":
        print("Generation complete!")
        for i, img_b64 in enumerate(status["result"]["images"]):
            with open(f"output_{i}.png", "wb") as f:
                f.write(base64.b64decode(img_b64))
        break
    elif status["status"] == "failed":
        print(f"Generation failed: {status.get('error')}")
        break

    print(f"Progress: {status['progress'] * 100:.1f}%")
    time.sleep(2)
```

### Example 3: Use Pre-Quantized Model for Speed

```bash
# Pre-quantized models load faster
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cyberpunk street scene",
    "model": "flux2-klein-4b-mlx-4bit",
    "steps": 4
  }'
```

### Example 4: Batch Generation with Different Seeds

```python
import requests
import base64

prompts = [
    "a peaceful forest scene",
    "a bustling city street",
    "an alien landscape"
]

for i, prompt in enumerate(prompts):
    response = requests.post(
        "http://localhost:7861/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "model": "flux2-klein-4b",
            "steps": 4,
            "seed": i * 100  # Different seed for each
        }
    )

    img_b64 = response.json()["images"][0]
    with open(f"batch_{i}.png", "wb") as f:
        f.write(base64.b64decode(img_b64))

    print(f"Generated: {prompt}")
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (model/job not found) |
| 500 | Internal Server Error (generation failed) |
| 503 | Service Unavailable (server overloaded) |

### Error Response Format

```json
{
  "error": "Invalid model specified",
  "code": "INVALID_MODEL",
  "details": {
    "model": "invalid-model-name",
    "available_models": ["flux2-klein-4b", "flux2-klein-9b"]
  }
}
```

### Common Errors

**Invalid Model:**
```json
{
  "error": "Model 'schnell' not found. Flux1 models removed. Use flux2-klein-4b or flux2-klein-9b",
  "code": "MODEL_NOT_FOUND"
}
```

**Out of Memory:**
```json
{
  "error": "Insufficient memory. Try using a quantized model or low_ram mode",
  "code": "OOM_ERROR",
  "details": {
    "suggestion": "Use flux2-klein-4b-mlx-4bit or set low_ram=true"
  }
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MFLUX_API_HOST` | `0.0.0.0` | Host to bind to |
| `MFLUX_API_PORT` | `7861` | Port to listen on |
| `MFLUX_DEFAULT_MODEL` | `flux2-klein-4b` | Default model |
| `MFLUX_OUTPUT_DIR` | `./output` | Output directory for images |
| `MFLUX_MODELS_DIR` | `./models` | Model cache directory |

### Config File

Copy `config.yaml.example` to `config.yaml` and customize:

```yaml
server:
  host: "0.0.0.0"
  port: 7861

models:
  default: "flux2-klein-4b"
  auto_download: true
  cache_dir: "./models"

generation:
  default_steps: 4
  max_batch_size: 4
  output_dir: "./output"
```

---

## Migration from UI Version

### UI Action → API Equivalent

| Old UI Action | New API Approach |
|---------------|------------------|
| Select model in dropdown | Set `"model": "flux2-klein-4b"` in request |
| Enter prompt in text area | Set `"prompt": "..."` in request |
| Click "Generate" button | POST to `/sdapi/v1/txt2img` |
| Adjust steps slider | Set `"steps": 8` in request |
| Upload LoRA | Use `"lora_files": ["/path/to/lora"]` |
| View output gallery | Parse `images` array from response |
| Download image | Decode base64 and save |

### Removed Features (Flux1 Only)

- ❌ ControlNet (Flux1 only)
- ❌ In-Context LoRA (Flux1 only)
- ❌ Kontext mode (Flux1 only)
- ✅ Adjustable guidance for base models (distilled models fixed at 1.0)

---

## Performance Tips

### Model Selection

- **Fast generation**: Use `flux2-klein-4b-mlx-4bit`
- **Best quality**: Use `flux2-klein-9b`
- **Balanced**: Use `flux2-klein-4b-mlx-8bit` or `flux2-klein-9b-mlx-4bit`

### Memory Optimization

```python
# Low RAM mode (slower but uses less memory)
response = requests.post(
    "http://localhost:7861/sdapi/v1/txt2img",
    json={
        "prompt": "...",
        "model": "flux2-klein-4b-mlx-4bit",
        "low_ram": True
    }
)
```

### Batch Processing

Generate multiple images by increasing `num_images` instead of making multiple API calls:

```python
# Better: Single request with batch
response = requests.post("...", json={"num_images": 4})

# Worse: Multiple requests
for i in range(4):
    response = requests.post("...", json={"num_images": 1})
```

---

## Support

- **GitHub**: [MFLUX-WEBUI Repository](https://github.com/your-repo)
- **Issues**: [Report a bug](https://github.com/your-repo/issues)
- **Documentation**: This file and `config.yaml.example`

---

## Changelog

### Version 2.0.0 (API-Only)
- ✅ Removed Gradio UI (API-only)
- ✅ Removed Flux1 models (Flux2 Klein only)
- ✅ Updated default model to `flux2-klein-4b`
- ✅ Added comprehensive API documentation
- ✅ Added `api_main.py` entry point
- ✅ Created `config.yaml.example`
- ✅ Removed Gradio dependency
