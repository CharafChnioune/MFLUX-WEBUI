# MFLUX-WEBUI API Documentation

## Overview

MFLUX-WEBUI exposes a REST API on `http://localhost:7861` (configurable via command-line arguments).

- **CORS**: All origins allowed (`Access-Control-Allow-Origin: *`)
- **Content-Type**: `application/json` for all request/response bodies
- **Methods**: `GET`, `POST`, `DELETE`, `OPTIONS`

The API has two layers:

1. **SD WebUI-compatible endpoints** — synchronous (blocking) endpoints compatible with Stable Diffusion WebUI clients such as Open WebUI.
2. **Async Job API** — non-blocking endpoints that return a job ID immediately and stream progress via Server-Sent Events (SSE).

---

## SD WebUI-Compatible Endpoints

These endpoints block until the operation completes and return the result directly.

### POST /sdapi/v1/txt2img

Generate images from a text prompt.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *(required)* | Text prompt for generation |
| `model` | string | current model | Model alias to use |
| `sd_model_checkpoint` | string | current model | Alternative model field (SD WebUI compat) |
| `override_settings.sd_model_checkpoint` | string | — | Nested model override |
| `seed` | int/null | null | Random seed (null = random) |
| `width` | int | 576 | Image width in pixels |
| `height` | int | 1024 | Image height in pixels |
| `steps` | int/string | "" (model default) | Number of inference steps |
| `guidance` | float | 3.5 | Guidance scale |
| `num_images` | int | 1 | Number of images to generate |
| `auto_seeds` | bool | false | Auto-increment seeds for batch |
| `lora_files` | array/null | null | LoRA file paths to apply |
| `low_ram` | bool | false | Enable low-RAM mode |

**Response (200):**

```json
{
  "images": ["<base64-png>", ...],
  "parameters": { ... },
  "info": "...",
  "prompt": "actual prompt used"
}
```

---

### POST /sdapi/v1/img2img

Generate images from a text prompt + input image.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *(required)* | Text prompt |
| `init_images` | array | *(required)* | Array of base64-encoded images (first is used) |
| `images` | array | — | Alternative field for `init_images` |
| `model` | string | current model | Model alias |
| `seed` | int/null | null | Random seed |
| `width` | int | input image width | Output width |
| `height` | int | input image height | Output height |
| `steps` | int/string | "" | Inference steps |
| `guidance` | float | 3.5 | Guidance scale |
| `num_images` | int | 1 | Number of images |
| `auto_seeds` | bool | false | Auto-increment seeds |
| `image_strength` | float | 0.4 | How much to preserve the input image (0.0–1.0) |
| `lora_files` | array/null | null | LoRA files |
| `low_ram` | bool | false | Low-RAM mode |

**Response (200):** Same format as txt2img.

---

### POST /sdapi/v1/controlnet

Generate images using a ControlNet conditioning image.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *(required)* | Text prompt |
| `controlnet_image` | array | *(required)* | Base64-encoded ControlNet images |
| `controlnet_images` | array | — | Alternative field |
| `init_images` | array | — | Alternative field |
| `model` | string | current model | Model alias |
| `seed` | int/null | null | Random seed |
| `width` | int | input image width | Output width |
| `height` | int | input image height | Output height |
| `steps` | int/string | "" | Inference steps |
| `guidance` | float | 3.5 | Guidance scale |
| `controlnet_strength` | float | 0.4 | ControlNet conditioning strength (0.0–1.0) |
| `lora_files` | array/null | null | LoRA files |
| `low_ram` | bool | false | Low-RAM mode |

**Response (200):** Same format as txt2img.

---

### POST /api/upscale

Upscale an image by a given factor.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | *(required)* | Base64-encoded image |
| `upscale_factor` | int | 2 | Upscale factor |
| `output_format` | string | "PNG" | Output format (e.g. "PNG", "JPEG") |
| `metadata` | bool | false | Include metadata in output |

**Response (200):**

```json
{
  "images": ["<base64-png>"],
  "info": "...",
  "parameters": { ... }
}
```

---

### GET /sdapi/v1/options

Get current server options (SD WebUI compatibility).

**Response (200):**

```json
{
  "sd_model_checkpoint": "schnell-4-bit",
  "sd_model_checkpoint_hash": "",
  "sd_vae": "auto",
  "CLIP_stop_at_last_layers": 2,
  "inpainting_fill": 1
}
```

---

### POST /sdapi/v1/options

Update server options (primarily model selection).

**Request body:**

| Field | Type | Description |
|-------|------|-------------|
| `sd_model_checkpoint` | string | Model alias to select |

**Response (200):** Returns the updated options (same format as GET).

---

### GET /sdapi/v1/sd-models

List available models in SD WebUI format.

**Response (200):**

```json
[
  {
    "title": "schnell-4-bit",
    "model_name": "mlx-community/...",
    "hash": "",
    "sha256": "",
    "filename": "schnell-4-bit",
    "config": "flux1"
  }
]
```

---

## Async Job API

Non-blocking endpoints. Submit a job, get an ID back, and track progress via polling or SSE.

### POST /api/v1/generate

Submit an async generation job.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | "txt2img" | Job type: `txt2img`, `img2img`, `controlnet`, `upscale` |
| `prompt` | string | *(required for txt2img/img2img/controlnet)* | Text prompt |
| All other fields | — | — | Same as the corresponding sync endpoint |

**Response (202):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "type": "txt2img"
}
```

---

### GET /api/v1/jobs/{id}

Get job status and result.

**Response (200):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "type": "txt2img",
  "status": "running",
  "progress": {
    "current_image": 1,
    "total_images": 2,
    "percent": 50.0,
    "stage": "generating"
  },
  "created_at": 1700000000.0,
  "started_at": 1700000001.0,
  "completed_at": null,
  "result": null,
  "error": null
}
```

When completed, `result` contains:

```json
{
  "images": ["<base64-png>", ...],
  "info": "...",
  "prompt": "actual prompt used"
}
```

When failed, `error` contains:

```json
{
  "code": "GENERATION_FAILED",
  "message": "Error description",
  "details": "full traceback",
  "stage": "generating"
}
```

---

### GET /api/v1/jobs/{id}/stream

Stream job events via Server-Sent Events (SSE).

**Response headers:**

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

The stream immediately sends the current status and replays any existing log lines. A heartbeat comment (`: heartbeat`) is sent every 15 seconds to keep the connection alive. The stream closes automatically when the job reaches a terminal state (`completed`, `failed`, or `cancelled`).

See [SSE Event Types](#sse-event-types) for event details.

---

### DELETE /api/v1/jobs/{id}

Cancel a running or queued job.

**Response (200):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "cancelled"
}
```

---

### GET /api/v1/jobs

List all jobs.

**Response (200):**

```json
{
  "jobs": [
    {
      "job_id": "a1b2c3d4e5f6",
      "type": "txt2img",
      "status": "completed",
      "progress": { ... },
      "created_at": 1700000000.0,
      "started_at": 1700000001.0,
      "completed_at": 1700000030.0,
      "result": { ... }
    }
  ]
}
```

---

## System Endpoints

### GET /api/v1/health

Health check.

**Response (200):**

```json
{
  "status": "ok",
  "timestamp": 1700000000.0
}
```

---

### GET /api/v1/models

List available models with capabilities.

**Response (200):**

```json
{
  "models": [
    {
      "name": "schnell-4-bit",
      "capabilities": ["txt2img", "img2img", "controlnet"],
      "base_arch": "flux1",
      "hf_name": "mlx-community/..."
    }
  ]
}
```

> Note: Models with `base_arch: "flux2"` only support `txt2img`.

---

### GET /api/v1/system

System information including memory usage and queue depth.

**Response (200):**

```json
{
  "active_model": "schnell-4-bit",
  "queue_depth": 0,
  "memory": {
    "active_mb": 1234.56,
    "peak_mb": 2345.67
  }
}
```

`memory` is `null` when MLX Metal memory info is not available.

---

### GET /api/v1/queue

Current queue status.

**Response (200):**

```json
{
  "pending": [ { ...job... } ],
  "pending_count": 1,
  "running": [ { ...job... } ],
  "running_count": 1
}
```

---

### GET /api/v1/stats

Job statistics.

**Response (200):**

```json
{
  "total_jobs": 42,
  "by_status": {
    "completed": 38,
    "failed": 2,
    "running": 1,
    "queued": 1
  },
  "queue_depth": 1,
  "avg_duration": 12.34
}
```

---

## SSE Event Types

Events sent on the `GET /api/v1/jobs/{id}/stream` endpoint:

| Event | Description | Data fields |
|-------|-------------|-------------|
| `status` | Job status changed | `job_id`, `status` |
| `progress` | Generation progress update | `current_image`, `total_images`, `percent`, `stage` |
| `log` | Captured console output line | `timestamp`, `level`, `message` |
| `result` | Job completed with images | `images` (base64 array), `info`, `prompt` |
| `error` | Job failed or cancelled | `code`, `message`, `details` (optional), `stage` (optional) |

**Progress stages:** `loading_model`, `generating`, `saving`, `upscaling`, `completed`

**SSE format example:**

```
event: progress
data: {"current_image":1,"total_images":2,"percent":50.0,"stage":"generating"}

event: result
data: {"images":["<base64>"],"info":"...","prompt":"..."}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_JSON` | 400 | Request body is not valid JSON |
| `MISSING_PARAM` | 400 | A required parameter is missing |
| `INVALID_PARAM` | 400 | A parameter has an invalid value |
| `GENERATION_FAILED` | 500 | Image generation failed |
| `MODEL_LOAD_FAILED` | 500 | Model could not be loaded |
| `OUT_OF_MEMORY` | 500 | GPU/Metal memory exhausted |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `JOB_ALREADY_COMPLETED` | — | Job has already finished (cannot cancel) |
| `CANCELLED` | — | Job was cancelled by the user |

**Error response format:**

```json
{
  "error": {
    "code": "MISSING_PARAM",
    "message": "prompt is required",
    "details": "optional extra info",
    "stage": "optional stage where error occurred"
  }
}
```

For sync endpoints, errors use the simpler format:

```json
{
  "error": "error message string"
}
```

---

## Examples

### curl — Text to Image (sync)

```bash
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat sitting on a windowsill, golden hour",
    "model": "schnell-4-bit",
    "width": 576,
    "height": 1024,
    "steps": 4,
    "guidance": 3.5
  }'
```

### curl — Async generation with SSE streaming

```bash
# Submit job
JOB=$(curl -s -X POST http://localhost:7861/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "txt2img",
    "prompt": "a cat sitting on a windowsill, golden hour",
    "model": "schnell-4-bit",
    "width": 576,
    "height": 1024
  }')

JOB_ID=$(echo $JOB | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Stream events
curl -N http://localhost:7861/api/v1/jobs/$JOB_ID/stream
```

### curl — Image to Image (sync)

```bash
# Encode an image to base64
IMG_B64=$(base64 < input.png)

curl -X POST http://localhost:7861/sdapi/v1/img2img \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"oil painting style\",
    \"init_images\": [\"$IMG_B64\"],
    \"image_strength\": 0.4
  }"
```

### JavaScript — Async generation with EventSource

```javascript
// Submit a job
const response = await fetch("http://localhost:7861/api/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    type: "txt2img",
    prompt: "a cat sitting on a windowsill, golden hour",
    model: "schnell-4-bit",
    width: 576,
    height: 1024,
  }),
});

const { job_id } = await response.json();

// Stream events
const source = new EventSource(
  `http://localhost:7861/api/v1/jobs/${job_id}/stream`
);

source.addEventListener("progress", (e) => {
  const data = JSON.parse(e.data);
  console.log(`Progress: ${data.percent}% — ${data.stage}`);
});

source.addEventListener("result", (e) => {
  const data = JSON.parse(e.data);
  const img = document.createElement("img");
  img.src = `data:image/png;base64,${data.images[0]}`;
  document.body.appendChild(img);
  source.close();
});

source.addEventListener("error", (e) => {
  if (e.data) {
    const data = JSON.parse(e.data);
    console.error(`Job failed: ${data.code} — ${data.message}`);
  }
  source.close();
});
```

### JavaScript — Poll job status

```javascript
async function pollJob(jobId) {
  while (true) {
    const res = await fetch(
      `http://localhost:7861/api/v1/jobs/${jobId}`
    );
    const job = await res.json();

    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error.message);
    if (job.status === "cancelled") throw new Error("Job cancelled");

    await new Promise((r) => setTimeout(r, 1000));
  }
}
```
