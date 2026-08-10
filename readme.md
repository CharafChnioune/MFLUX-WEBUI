# MLX Media

> **We are back — now as a local media studio.**

MLX Media is a private, Apple-silicon workspace for creating, editing, restoring and
organizing visual media. It evolves MFLUX-WebUI into one coherent photo-and-video
product while keeping model limits, licenses and runtime state visible instead of
pretending every backend works the same way.

<img src="frontend/public/assets/mlx-media-emblem.png" alt="MLX Media emblem" width="180" />

## What works today

| Area | Status | Notes |
| --- | --- | --- |
| Local photo API | Available | Async image jobs, job history and SSE progress through the existing API |
| Photo import | Available | Local inventory with EXIF/GPS discovery, explicit privacy modes and manual location overrides |
| SeedVR2 restoration batches | Available | Folder planning, safe outputs, progress and original-file preservation |
| React studio | Buildable preview | Premium dark interface for Home, Images, Video, Restore, Time Lens, Library, Models, Activity and Settings |
| MLX video | Integration preview | Audited capability shell only; no video runner, model download or generation is enabled yet |

The source checkout and an installed Pinokio application are separate. Updating this
repository does not restart or mutate an already-running restoration job.

## Product principles

- **Local by default.** Source media stays on the Mac unless the user explicitly chooses
  another path.
- **Originals are protected.** Restoration and edits produce new artifacts instead of
  overwriting source files.
- **Location is opt-in.** EXIF/GPS may suggest a location, can be disabled per import,
  and can be replaced with a manual label. MLX Media never presents inferred history as
  fact.
- **One honest queue.** Large photo and future video jobs must be serialized to protect
  unified memory and predictable performance.
- **Capability-driven UI.** Models expose their own supported operations, safe defaults,
  constraints, cache state and license requirements.

## Photo capabilities

The backend is built around MFLUX on Apple silicon and includes the established async
generation API plus local photo inventory and SeedVR2 batch restoration. Exact request
and response formats live in [API.md](API.md).

Highlights:

- text-to-image and image editing through the configured MFLUX model paths;
- asynchronous jobs with status, history and Server-Sent Events;
- SeedVR2 3B/7B restoration and upscale planning;
- non-destructive batch outputs with metadata/orientation handling;
- local photo inventory with HEIF support and privacy-aware GPS suggestions;
- model-specific settings rather than a single unsafe universal form.

## Video: audited, not overclaimed

The interface now contains an MLX video workflow preview based on a concrete audit of
[`Blaizzy/mlx-video`](https://github.com/Blaizzy/mlx-video). The upstream project
contains LTX-2/2.3 and Wan 2.1/2.2 generation code, but it does not provide the web/job
API required by this product and its dependency graph must not be mixed into the photo
environment.

Therefore:

- the video submit action is disabled;
- no video model is downloaded automatically;
- the model cards show preview/setup state and license cautions;
- the proposed backend reuses the existing serialized job queue through a separate,
  pinned subprocess environment;
- a capability will only become available after an exact checkpoint passes a local
  Apple-silicon smoke test.

Read the full [MLX video integration audit](docs/MLX_VIDEO_INTEGRATION.md).

## AI assistance: capability-first

The repository contains several separate local-assistance building blocks, but they are
not interchangeable:

- the current Nativ integration only checks a configurable local server's health and
  detected model IDs through `GET /api/v1/providers/nativ`;
- prompt refinement exists in legacy backend code through Ollama or local MLX paths,
  but is not wired into the React studio yet;
- local MLX-VLM captioning code exists, but is not exposed as a tested automatic-tagging
  product flow.

For that reason the interface presents Nativ as **status integration only** and keeps
prompt refinement labelled as unconnected. It does not claim a Nativ assistant, image
understanding, reliable automatic tags or structured creative assistance.

## Run from source

### Backend API

Requirements: macOS on Apple silicon, Python 3.11+ and enough local memory/disk for the
chosen model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python api_main.py
```

The local API binds to `127.0.0.1:7861` by default. Use `webui.py` as the Pinokio-
compatible entry point; it currently starts the same API server.

To allow local photo inventory and restoration planning, configure a dedicated import
root before starting the API:

```bash
export MFLUX_PHOTO_IMPORT_ROOT="/absolute/path/to/photos"
python api_main.py
```

The photo endpoints reject non-loopback access and paths outside that root.

### React studio

```bash
cd frontend
npm install
npm run dev
```

The React app is currently a product shell. It demonstrates validated forms, privacy
controls and readiness states; not every visible photo action is wired to the API yet.

Production build:

```bash
cd frontend
npm run build
```

## Main API surfaces

- `GET /api/v1/health` — server and capability health
- `POST /api/v1/generate` — asynchronous generation and photo-batch submission
- `GET /api/v1/jobs` — job inventory
- `GET /api/v1/jobs/{id}` — job state
- `GET /api/v1/jobs/{id}/stream` — Server-Sent Event progress
- `DELETE /api/v1/jobs/{id}` — cancellation request
- `GET /api/v1/photo-imports/config` — local import policy
- `POST /api/v1/photo-imports/inventory` — metadata/privacy-aware inventory
- `POST /api/v1/photo-batches/plan` — safe SeedVR2 batch plan

Video capability/status routes documented in the integration audit are proposed and are
not served yet.

## Development checks

```bash
python -m pytest -q tests/test_api_health.py tests/test_photo_imports.py tests/test_photo_batch.py
cd frontend && npm run build
```

## Relaunch notes

- New product name and navigation: **MLX Media**.
- A single premium dark visual system across desktop and mobile.
- General-purpose examples with no destination-specific hardcoding.
- Privacy-first photo import and Time Lens language.
- First-class Restore workflow for SeedVR2.
- New Video and cross-media Models surfaces with honest readiness gates.
- Shared Activity and Settings concepts for future photo/video resource scheduling.

## Licensing and attribution

This repository's code license does not replace the terms attached to model weights.
Review the repository license plus every selected model's license before downloading or
using it. In particular, LTX-2 uses its own community license, while official Wan 2.1
and Wan 2.2 weights advertise Apache-2.0 terms.

Core upstream projects include:

- [MFLUX](https://github.com/mflux-community/mflux)
- [MLX](https://github.com/ml-explore/mlx)
- [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) — audited future video engine; not yet installed by MLX Media
- [Pinokio](https://pinokio.computer)

## Safety note

Large local models can consume substantial unified memory and disk. Start with
conservative presets, keep the application local-only, preserve originals, and verify
the output of any generative or restoration model before relying on it.
