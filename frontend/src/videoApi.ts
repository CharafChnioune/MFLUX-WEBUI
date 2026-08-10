export type VideoModelFamilyId = "ltx-2" | "wan-2.1" | "wan-2.2";

export type VideoTask =
  | "text-to-video"
  | "image-to-video"
  | "audio-to-video";

export type LtxPipeline =
  | "distilled"
  | "dev"
  | "dev-two-stage"
  | "dev-two-stage-hq";

export type WanScheduler = "euler" | "dpm++" | "unipc";

export type VideoRuntimeState =
  | "not-installed"
  | "checking"
  | "ready"
  | "busy"
  | "degraded";

export type VideoJobPhase =
  | "queued"
  | "validating"
  | "downloading"
  | "loading"
  | "conditioning"
  | "denoising"
  | "decoding"
  | "muxing"
  | "completed"
  | "failed"
  | "cancelled";

export type VideoProgress =
  | { kind: "indeterminate"; message: string }
  | { kind: "steps"; current: number; total: number; message?: string }
  | { kind: "frames"; current: number; total: number; message?: string };

export type VideoModelSource =
  | { kind: "hugging-face"; repo_id: string; revision?: string }
  | { kind: "local-converted"; model_dir: string };

export interface VideoGenerationRequest {
  schema_version: "1";
  type: "video";
  operation: VideoTask;
  capability_id: VideoModelFamilyId;
  model_source: VideoModelSource;
  prompt: string;
  negative_prompt?: string;
  conditioning?: {
    first_image_path?: string;
    last_image_path?: string;
    audio_path?: string;
    image_strength?: number;
  };
  output: {
    width: number;
    height: number;
    num_frames: number;
    fps: number;
    container: "mp4";
  };
  sampling: {
    seed: number;
    steps?: number;
    guidance?: number | [number, number];
    ltx_pipeline?: LtxPipeline;
    wan_scheduler?: WanScheduler;
    tiling: "auto" | "none" | "default" | "aggressive" | "conservative" | "spatial" | "temporal";
  };
  generate_audio?: boolean;
  license_acknowledgements: string[];
}

export interface VideoRuntimeStatus {
  schema_version: "1";
  state: VideoRuntimeState;
  engine: {
    name: "mlx-video";
    source: "Blaizzy/mlx-video";
    version?: string;
    pinned_revision?: string;
    tested: boolean;
  };
  apple_silicon: boolean;
  isolation: "separate-subprocess";
  concurrency: "serialized";
  cancel_mode: "queued-only" | "cooperative";
  active_media_job?: { id: string; type: "photo_batch" | "video" };
  warnings: string[];
}

export interface VideoJobStatus {
  id: string;
  phase: VideoJobPhase;
  progress: VideoProgress;
  can_cancel: boolean;
  created_at: string;
  output_path?: string;
  error?: { code: string; message: string; retryable: boolean };
}

export interface VideoCapability {
  id: VideoModelFamilyId;
  label: string;
  tasks: VideoTask[];
  source_kind: VideoModelSource["kind"];
  frame_rule: string;
  dimension_rule: string;
  pipelines: string[];
  license: string;
  availability: "preview" | "unavailable" | "ready";
  availability_reason: string;
  engine: {
    source: "Blaizzy/mlx-video";
    pinned_revision: string | null;
    tested: boolean;
  };
  model_state: {
    configured: boolean;
    cached: boolean;
    ready: boolean;
  };
  output: {
    container: "mp4";
    audio: "optional" | "none";
  };
  integration: "contract-preview";
  caution: string;
}

export const VIDEO_CAPABILITIES: VideoCapability[] = [
  {
    id: "ltx-2",
    label: "LTX-2 / 2.3",
    tasks: ["text-to-video", "image-to-video", "audio-to-video"],
    source_kind: "hugging-face",
    frame_rule: "1 + 8k frames",
    dimension_rule: "Multiples of 32, or 64 for two-stage pipelines",
    pipelines: ["distilled", "dev", "dev-two-stage", "dev-two-stage-hq"],
    license: "LTX-2 Community License — review required",
    availability: "preview",
    availability_reason: "Isolated runner and exact checkpoint smoke test are not complete.",
    engine: { source: "Blaizzy/mlx-video", pinned_revision: null, tested: false },
    model_state: { configured: false, cached: false, ready: false },
    output: { container: "mp4", audio: "optional" },
    integration: "contract-preview",
    caution: "Upstream has open LTX-2.3 I2V and audio regressions; validate the exact checkpoint before enabling jobs.",
  },
  {
    id: "wan-2.1",
    label: "Wan 2.1",
    tasks: ["text-to-video", "image-to-video"],
    source_kind: "local-converted",
    frame_rule: "4n + 1 frames",
    dimension_rule: "Checkpoint-specific; validate before submission",
    pipelines: ["single-model", "Euler", "DPM++", "UniPC"],
    license: "Apache-2.0 model weights",
    availability: "preview",
    availability_reason: "A compatible converted model directory has not been validated.",
    engine: { source: "Blaizzy/mlx-video", pinned_revision: null, tested: false },
    model_state: { configured: false, cached: false, ready: false },
    output: { container: "mp4", audio: "none" },
    integration: "contract-preview",
    caution: "The selected converted directory determines whether text-to-video or image-to-video is available.",
  },
  {
    id: "wan-2.2",
    label: "Wan 2.2",
    tasks: ["text-to-video", "image-to-video"],
    source_kind: "local-converted",
    frame_rule: "4n + 1 frames",
    dimension_rule: "Checkpoint-specific; validate before submission",
    pipelines: ["single-model", "dual-model", "Euler", "DPM++", "UniPC", "LoRA"],
    license: "Apache-2.0 model weights",
    availability: "preview",
    availability_reason: "A compatible converted model directory has not been validated.",
    engine: { source: "Blaizzy/mlx-video", pinned_revision: null, tested: false },
    model_state: { configured: false, cached: false, ready: false },
    output: { container: "mp4", audio: "none" },
    integration: "contract-preview",
    caution: "Quantized 14B paths have open artifact reports; do not mark them ready without a local smoke test.",
  },
];

export const VIDEO_API_ROUTES = {
  capabilities: "/api/v1/video/capabilities",
  status: "/api/v1/video/status",
  submit: "/api/v1/generate",
  job: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}`,
  events: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/stream`,
} as const;

export function tasksForModel(model: VideoModelFamilyId): VideoTask[] {
  return VIDEO_CAPABILITIES.find((capability) => capability.id === model)?.tasks ?? [];
}

export function isFrameCountValid(model: VideoModelFamilyId, frames: number): boolean {
  if (!Number.isInteger(frames) || frames < 1) return false;
  return model === "ltx-2" ? (frames - 1) % 8 === 0 : (frames - 1) % 4 === 0;
}
