export type VideoAvailability = "ready" | "setup-required";
export type VideoRuntimeState = "ready" | "setup-required" | "busy";
export type VideoJobState = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface VideoCapability {
  id: string;
  label: string;
  type: "video";
  operations: string[];
  availability: VideoAvailability;
  availability_reason: string;
  engine: {
    source: string;
    revision: string;
    license: string;
    tested: boolean;
  };
  model: {
    source: string;
    revision: string;
    license: string;
    converted_local: boolean;
    configured: boolean;
    cached: boolean;
    smoke_tested: boolean;
  };
  parameters: {
    prompt: { min_length: number; max_length: number };
    width: { fixed: number };
    height: { fixed: number };
    num_frames: { minimum: number; maximum: number; rule: string };
    fps: { fixed: number };
    steps: { minimum: number; maximum: number; default: number };
    scheduler: { fixed: string };
    tiling: { allowed: string[]; default: string };
    seed: { minimum: number; maximum: number; optional: boolean };
  };
  output: { container: "mp4"; audio: "none" | string };
  isolation: string;
  concurrency: string;
  cancel_mode: string;
}

export interface VideoCapabilityRegistry {
  schema_version: number;
  default_capability_id: string;
  capabilities: VideoCapability[];
}

export interface VideoRuntimeReason {
  code: string;
  message: string;
}

export interface VideoRuntimeStatus {
  schema_version: number;
  state: VideoRuntimeState;
  ready: boolean;
  apple_silicon: boolean;
  isolation: string;
  concurrency: string;
  cancel_mode: string;
  network_mode: string;
  engine: {
    name: string;
    source: string;
    revision: string;
    license: string;
    configured: boolean;
    available: boolean;
    tested: boolean;
  };
  model: {
    source: string;
    revision: string;
    license: string;
    configured: boolean;
    cached: boolean;
    converted: boolean;
    smoke_tested: boolean;
  };
  reasons: VideoRuntimeReason[];
  active_media_job?: { id: string; type: "video" | "photo_batch" };
}

export interface VideoGenerationRequest {
  schema_version: 1;
  type: "video";
  operation: string;
  capability_id: string;
  prompt: string;
  output: {
    width: number;
    height: number;
    num_frames: number;
    fps: number;
    container: "mp4";
  };
  sampling: {
    steps: number;
    tiling: string;
    seed?: number;
  };
}

export interface VideoSubmitResponse {
  job_id: string;
  status: VideoJobState;
  type: "video";
}

export interface VideoArtifactUrls {
  video?: string;
  provenance?: string;
  request?: string;
}

export interface VideoJobResult {
  schema_version: number;
  status: "completed" | "cancelled";
  job_id: string;
  capability_id: string;
  request_hash?: string;
  artifact_urls?: VideoArtifactUrls;
  output?: {
    container: "mp4";
    sha256: string;
    size_bytes: number;
    verification?: {
      width?: number;
      height?: number;
      fps?: number;
      num_frames?: number;
    };
  };
}

export interface VideoJobResponse {
  job_id: string;
  type: "video";
  status: VideoJobState;
  progress: {
    current_image: number;
    total_images: number;
    percent: number;
    stage: string;
    current_step?: number | null;
    total_steps?: number | null;
  };
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  result?: VideoJobResult;
  error?: { code: string; message: string; details?: string; stage?: string };
}

export interface VideoWorkspaceState {
  registry: VideoCapabilityRegistry | null;
  runtime: VideoRuntimeStatus | null;
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
}

const routes = {
  capabilities: "/api/v1/video/capabilities",
  status: "/api/v1/video/status",
  submit: "/api/v1/generate",
  job: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}`,
} as const;

export class VideoApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "VideoApiError";
    this.status = status;
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    cache: "no-store",
    credentials: "same-origin",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : { error: `The local server returned ${response.status}.` };

  if (!response.ok) {
    const nestedError = body?.error;
    const message = typeof nestedError === "string"
      ? nestedError
      : typeof nestedError?.message === "string"
        ? nestedError.message
        : `The local video service returned ${response.status}.`;
    throw new VideoApiError(message, response.status);
  }

  return body as T;
}

export async function fetchVideoCapabilities(): Promise<VideoCapabilityRegistry> {
  return requestJson<VideoCapabilityRegistry>(routes.capabilities);
}

export async function fetchVideoStatus(): Promise<VideoRuntimeStatus> {
  return requestJson<VideoRuntimeStatus>(routes.status);
}

export async function submitVideoJob(request: VideoGenerationRequest): Promise<VideoSubmitResponse> {
  return requestJson<VideoSubmitResponse>(routes.submit, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function fetchVideoJob(id: string): Promise<VideoJobResponse> {
  return requestJson<VideoJobResponse>(routes.job(id));
}

export async function cancelVideoJob(id: string): Promise<{ job_id: string; status: VideoJobState }> {
  return requestJson(routes.job(id), { method: "DELETE" });
}

export function isVideoJobTerminal(status: VideoJobState): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

export function isFrameCountValid(capability: VideoCapability, frames: number): boolean {
  const limits = capability.parameters.num_frames;
  if (!Number.isInteger(frames) || frames < limits.minimum || frames > limits.maximum) return false;
  const rule = limits.rule.replaceAll(" ", "").toLowerCase();
  if (rule === "4n+1") return (frames - 1) % 4 === 0;
  return false;
}

export function localArtifactUrl(path?: string): string {
  if (!path || typeof window === "undefined") return "";
  try {
    const url = new URL(path, window.location.href);
    return url.origin === window.location.origin ? url.toString() : "";
  } catch {
    return "";
  }
}
