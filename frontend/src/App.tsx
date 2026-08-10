import {
  Activity,
  Aperture,
  ArrowRight,
  Check,
  ChevronDown,
  CircleGauge,
  Clock3,
  CloudOff,
  Command,
  Cpu,
  Download,
  Eye,
  ExternalLink,
  Film,
  FolderOpen,
  GalleryVerticalEnd,
  Gauge,
  HardDrive,
  History,
  Image as ImageIcon,
  Info,
  Layers3,
  Library,
  LoaderCircle,
  LockKeyhole,
  MapPin,
  Menu,
  MoreHorizontal,
  MonitorPlay,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  Tag,
  TriangleAlert,
  Upload,
  WandSparkles,
  Zap,
  type LucideIcon,
} from "lucide-react";
import {
  type ChangeEvent,
  type CSSProperties,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  cancelVideoJob,
  fetchVideoCapabilities,
  fetchVideoJob,
  fetchVideoStatus,
  isFrameCountValid,
  isVideoJobTerminal,
  localArtifactUrl,
  submitVideoJob,
  type VideoCapabilityRegistry,
  type VideoGenerationRequest,
  type VideoJobResponse,
  type VideoRuntimeStatus,
  type VideoWorkspaceState,
} from "./videoApi";

type PageId =
  | "home"
  | "create"
  | "video"
  | "restore"
  | "time-lens"
  | "library"
  | "models"
  | "activity"
  | "settings";

type NavItem = {
  id: PageId;
  label: string;
  description: string;
  icon: LucideIcon;
};

const navigation: NavItem[] = [
  { id: "home", label: "Home", description: "Your local media launchpad", icon: Aperture },
  { id: "create", label: "Images", description: "Generate and edit stills", icon: Sparkles },
  { id: "video", label: "Video", description: "Generate local motion", icon: Film },
  { id: "restore", label: "Restore", description: "Faithful photo enhancement", icon: WandSparkles },
  { id: "time-lens", label: "Time Lens", description: "Concept preview · unavailable", icon: History },
  { id: "library", label: "Library", description: "Projects and outputs", icon: Library },
  { id: "models", label: "Models", description: "Your local model catalog", icon: Layers3 },
  { id: "activity", label: "Activity", description: "Queue and history", icon: Activity },
  { id: "settings", label: "Settings", description: "Performance and privacy", icon: Settings },
];

const quickActions = [
  {
    id: "restore" as PageId,
    title: "Restore a memory",
    copy: "Upscale a soft or compressed photo while protecting the original.",
    icon: WandSparkles,
    tone: "cyan",
  },
  {
    id: "create" as PageId,
    title: "Create an image",
    copy: "Start with an idea and let Auto choose the right local model.",
    icon: Sparkles,
    tone: "violet",
  },
  {
    id: "video" as PageId,
    title: "Generate a video",
    copy: "Turn a motion brief into a local MP4 with server-validated settings.",
    icon: Film,
    tone: "blue",
  },
  {
    id: "time-lens" as PageId,
    title: "Preview Time Lens",
    copy: "Explore the future guided-restoration concept. Processing is not connected yet.",
    icon: History,
    tone: "magenta",
  },
];

const modelCards = [
  {
    name: "SeedVR2 3B",
    role: "Faithful restore",
    copy: "The balanced choice for photographs, batch restoration and print-ready upscales.",
    tags: ["Restore", "3B", "Promptless"],
    status: "Recommended",
    tone: "cyan",
    kind: "Photo",
  },
  {
    name: "SeedVR2 7B",
    role: "Maximum detail",
    copy: "A larger restoration model for difficult sources and final-quality exports.",
    tags: ["Restore", "7B", "High memory"],
    status: "Available",
    tone: "violet",
    kind: "Photo",
  },
  {
    name: "FLUX.2 Klein",
    role: "Fast create & edit",
    copy: "Responsive generation with single- and multi-reference editing workflows.",
    tags: ["Generate", "Edit", "4B / 9B"],
    status: "Catalog",
    tone: "blue",
    kind: "Photo",
  },
  {
    name: "Krea 2 Turbo",
    role: "Creative exploration",
    copy: "Style-rich visual generation with a fast, curated creative workflow.",
    tags: ["Generate", "Turbo", "Gated"],
    status: "Catalog",
    tone: "orange",
    kind: "Photo",
  },
  {
    name: "Ideogram 4",
    role: "Text & layout",
    copy: "Structured composition for typography, posters and designed images.",
    tags: ["Text", "Layout", "Structured"],
    status: "Catalog",
    tone: "magenta",
    kind: "Photo",
  },
  {
    name: "Qwen Image Edit",
    role: "Natural-language edit",
    copy: "Multi-image edits with strong visual understanding and local execution.",
    tags: ["Edit", "Multi-image", "20B"],
    status: "Catalog",
    tone: "green",
    kind: "Photo",
  },
];

const libraryItems = [
  { title: "Golden-hour walk", meta: "18 photographs", tag: "Travel", art: "sunset" },
  { title: "Family archive", meta: "42 restored", tag: "Archive", art: "archive" },
  { title: "Coastal light", meta: "12 variations", tag: "Create", art: "coast" },
  { title: "Overnight journey", meta: "7 photographs", tag: "Travel", art: "night" },
  { title: "Poster studies", meta: "24 outputs", tag: "Typography", art: "poster" },
  { title: "Print collection", meta: "9 print-ready", tag: "Restore", art: "print" },
];

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <img src="/assets/mlx-media-emblem.png" alt="" />
    </span>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  disabled = false,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      className={`toggle ${checked ? "is-on" : ""}`}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span />
    </button>
  );
}

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" aria-label={label}>
      {options.map((option) => (
        <button
          type="button"
          key={option}
          className={value === option ? "is-active" : ""}
          aria-pressed={value === option}
          onClick={() => onChange(option)}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function SectionHeading({
  eyebrow,
  title,
  copy,
  action,
}: {
  eyebrow?: string;
  title: string;
  copy?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
        {copy && <p>{copy}</p>}
      </div>
      {action}
    </div>
  );
}

function MockPhoto({ variant = "sunset" }: { variant?: string }) {
  return (
    <div className={`mock-photo mock-photo--${variant}`} aria-hidden="true">
      <span className="mock-sun" />
      <span className="mock-ridge mock-ridge--back" />
      <span className="mock-ridge mock-ridge--front" />
      <span className="mock-structure" />
      <span className="mock-grain" />
    </div>
  );
}

function PhotoLayer({
  src,
  variant,
  alt,
}: {
  src: string | null;
  variant: string;
  alt: string;
}) {
  return src ? <img src={src} alt={alt} /> : <MockPhoto variant={variant} />;
}

function ComparisonStage({
  src,
  value,
  onChange,
  variant = "sunset",
  compact = false,
  disabled = false,
}: {
  src: string | null;
  value: number;
  onChange: (value: number) => void;
  variant?: string;
  compact?: boolean;
  disabled?: boolean;
}) {
  const clipStyle = { "--compare": `${value}%` } as CSSProperties;

  return (
    <div className={`comparison ${compact ? "comparison--compact" : ""}`}>
      <div className="comparison-stage" style={clipStyle}>
        <div className="comparison-layer comparison-layer--before">
          <PhotoLayer src={src} variant={variant} alt="Original preview" />
        </div>
        <div className="comparison-layer comparison-layer--after">
          <PhotoLayer src={src} variant={variant} alt="Restored preview" />
        </div>
        <span className="comparison-label comparison-label--left">Original</span>
        <span className="comparison-label comparison-label--right">Restored</span>
        <span className="comparison-line" aria-hidden="true">
          <span><ChevronDown size={14} /></span>
        </span>
      </div>
      <label className="comparison-control">
        <span className="sr-only">Move before and after comparison</span>
        <input
          type="range"
          min="0"
          max="100"
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          disabled={disabled}
        />
      </label>
    </div>
  );
}

function HomePage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  return (
    <div className="page-stack home-page">
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="release-pill"><span /> MLX MEDIA · RELAUNCH PREVIEW</span>
          <h1>Photo and video.<br />One local studio.</h1>
          <p>
            Create, edit and restore visual media on Apple silicon—with clear model boundaries,
            private source files and one calm workspace for stills and motion.
          </p>
          <div className="button-row">
            <button className="button button--primary" type="button" onClick={() => onNavigate("restore")}>
              Restore a photo <ArrowRight size={16} />
            </button>
            <button className="button button--ghost" type="button" onClick={() => onNavigate("create")}>
              Create an image
            </button>
            <button className="button button--ghost" type="button" onClick={() => onNavigate("video")}>
              Explore video
            </button>
          </div>
        </div>
        <div className="hero-visual" aria-label="Abstract preview of recent MLX Media work">
          <img className="hero-emblem" src="/assets/mlx-media-emblem.png" alt="" />
          <div className="hero-orbit hero-orbit--one" />
          <div className="hero-orbit hero-orbit--two" />
          <div className="floating-output floating-output--back"><MockPhoto variant="coast" /></div>
          <div className="floating-output floating-output--front"><MockPhoto variant="sunset" /></div>
          <div className="hero-status"><ShieldCheck size={15} /> Local only · Capability aware</div>
        </div>
      </section>

      <section>
        <SectionHeading
          eyebrow="Start here"
          title="What do you want to make?"
          copy="Choose an intent. MLX Media reveals model constraints only when they matter."
        />
        <div className="quick-grid">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <button
                type="button"
                className={`quick-card quick-card--${action.tone}`}
                key={action.id}
                onClick={() => onNavigate(action.id)}
              >
                <span className="quick-icon"><Icon size={21} /></span>
                <span>
                  <strong>{action.title}</strong>
                  <small>{action.copy}</small>
                </span>
                <ArrowRight size={18} className="quick-arrow" />
              </button>
            );
          })}
        </div>
      </section>

      <section className="home-bottom-grid">
        <div className="panel recent-panel">
          <SectionHeading
            eyebrow="Continue"
            title="Recent work"
            action={<button className="text-button" type="button" onClick={() => onNavigate("library")}>Open library</button>}
          />
          <div className="recent-list">
            {libraryItems.slice(0, 3).map((item) => (
              <button type="button" className="recent-row" key={item.title} onClick={() => onNavigate("library")}>
                <span className={`recent-thumb recent-thumb--${item.art}`}><MockPhoto variant={item.art} /></span>
                <span className="recent-copy"><strong>{item.title}</strong><small>{item.meta}</small></span>
                <span className="tag-chip">{item.tag}</span>
                <MoreHorizontal size={17} />
              </button>
            ))}
          </div>
        </div>
        <div className="panel system-card">
          <div className="system-card-head">
            <span className="system-icon"><Zap size={19} /></span>
            <div><span className="eyebrow">Apple silicon</span><h3>Private workspace</h3></div>
            <span className="status-dot" aria-label="Local workspace" />
          </div>
          <div className="system-meter"><span style={{ width: "32%" }} /></div>
          <div className="system-stats">
            <span><small>Memory profile</small><strong>Balanced</strong></span>
            <span><small>Queue</small><strong>Server managed</strong></span>
          </div>
          <p><CloudOff size={14} /> Your media stays on this Mac by default.</p>
        </div>
      </section>
    </div>
  );
}

function CreatePage({ notify }: { notify: (message: string) => void }) {
  const [quality, setQuality] = useState("Balanced");
  const [prompt, setPrompt] = useState("A quiet travel courtyard at blue hour, warm windows, cinematic photography");

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow="Create"
        title="Turn an idea into an image"
        copy="Describe the result you want. Auto keeps model-specific settings sane."
        action={<span className="context-pill"><Sparkles size={14} /> Auto model</span>}
      />
      <div className="studio-grid">
        <section className="panel prompt-panel" aria-label="Creation brief">
          <label className="field-label" htmlFor="create-prompt">Creative brief</label>
          <textarea id="create-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={6} />
          <div className="prompt-tools">
            <button type="button" className="tool-button"><Plus size={15} /> Reference</button>
            <button type="button" className="tool-button"><Tag size={15} /> Style</button>
            <button type="button" className="tool-button" onClick={() => notify("Local prompt refinement is available in the backend but is not wired to this studio preview yet.")}><RefreshCw size={15} /> Local refine</button>
          </div>
          <div className="field-stack">
            <span className="field-label">Output intent</span>
            <div className="intent-list">
              <button type="button" className="intent-chip is-active">Photographic</button>
              <button type="button" className="intent-chip">Illustrative</button>
              <button type="button" className="intent-chip">Text & layout</button>
            </div>
          </div>
          <div className="panel-divider" />
          <div className="field-stack">
            <span className="field-label">Performance</span>
            <Segmented label="Generation quality" options={["Speed", "Balanced", "Quality"]} value={quality} onChange={setQuality} />
          </div>
          <button className="button button--primary button--wide" type="button" onClick={() => notify("Creative brief prepared for the local queue.")}>
            <Sparkles size={17} /> Prepare generation
          </button>
        </section>

        <section className="panel canvas-panel" aria-label="Image preview">
          <div className="canvas-toolbar">
            <span><span className="status-dot" /> Preview canvas</span>
            <div><button type="button" aria-label="Fit preview"><Eye size={16} /></button><button type="button" aria-label="More preview options"><MoreHorizontal size={17} /></button></div>
          </div>
          <div className="create-canvas"><MockPhoto variant="night" /></div>
          <div className="canvas-caption"><span>Concept preview</span><span>1024 × 1024 · Auto</span></div>
        </section>

        <aside className="panel inspector-panel" aria-label="Generation inspector">
          <div className="inspector-title"><SlidersHorizontal size={17} /><strong>Inspector</strong><span>Simple</span></div>
          <div className="inspector-group"><small>Model route</small><button type="button" className="select-button"><span><strong>Auto</strong><small>Best fit for this brief</small></span><ChevronDown size={16} /></button></div>
          <div className="inspector-group"><small>Canvas</small><div className="ratio-grid"><button className="is-active" type="button">1:1</button><button type="button">4:5</button><button type="button">3:2</button></div></div>
          <div className="inspector-group"><small>Variations</small><div className="stepper"><button type="button">−</button><strong>4</strong><button type="button">+</button></div></div>
          <div className="info-callout"><CircleGauge size={17} /><span><strong>Balanced route</strong><small>Good detail without maxing out memory.</small></span></div>
          <button type="button" className="text-button expert-link"><Settings size={14} /> Show expert controls</button>
        </aside>
      </div>
    </div>
  );
}

function useVideoWorkspace(): VideoWorkspaceState {
  const [registry, setRegistry] = useState<VideoCapabilityRegistry | null>(null);
  const [runtime, setRuntime] = useState<VideoRuntimeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [nextRegistry, nextRuntime] = await Promise.all([
        fetchVideoCapabilities(),
        fetchVideoStatus(),
      ]);
      setRegistry(nextRegistry);
      setRuntime(nextRuntime);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "The local video service is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void fetchVideoStatus()
        .then((nextRuntime) => {
          setRuntime(nextRuntime);
          setError("");
        })
        .catch((nextError) => {
          setError(nextError instanceof Error ? nextError.message : "The local video service is unavailable.");
        });
    }, 8000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  return { registry, runtime, loading, error, refresh };
}

function humanizeVideoStage(value: string): string {
  if (!value) return "Waiting for the local runner";
  return value
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}

function videoRuntimeLabel(video: VideoWorkspaceState): string {
  if (video.loading && !video.runtime) return "Checking";
  if (video.error && !video.runtime) return "Unavailable";
  if (video.runtime?.state === "ready") return "Ready";
  if (video.runtime?.state === "busy") return "Busy";
  return "Setup required";
}

function VideoPage({ video, notify }: { video: VideoWorkspaceState; notify: (message: string) => void }) {
  const capabilities = video.registry?.capabilities ?? [];
  const [capabilityId, setCapabilityId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [frames, setFrames] = useState(5);
  const [steps, setSteps] = useState(10);
  const [tiling, setTiling] = useState("auto");
  const [seed, setSeed] = useState("");
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<VideoJobResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [jobError, setJobError] = useState("");
  const capability = capabilities.find((item) => item.id === capabilityId) ?? null;

  useEffect(() => {
    if (!capabilities.length) {
      setCapabilityId("");
      return;
    }
    if (capabilities.some((item) => item.id === capabilityId)) return;
    const preferred = capabilities.find((item) => item.id === video.registry?.default_capability_id) ?? capabilities[0];
    setCapabilityId(preferred.id);
  }, [capabilities, capabilityId, video.registry?.default_capability_id]);

  useEffect(() => {
    if (!capability) return;
    setFrames(capability.parameters.num_frames.minimum);
    setSteps(capability.parameters.steps.default);
    setTiling(capability.parameters.tiling.default);
    setSeed("");
  }, [capability?.id]);

  useEffect(() => {
    if (!jobId || (job && isVideoJobTerminal(job.status))) return undefined;
    let disposed = false;

    const update = async () => {
      try {
        const next = await fetchVideoJob(jobId);
        if (disposed) return;
        setJob(next);
        setJobError(next.error?.message ?? "");
        if (isVideoJobTerminal(next.status)) void video.refresh();
      } catch (nextError) {
        if (!disposed) setJobError(nextError instanceof Error ? nextError.message : "The video job could not be read.");
      }
    };

    void update();
    const timer = window.setInterval(() => void update(), 1200);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [job?.status, jobId, video.refresh]);

  const frameValid = capability ? isFrameCountValid(capability, frames) : false;
  const promptValid = capability
    ? prompt.trim().length >= capability.parameters.prompt.min_length
      && prompt.trim().length <= capability.parameters.prompt.max_length
    : false;
  const seedNumber = seed === "" ? undefined : Number(seed);
  const seedValid = capability
    ? seedNumber === undefined
      || (Number.isInteger(seedNumber)
        && seedNumber >= capability.parameters.seed.minimum
        && seedNumber <= capability.parameters.seed.maximum)
    : false;
  const stepsValid = capability
    ? Number.isInteger(steps)
      && steps >= capability.parameters.steps.minimum
      && steps <= capability.parameters.steps.maximum
    : false;
  const tilingValid = capability?.parameters.tiling.allowed.includes(tiling) ?? false;
  const jobActive = Boolean(jobId && (!job || !isVideoJobTerminal(job.status)));
  const runtimeReady = video.runtime?.ready === true && video.runtime.state === "ready";
  const capabilityReady = capability?.availability === "ready";
  const operation = capability?.operations.includes("text-to-video") ? "text-to-video" : "";
  const canSubmit = Boolean(
    capability
    && operation
    && runtimeReady
    && capabilityReady
    && frameValid
    && promptValid
    && seedValid
    && stepsValid
    && tilingValid
    && !submitting
    && !jobActive,
  );
  const progress = Math.max(0, Math.min(100, job?.progress.percent ?? 0));
  const artifactUrls = job?.result?.artifact_urls;
  const videoUrl = localArtifactUrl(artifactUrls?.video);
  const duration = capability ? (frames / capability.parameters.fps.fixed).toFixed(2) : "—";
  const runtimeLabel = videoRuntimeLabel(video);

  const submit = async () => {
    if (!capability || !canSubmit) return;
    const request: VideoGenerationRequest = {
      schema_version: 1,
      type: "video",
      operation,
      capability_id: capability.id,
      prompt: prompt.trim(),
      output: {
        width: capability.parameters.width.fixed,
        height: capability.parameters.height.fixed,
        num_frames: frames,
        fps: capability.parameters.fps.fixed,
        container: capability.output.container,
      },
      sampling: {
        steps,
        tiling,
        ...(seedNumber === undefined ? {} : { seed: seedNumber }),
      },
    };

    setSubmitting(true);
    setJob(null);
    setJobId("");
    setJobError("");
    try {
      const submitted = await submitVideoJob(request);
      setJobId(submitted.job_id);
      notify("Video added to the local media queue.");
      void video.refresh();
    } catch (nextError) {
      setJobError(nextError instanceof Error ? nextError.message : "The video job could not be submitted.");
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async () => {
    if (!jobId || !jobActive) return;
    setCancelling(true);
    try {
      await cancelVideoJob(jobId);
      setJob((current) => current ? { ...current, status: "cancelled" } : current);
      notify("Video cancellation requested.");
      void video.refresh();
    } catch (nextError) {
      setJobError(nextError instanceof Error ? nextError.message : "The video job could not be cancelled.");
    } finally {
      setCancelling(false);
    }
  };

  const clearJob = () => {
    setJobId("");
    setJob(null);
    setJobError("");
  };

  return (
    <div className="page-stack video-page">
      <SectionHeading
        eyebrow="Video"
        title="Turn a motion brief into a local MP4"
        copy="Every control comes from the connected server capability. Jobs share the same serialized media queue as photo work."
        action={(
          <span className={`context-pill video-state-pill video-state-pill--${video.runtime?.state ?? "setup-required"}`}>
            {video.loading && !video.runtime ? <LoaderCircle className="is-spinning" size={14} /> : <MonitorPlay size={14} />}
            {runtimeLabel}
          </span>
        )}
      />

      <div className="video-studio-grid">
        <section className="panel video-brief-panel" aria-label="Video generation controls">
          <div className="video-live-header">
            <div><span className="eyebrow">Server capability</span><h3>{capability?.label ?? "No video capability connected"}</h3></div>
            <button type="button" className="button button--ghost button--compact" onClick={() => void video.refresh()} disabled={video.loading}>
              <RefreshCw className={video.loading ? "is-spinning" : ""} size={14} /> Refresh
            </button>
          </div>

          {capabilities.length > 1 && (
            <label className="field-stack" htmlFor="video-capability">
              <span className="field-label">Video capability</span>
              <select id="video-capability" value={capabilityId} onChange={(event) => setCapabilityId(event.target.value)}>
                {capabilities.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
              </select>
            </label>
          )}

          <label className="field-stack" htmlFor="video-prompt">
            <span className="field-label">Motion brief</span>
            <textarea
              id="video-prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={6}
              maxLength={capability?.parameters.prompt.max_length}
              placeholder="Describe the scene, camera movement, light and motion…"
              disabled={!capability || jobActive}
            />
            <small className="field-help">{prompt.length}{capability ? ` / ${capability.parameters.prompt.max_length}` : ""} characters</small>
          </label>

          <div className="video-output-grid video-output-grid--live">
            <label className="field-stack">
              <span className="field-label">Frames</span>
              <input
                className={capability && !frameValid ? "is-invalid" : ""}
                type="number"
                min={capability?.parameters.num_frames.minimum}
                max={capability?.parameters.num_frames.maximum}
                step={4}
                value={frames}
                onChange={(event) => setFrames(Number(event.target.value))}
                disabled={!capability || jobActive}
              />
              <small>{capability ? `${capability.parameters.num_frames.minimum}–${capability.parameters.num_frames.maximum} · ${capability.parameters.num_frames.rule}` : "From server"}</small>
            </label>
            <label className="field-stack">
              <span className="field-label">Steps</span>
              <input
                type="number"
                min={capability?.parameters.steps.minimum}
                max={capability?.parameters.steps.maximum}
                value={steps}
                onChange={(event) => setSteps(Number(event.target.value))}
                disabled={!capability || jobActive}
              />
              <small>{capability ? `${capability.parameters.steps.minimum}–${capability.parameters.steps.maximum}` : "From server"}</small>
            </label>
            <label className="field-stack">
              <span className="field-label">Tiling</span>
              <select value={tiling} onChange={(event) => setTiling(event.target.value)} disabled={!capability || jobActive}>
                {(capability?.parameters.tiling.allowed ?? []).map((item) => <option key={item} value={item}>{humanizeVideoStage(item)}</option>)}
              </select>
              <small>Server-allowed modes</small>
            </label>
            <label className="field-stack">
              <span className="field-label">Seed <small>optional</small></span>
              <input
                className={capability && !seedValid ? "is-invalid" : ""}
                type="number"
                min={capability?.parameters.seed.minimum}
                max={capability?.parameters.seed.maximum}
                value={seed}
                placeholder="Random"
                onChange={(event) => setSeed(event.target.value)}
                disabled={!capability || jobActive}
              />
              <small>Blank chooses a secure random seed</small>
            </label>
          </div>

          {capability && (
            <div className="video-spec-strip">
              <span><small>Output</small><strong>{capability.parameters.width.fixed} × {capability.parameters.height.fixed}</strong></span>
              <span><small>Frame rate</small><strong>{capability.parameters.fps.fixed} fps</strong></span>
              <span><small>Duration</small><strong>{duration}s</strong></span>
              <span><small>Scheduler</small><strong>{capability.parameters.scheduler.fixed}</strong></span>
              <span><small>Container</small><strong>{capability.output.container.toUpperCase()}</strong></span>
            </div>
          )}

          {capability && !frameValid && <p className="validation-note"><TriangleAlert size={14} /> Frames must stay within the server range and follow {capability.parameters.num_frames.rule}.</p>}
          {capability && !seedValid && <p className="validation-note"><TriangleAlert size={14} /> Seed must be a whole number in the server-provided range.</p>}
          {video.error && <p className="video-inline-error" role="alert"><TriangleAlert size={15} /> {video.error}</p>}
          {jobError && <p className="video-inline-error" role="alert"><TriangleAlert size={15} /> {jobError}</p>}

          <div className="button-row video-submit-row">
            <button className="button button--primary button--grow" type="button" disabled={!canSubmit} onClick={() => void submit()}>
              {submitting ? <LoaderCircle className="is-spinning" size={17} /> : <Film size={17} />}
              {submitting ? "Submitting…" : runtimeReady ? "Generate video" : runtimeLabel}
            </button>
            {jobActive && (
              <button className="button button--danger" type="button" onClick={() => void cancel()} disabled={cancelling}>
                {cancelling ? <LoaderCircle className="is-spinning" size={15} /> : <Square size={14} />} Cancel
              </button>
            )}
          </div>
        </section>

        <section className="panel video-preview-panel video-preview-panel--live" aria-label="Video job and output">
          <div className="canvas-toolbar">
            <span><span className={`status-dot status-dot--${job?.status ?? video.runtime?.state ?? "setup-required"}`} /> Local output</span>
            <span className={`preview-badge preview-badge--${job?.status ?? video.runtime?.state ?? "setup-required"}`}>{job?.status ?? runtimeLabel}</span>
          </div>

          {videoUrl ? (
            <div className="video-result">
              <video controls preload="metadata" src={videoUrl} aria-label="Generated local video" />
              <div className="video-result-meta">
                <span><strong>Generation complete</strong><small>{capability?.label ?? job?.result?.capability_id}</small></span>
                <a className="button button--ghost button--compact" href={videoUrl} download><Download size={14} /> Download MP4</a>
              </div>
              <div className="artifact-links" aria-label="Video artifact details">
                {localArtifactUrl(artifactUrls?.provenance) && <a href={localArtifactUrl(artifactUrls?.provenance)} target="_blank" rel="noreferrer">Provenance <ExternalLink size={12} /></a>}
                {localArtifactUrl(artifactUrls?.request) && <a href={localArtifactUrl(artifactUrls?.request)} target="_blank" rel="noreferrer">Request <ExternalLink size={12} /></a>}
              </div>
            </div>
          ) : (
            <div className={`video-job-stage video-job-stage--${job?.status ?? video.runtime?.state ?? "setup-required"}`}>
              <span className="video-job-stage__icon">
                {jobActive || video.loading ? <LoaderCircle className="is-spinning" size={28} /> : video.runtime?.state === "ready" ? <Check size={28} /> : video.runtime?.state === "busy" ? <Clock3 size={28} /> : <TriangleAlert size={28} />}
              </span>
              <span className="eyebrow">{job ? "Media queue" : "Video runtime"}</span>
              <h3>{job ? humanizeVideoStage(job.progress.stage || job.status) : runtimeLabel}</h3>
              <p>{job
                ? job.status === "queued"
                  ? "Waiting behind the current media job. Photo and video work stay serialized."
                  : job.status === "cancelled"
                    ? "The job was cancelled and partial video output is not exposed."
                    : job.status === "failed"
                      ? "The local runner stopped before a verified MP4 was published."
                      : "The isolated runner is working locally. You can leave this page open to follow progress."
                : video.runtime?.state === "ready"
                  ? "The pinned engine, converted model and smoke proof are ready for a real local job."
                  : video.runtime?.state === "busy"
                    ? `A ${video.runtime.active_media_job?.type === "photo_batch" ? "photo restoration" : "video"} job is using the media queue.`
                    : capability?.availability_reason ?? "Connect the local backend to read video capabilities."}</p>

              {job && (
                <div className="video-progress" aria-label={`${Math.round(progress)} percent complete`}>
                  <div><span style={{ width: `${progress}%` }} /></div>
                  <span><strong>{Math.round(progress)}%</strong><small>{job.progress.total_steps ? `Step ${job.progress.current_step ?? 0} of ${job.progress.total_steps}` : humanizeVideoStage(job.progress.stage)}</small></span>
                </div>
              )}

              {job && isVideoJobTerminal(job.status) && (
                <button type="button" className="button button--ghost" onClick={clearJob}>Start a new video</button>
              )}
            </div>
          )}

          <div className="video-readiness">
            <div className="video-readiness__head"><div><span className="eyebrow">Live checks</span><h3>Server-reported readiness</h3></div><span className={`preview-badge preview-badge--${video.runtime?.state ?? "setup-required"}`}>{runtimeLabel}</span></div>
            <div className="readiness-list">
              <span>{video.runtime?.engine.tested ? <Check size={14} /> : <TriangleAlert size={14} />}<span><strong>Isolated engine</strong><small>{video.runtime ? `${video.runtime.engine.name} · ${video.runtime.engine.tested ? "smoke tested" : "setup incomplete"}` : "Waiting for server status"}</small></span></span>
              <span>{video.runtime?.model.smoke_tested ? <Check size={14} /> : <Cpu size={14} />}<span><strong>Converted model</strong><small>{video.runtime ? `${video.runtime.model.source} · ${video.runtime.model.cached ? "cached" : "not cached"}` : "Waiting for server status"}</small></span></span>
              <span><ShieldCheck size={14} /><span><strong>License & isolation</strong><small>{capability ? `${capability.model.license} · ${capability.isolation}` : "Read from capability registry"}</small></span></span>
            </div>
            {video.runtime?.reasons?.length ? <ul className="video-reason-list">{video.runtime.reasons.map((reason) => <li key={reason.code}>{reason.message}</li>)}</ul> : null}
          </div>
        </section>
      </div>

      <section className="panel video-boundary-strip">
        <span className="video-boundary-icon"><ShieldCheck size={18} /></span>
        <span><strong>One serialized media queue</strong><small>{video.runtime?.active_media_job ? "New video work waits safely while the active media job finishes." : "Photo and video models never compete for unified memory."}</small></span>
        <button type="button" className="button button--ghost" onClick={() => void video.refresh()} disabled={video.loading}><RefreshCw className={video.loading ? "is-spinning" : ""} size={14} /> Refresh status</button>
      </section>
    </div>
  );
}

function RestorePage({
  restoreImage,
  restoreName,
  onFile,
  notify,
}: {
  restoreImage: string | null;
  restoreName: string;
  onFile: (event: ChangeEvent<HTMLInputElement>) => void;
  notify: (message: string) => void;
}) {
  const [compare, setCompare] = useState(58);
  const [preset, setPreset] = useState("Faithful");
  const [model, setModel] = useState("3B");
  const [target, setTarget] = useState("2×");
  const [preserveMetadata, setPreserveMetadata] = useState(true);
  const [locationMode, setLocationMode] = useState("Private");
  const [manualLocation, setManualLocation] = useState("");

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow="Restore"
        title="Bring back the detail. Keep the memory."
        copy="A faithful SeedVR2 workflow that never overwrites your original photograph."
        action={<span className="context-pill context-pill--safe"><ShieldCheck size={14} /> Original protected</span>}
      />
      <div className="restore-grid">
        <section className="panel compare-panel">
          <div className="canvas-toolbar">
            <span><WandSparkles size={15} /> Before & after</span>
            <div><button type="button" aria-label="View at one hundred percent">100%</button><button type="button" aria-label="More comparison options"><MoreHorizontal size={17} /></button></div>
          </div>
          <ComparisonStage src={restoreImage} value={compare} onChange={setCompare} />
          <div className="photo-meta">
            <span><ImageIcon size={15} /><span><strong>{restoreName || "travel_photo.jpg"}</strong><small>Original · Preview only</small></span></span>
            <span className="photo-meta-values"><small>Target</small><strong>{target} upscale</strong></span>
          </div>
        </section>

        <aside className="panel restore-controls" aria-label="Restore controls">
          <label className="upload-tile" htmlFor="restore-upload">
            <Upload size={19} />
            <span><strong>{restoreName || "Choose a photograph"}</strong><small>JPG, PNG, HEIC or WebP</small></span>
            <input id="restore-upload" type="file" accept="image/*" onChange={onFile} />
          </label>

          <div className="field-stack">
            <span className="field-label">Restoration recipe</span>
            <div className="recipe-grid">
              {["Faithful", "Soft photo", "Compression", "Print ready"].map((item) => (
                <button type="button" key={item} className={preset === item ? "is-active" : ""} onClick={() => setPreset(item)}>{item}{item === "Faithful" && <Check size={14} />}</button>
              ))}
            </div>
          </div>

          <div className="control-pair">
            <div className="field-stack"><span className="field-label">SeedVR2 model</span><Segmented label="SeedVR2 model" options={["3B", "7B"]} value={model} onChange={setModel} /></div>
            <div className="field-stack"><span className="field-label">Target size</span><Segmented label="Target size" options={["2×", "3×", "2160p"]} value={target} onChange={setTarget} /></div>
          </div>

          <div className="safety-box">
            <div className="setting-row"><span><strong>Keep camera and date</strong><small>Copies safe EXIF data into the export.</small></span><Toggle checked={preserveMetadata} onChange={setPreserveMetadata} label="Keep camera and date" /></div>
          </div>

          <div className="location-card">
            <div className="location-card__head">
              <span><MapPin size={16} /></span>
              <div><strong>Place information</strong><small>Decide if and how location is used.</small></div>
            </div>
            <Segmented
              label="Photo location privacy"
              options={["Private", "EXIF review", "Manual"]}
              value={locationMode}
              onChange={setLocationMode}
            />
            {locationMode === "Private" && (
              <p className="location-note"><LockKeyhole size={14} /> GPS stays unread and is excluded from share-safe exports.</p>
            )}
            {locationMode === "EXIF review" && (
              <p className="location-note"><Info size={14} /> If GPS exists, it will be shown for confirmation first. Metadata can be missing or inaccurate.</p>
            )}
            {locationMode === "Manual" && (
              <label className="manual-location">
                <span>Place label <small>optional</small></span>
                <input value={manualLocation} onChange={(event) => setManualLocation(event.target.value)} placeholder="Add your own place label" />
                <small>Saved as your note—not as a verified fact about the photograph.</small>
              </label>
            )}
          </div>

          <div className="restore-summary"><span><Gauge size={16} /><span><strong>{model === "3B" ? "Balanced memory" : "Maximum detail"}</strong><small>{model === "3B" ? "Best starting point for a photo batch." : "Slower, with a larger memory footprint."}</small></span></span><button type="button" aria-label="Open memory details"><ChevronDown size={15} /></button></div>
          <div className="button-row button-row--stack-mobile">
            <button className="button button--primary button--grow" type="button" onClick={() => notify("Restore recipe prepared. Backend connection comes next.")}><Play size={16} /> Preview restore</button>
            <button className="button button--icon" type="button" aria-label="Add a folder"><FolderOpen size={18} /></button>
          </div>
        </aside>
      </div>

      <section className="batch-strip panel">
        <div><span className="batch-icon"><GalleryVerticalEnd size={19} /></span><span><strong>Have a whole holiday folder?</strong><small>Preview one photo, then reuse the same safe recipe across the set.</small></span></div>
        <button type="button" className="button button--ghost" onClick={() => notify("Folder workflow is ready for backend integration.")}>Build a batch <ArrowRight size={15} /></button>
      </section>
    </div>
  );
}

function TimeLensPage({ restoreImage }: { restoreImage: string | null }) {
  const [compare, setCompare] = useState(52);
  const [year, setYear] = useState(1974);
  const [color, setColor] = useState(true);
  const [grain, setGrain] = useState(true);

  return (
    <div className="page-stack time-lens-page">
      <SectionHeading
        eyebrow="Time Lens"
        title="A careful idea for guided restoration"
        copy="This workflow is visible for product exploration, but no model-backed Time Lens processing is connected in this release."
        action={<span className="context-pill context-pill--unavailable"><LockKeyhole size={14} /> Not available</span>}
      />
      <section className="panel feature-unavailable" role="status">
        <span><Clock3 size={18} /></span>
        <div>
          <strong>Design preview only</strong>
          <p>The controls below are intentionally disabled. No photo is uploaded, transformed or interpreted by Time Lens.</p>
        </div>
      </section>
      <section className="time-hero panel time-lens-disabled" aria-disabled="true">
        <div className="time-visual">
          <ComparisonStage src={restoreImage} value={compare} onChange={setCompare} variant="archive" compact disabled />
          <div className="time-year"><small>Working era</small><strong>{year}</strong><span>Set manually</span></div>
        </div>
        <div className="time-story">
          <span className="eyebrow">A guided restoration</span>
          <h3>Keep the texture of the moment.</h3>
          <p>Time Lens separates repair from interpretation, so faces, places and the character of the original remain yours.</p>
          <label className="year-control"><span><strong>Possible era</strong><small>{year < 1980 ? "Film archive" : year < 2000 ? "Late analogue" : "Early digital"}</small></span><input type="range" min="1940" max="2026" value={year} onChange={(event) => setYear(Number(event.target.value))} disabled /></label>
          <p className="truth-note"><Info size={14} /> This era is a creative input, not a verified capture date. Confirm it from your own records before saving.</p>
          <div className="time-toggles">
            <div className="setting-row"><span><strong>Natural colour recovery</strong><small>Balanced skin tones and faded dyes.</small></span><Toggle checked={color} onChange={setColor} label="Natural colour recovery" disabled /></div>
            <div className="setting-row"><span><strong>Keep original grain</strong><small>Preserves the medium instead of polishing it away.</small></span><Toggle checked={grain} onChange={setGrain} label="Keep original grain" disabled /></div>
          </div>
        </div>
      </section>
      <section className="timeline panel time-lens-disabled" aria-label="Unavailable Time Lens workflow" aria-disabled="true">
        {["Original scan", "Repair", "Tone recovery", "Story export"].map((step, index) => (
          <div className={`timeline-step ${index === 0 ? "is-active" : ""}`} key={step}>
            <span>{index === 0 ? <Check size={14} /> : index + 1}</span>
            <div><strong>{step}</strong><small>{["Protected master", "Dust, tears, softness", "Colour and contrast", "Photo plus context"][index]}</small></div>
          </div>
        ))}
      </section>
    </div>
  );
}

function LibraryPage() {
  const [filter, setFilter] = useState("All work");
  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow="Library"
        title="Everything you have made, still traceable"
        copy="Projects, originals, recipes and exports live together—without hiding the source."
        action={<button className="button button--ghost" type="button"><Plus size={16} /> New collection</button>}
      />
      <div className="library-tools panel">
        <label className="search-field"><Search size={16} /><span className="sr-only">Search library</span><input placeholder="Search by project, tag or prompt" /></label>
        <div className="filter-tabs" aria-label="Library filter">
          {["All work", "Restored", "Created", "Favourites"].map((item) => <button type="button" key={item} className={filter === item ? "is-active" : ""} onClick={() => setFilter(item)}>{item}</button>)}
        </div>
        <button type="button" className="button button--icon" aria-label="Library display settings"><SlidersHorizontal size={17} /></button>
      </div>
      <div className="library-grid">
        {libraryItems.map((item) => (
          <article className="library-card" key={item.title}>
            <div className={`library-art library-art--${item.art}`}><MockPhoto variant={item.art} /><button type="button" aria-label={`Open options for ${item.title}`}><MoreHorizontal size={17} /></button><span>{item.tag}</span></div>
            <div className="library-card-copy"><div><h3>{item.title}</h3><p>{item.meta}</p></div><ArrowRight size={17} /></div>
          </article>
        ))}
      </div>
    </div>
  );
}

function ModelsPage({ video }: { video: VideoWorkspaceState }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const videoCards = useMemo(() => (video.registry?.capabilities ?? []).map((capability) => ({
    name: capability.label,
    role: capability.operations.map(humanizeVideoStage).join(" · "),
    copy: capability.availability_reason,
    tags: [
      capability.model.license,
      `${capability.parameters.width.fixed} × ${capability.parameters.height.fixed}`,
      `${capability.parameters.fps.fixed} fps`,
    ],
    status: capability.availability === "ready" && video.runtime?.ready
      ? video.runtime.state === "busy" ? "Busy" : "Ready"
      : "Setup required",
    tone: "blue",
    kind: "Video",
  })), [video.registry?.capabilities, video.runtime]);
  const catalogCards = useMemo(() => [...modelCards, ...videoCards], [videoCards]);
  const filteredModels = useMemo(() => catalogCards.filter((model) => {
    const matchesQuery = `${model.name} ${model.role} ${model.tags.join(" ")}`.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === "All" || model.kind === filter || model.tags.includes(filter);
    return matchesQuery && matchesFilter;
  }), [catalogCards, filter, query]);
  const showVideoConnection = videoCards.length === 0
    && (filter === "All" || filter === "Video")
    && "video runtime connection".includes(query.trim().toLowerCase());
  const runtimeLabel = videoRuntimeLabel(video);

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow="Models"
        title="A catalog that explains itself"
        copy="Choose photo and video models by capability, memory, maturity and license—not by cryptic checkpoint names."
        action={<span className={`context-pill video-state-pill video-state-pill--${video.runtime?.state ?? "setup-required"}`}><MonitorPlay size={14} /> Video {runtimeLabel.toLowerCase()}</span>}
      />
      <div className="model-toolbar panel">
        <label className="search-field"><Search size={16} /><span className="sr-only">Search models</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models or capabilities" /></label>
        <div className="filter-tabs" aria-label="Model type">
          {["All", "Photo", "Video", "Restore", "Edit"].map((item) => <button type="button" key={item} className={filter === item ? "is-active" : ""} onClick={() => setFilter(item)}>{item}</button>)}
        </div>
      </div>
      <div className="model-grid">
        {showVideoConnection && (
          <article className="model-card model-card--blue model-card--connection">
            <div className="model-card-top"><span className="model-glyph"><span /></span><span className="model-status">{runtimeLabel}</span></div>
            <span className="eyebrow">Video runtime</span>
            <h3>{video.loading ? "Reading server capabilities" : "No video capability reported"}</h3>
            <p>{video.error || "The local server has not returned a usable video capability yet. No model is claimed as available."}</p>
            <button className="model-action" type="button" onClick={() => void video.refresh()}><RefreshCw className={video.loading ? "is-spinning" : ""} size={15} /> Refresh server status</button>
          </article>
        )}
        {filteredModels.map((model) => (
          <article className={`model-card model-card--${model.tone}`} key={model.name}>
            <div className="model-card-top"><span className="model-glyph"><span /></span><span className="model-status">{model.status}</span></div>
            <span className="eyebrow">{model.role}</span>
            <h3>{model.name}</h3>
            <p>{model.copy}</p>
            <div className="model-tags">{model.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
            <button className="model-action" type="button">{model.kind === "Video" ? "Server capability" : "View capability card"} <ArrowRight size={15} /></button>
          </article>
        ))}
      </div>
    </div>
  );
}

function ActivityPage({ video }: { video: VideoWorkspaceState }) {
  const activeJob = video.runtime?.active_media_job;
  const runtimeLabel = videoRuntimeLabel(video);
  return (
    <div className="page-stack">
      <SectionHeading eyebrow="Activity" title="A truthful view of what your Mac is doing" copy="Photo and video work share one local media queue, while detailed video stages stay visible in the Video workspace." />
      <section className="activity-hero panel">
        <div className="activity-empty">
          <span>{video.runtime?.state === "busy" ? <LoaderCircle className="is-spinning" size={21} /> : video.runtime?.ready ? <Check size={21} /> : <TriangleAlert size={21} />}</span>
          <div>
            <h3>{activeJob ? `${activeJob.type === "photo_batch" ? "Photo restoration" : "Video generation"} is active` : video.runtime?.ready ? "The media queue is clear" : runtimeLabel}</h3>
            <p>{activeJob ? "New work waits safely until the active media job releases unified memory." : video.error || video.runtime?.reasons?.[0]?.message || "New local work will appear here when the server reports it."}</p>
          </div>
        </div>
        <div className="activity-metrics"><span><small>Queue</small><strong>{activeJob ? "Occupied" : video.runtime?.ready ? "Clear" : "Unknown"}</strong></span><span><small>Video runtime</small><strong>{runtimeLabel}</strong></span><span><small>Isolation</small><strong>{video.runtime ? humanizeVideoStage(video.runtime.isolation) : "—"}</strong></span></div>
      </section>
      <section className="panel history-panel">
        <SectionHeading eyebrow="History" title="No connected history feed" copy="This screen does not invent completed jobs. Generated video artifacts are available from the active Video workspace." />
      </section>
    </div>
  );
}

function SettingsPage({ video }: { video: VideoWorkspaceState }) {
  const [preserveOriginals, setPreserveOriginals] = useState(true);
  const [localOnly, setLocalOnly] = useState(true);
  const [batteryAware, setBatteryAware] = useState(true);
  const [stepPreviews, setStepPreviews] = useState(false);
  const runtimeLabel = videoRuntimeLabel(video);
  const runtimeDetail = video.error
    || video.runtime?.reasons?.[0]?.message
    || (video.runtime?.state === "busy"
      ? "The serialized media queue is currently occupied."
      : video.runtime?.ready
        ? "The server reports a tested isolated engine and converted local model."
        : "The local server has not reported a ready video runtime.");
  return (
    <div className="page-stack settings-page">
      <SectionHeading eyebrow="Settings" title="Make local AI feel predictable" copy="Defaults are conservative, private and reversible." />
      <div className="settings-grid">
        <section className="panel settings-section"><div className="settings-section-head"><span><ShieldCheck size={18} /></span><div><h3>Privacy & originals</h3><p>Control what leaves this Mac and what can be changed.</p></div></div><div className="settings-list"><div className="setting-row"><span><strong>Local-only server</strong><small>Blocks network access unless you explicitly enable sharing.</small></span><Toggle checked={localOnly} onChange={setLocalOnly} label="Local-only server" /></div><div className="setting-row"><span><strong>Never overwrite originals</strong><small>Every edit becomes a new, traceable version.</small></span><Toggle checked={preserveOriginals} onChange={setPreserveOriginals} label="Never overwrite originals" /></div></div></section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Gauge size={18} /></span><div><h3>Apple silicon performance</h3><p>Balance speed, memory and battery use.</p></div></div><div className="settings-list"><div className="setting-row"><span><strong>Battery-aware mode</strong><small>Uses gentler defaults while disconnected from power.</small></span><Toggle checked={batteryAware} onChange={setBatteryAware} label="Battery-aware mode" /></div><div className="setting-row"><span><strong>Step previews</strong><small>Show intermediate images during longer generations.</small></span><Toggle checked={stepPreviews} onChange={setStepPreviews} label="Step previews" /></div></div></section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Sparkles size={18} /></span><div><h3>AI assist connections</h3><p>Provider state is shown separately from creative capabilities.</p></div><span className="preview-badge">Preview</span></div><div className="settings-list"><div className="setting-row"><span><strong>Nativ local server</strong><small>Status and detected-model discovery only; no prompt or vision calls.</small></span><span className="setting-value">Status only</span></div><div className="setting-row"><span><strong>Local prompt refinement</strong><small>Existing Ollama or MLX path; not wired into this React preview.</small></span><span className="setting-value">Not connected</span></div></div></section>
        <section className="panel settings-section">
          <div className="settings-section-head"><span><Film size={18} /></span><div><h3>Video runtime</h3><p>{runtimeDetail}</p></div><span className={`preview-badge preview-badge--${video.runtime?.state ?? "setup-required"}`}>{runtimeLabel}</span></div>
          <div className="settings-list">
            <div className="setting-row"><span><strong>Isolated engine</strong><small>Status is read from the local runtime, never assumed by the interface.</small></span><span className={`setting-value ${video.runtime?.engine.tested ? "setting-value--safe" : ""}`}>{video.runtime?.engine.tested ? "Smoke tested" : video.runtime?.engine.configured ? "Configured" : "Setup needed"}</span></div>
            <div className="setting-row"><span><strong>Converted video model</strong><small>Ready requires a cached conversion and a successful server smoke proof.</small></span><span className={`setting-value ${video.runtime?.model.smoke_tested ? "setting-value--safe" : ""}`}>{video.runtime?.model.smoke_tested ? "Ready" : video.runtime?.model.cached ? "Cached · unverified" : "Not ready"}</span></div>
            <div className="setting-row"><span><strong>Media job isolation</strong><small>Photo and video jobs use the same server-controlled concurrency policy.</small></span><span className="setting-value">{video.runtime ? humanizeVideoStage(video.runtime.concurrency) : "Waiting"}</span></div>
          </div>
          <button type="button" className="button button--ghost button--compact settings-refresh" onClick={() => void video.refresh()} disabled={video.loading}><RefreshCw className={video.loading ? "is-spinning" : ""} size={14} /> Refresh runtime</button>
        </section>
        <section className="panel settings-section settings-section--wide"><div className="settings-section-head"><span><HardDrive size={18} /></span><div><h3>Storage</h3><p>Keep large local models and outputs understandable.</p></div><button className="button button--ghost" type="button">Choose folders</button></div><div className="storage-bar"><span style={{ width: "37%" }} /></div><div className="storage-legend"><span><i className="legend-dot legend-dot--models" />Models · —</span><span><i className="legend-dot legend-dot--outputs" />Outputs · —</span><span>Connect backend for live totals</span></div></section>
      </div>
    </div>
  );
}

export default function App() {
  const video = useVideoWorkspace();
  const [activePage, setActivePage] = useState<PageId>("home");
  const [restoreImage, setRestoreImage] = useState<string | null>(null);
  const [restoreName, setRestoreName] = useState("");
  const [notice, setNotice] = useState("");
  const [moreOpen, setMoreOpen] = useState(false);
  const searchRef = useRef<HTMLButtonElement>(null);
  const moreMenuFirstRef = useRef<HTMLButtonElement>(null);
  const activeMeta = navigation.find((item) => item.id === activePage) ?? navigation[0];

  useEffect(() => {
    if (!restoreImage) return undefined;
    return () => URL.revokeObjectURL(restoreImage);
  }, [restoreImage]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(""), 3200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === "Escape") setMoreOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (!moreOpen) return;
    window.requestAnimationFrame(() => moreMenuFirstRef.current?.focus());
  }, [moreOpen]);

  const onFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setRestoreImage(URL.createObjectURL(file));
    setRestoreName(file.name);
    setNotice("Photo loaded locally for preview.");
  };

  const navigate = (page: PageId) => {
    setActivePage(page);
    setMoreOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const page = (() => {
    switch (activePage) {
      case "home": return <HomePage onNavigate={navigate} />;
      case "create": return <CreatePage notify={setNotice} />;
      case "video": return <VideoPage video={video} notify={setNotice} />;
      case "restore": return <RestorePage restoreImage={restoreImage} restoreName={restoreName} onFile={onFile} notify={setNotice} />;
      case "time-lens": return <TimeLensPage restoreImage={restoreImage} />;
      case "library": return <LibraryPage />;
      case "models": return <ModelsPage video={video} />;
      case "activity": return <ActivityPage video={video} />;
      case "settings": return <SettingsPage video={video} />;
      default: return null;
    }
  })();

  return (
    <div className="app-shell">
      <a className="skip-link" href="#studio-main">Skip to workspace</a>
      <aside className="sidebar">
        <button type="button" className="brand" onClick={() => navigate("home")} aria-label="MLX Media home">
          <BrandMark />
          <span><strong>MLX</strong><small>MEDIA</small></span>
        </button>
        <nav className="primary-nav" aria-label="Studio navigation">
          <span className="nav-label">Workspace</span>
          {navigation.slice(0, 6).map((item) => {
            const Icon = item.icon;
            return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={18} /><span>{item.label}</span>{item.id === "restore" && <i>New</i>}</button>;
          })}
          <span className="nav-label nav-label--utility">System</span>
          {navigation.slice(6).map((item) => {
            const Icon = item.icon;
            return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={18} /><span>{item.label}</span></button>;
          })}
        </nav>
        <div className="sidebar-footer"><span className="mini-status"><i /> Local media</span><button type="button" aria-label="Open workspace menu"><MoreHorizontal size={17} /></button></div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className="mobile-brand"><BrandMark /><strong>MLX MEDIA</strong></div>
          <div className="page-identity"><span>{activeMeta.label}</span><small>{activeMeta.description}</small></div>
          <div className="topbar-actions">
            <button ref={searchRef} type="button" className="search-button" aria-label="Open command search"><Search size={16} /><span>Search studio</span><kbd><Command size={11} />K</kbd></button>
            <span className="local-pill"><span /> Local only</span>
            <button
              type="button"
              className="icon-button"
              aria-label={moreOpen ? "Close quick menu" : "Open quick menu"}
              aria-expanded={moreOpen}
              aria-controls="studio-quick-menu"
              onClick={() => setMoreOpen((open) => !open)}
            >
              <Menu size={18} />
            </button>
          </div>
        </header>
        <main id="studio-main" className="page-content" tabIndex={-1}>{page}</main>
      </div>

      <nav className="mobile-nav" aria-label="Mobile studio navigation">
        {navigation.slice(0, 5).map((item) => {
          const Icon = item.icon;
          return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-label={item.label} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={19} /><span>{item.label}</span></button>;
        })}
        <button
          type="button"
          className={activePage === "library" || activePage === "settings" || activePage === "models" || activePage === "activity" || moreOpen ? "is-active" : ""}
          aria-label="More tools"
          aria-expanded={moreOpen}
          aria-controls="studio-quick-menu"
          onClick={() => setMoreOpen((open) => !open)}
        >
          <Menu size={19} /><span>More</span>
        </button>
      </nav>
      {moreOpen && (
        <>
          <button className="quick-menu-scrim" type="button" aria-label="Close quick menu" onClick={() => setMoreOpen(false)} />
          <aside className="quick-menu" id="studio-quick-menu" aria-label="Quick menu">
            <div className="quick-menu__header"><span className="eyebrow">Studio</span><strong>More tools</strong></div>
            {navigation.slice(5).map((item, index) => {
              const Icon = item.icon;
              return (
                <button
                  ref={index === 0 ? moreMenuFirstRef : undefined}
                  type="button"
                  key={item.id}
                  className={activePage === item.id ? "is-active" : ""}
                  aria-current={activePage === item.id ? "page" : undefined}
                  onClick={() => navigate(item.id)}
                >
                  <span className="quick-menu__icon"><Icon size={17} /></span>
                  <span><strong>{item.label}</strong><small>{item.description}</small></span>
                  <ArrowRight size={15} />
                </button>
              );
            })}
          </aside>
        </>
      )}
      {notice && <div className="toast" role="status"><Check size={16} />{notice}</div>}
    </div>
  );
}
