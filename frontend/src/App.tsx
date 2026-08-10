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
  VideoApiError,
  type VideoCapabilityRegistry,
  type VideoGenerationRequest,
  type VideoJobResponse,
  type VideoRuntimeStatus,
  type VideoWorkspaceState,
} from "./videoApi";
import { languageOptions, useI18n, type Translate } from "./i18n";

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

function buildNavigation(t: Translate): NavItem[] {
  return [
    { id: "home", label: t("nav.home"), description: t("nav.homeDescription"), icon: Aperture },
    { id: "create", label: t("nav.images"), description: t("nav.imagesDescription"), icon: Sparkles },
    { id: "video", label: t("nav.video"), description: t("nav.videoDescription"), icon: Film },
    { id: "restore", label: t("nav.restore"), description: t("nav.restoreDescription"), icon: WandSparkles },
    { id: "time-lens", label: t("nav.timeLens"), description: t("nav.timeLensDescription"), icon: History },
    { id: "library", label: t("nav.library"), description: t("nav.libraryDescription"), icon: Library },
    { id: "models", label: t("nav.models"), description: t("nav.modelsDescription"), icon: Layers3 },
    { id: "activity", label: t("nav.activity"), description: t("nav.activityDescription"), icon: Activity },
    { id: "settings", label: t("nav.settings"), description: t("nav.settingsDescription"), icon: Settings },
  ];
}

function buildQuickActions(t: Translate) {
  return [
  {
    id: "restore" as PageId,
    title: t("quick.restoreTitle"),
    copy: t("quick.restoreCopy"),
    icon: WandSparkles,
    tone: "cyan",
  },
  {
    id: "create" as PageId,
    title: t("quick.createTitle"),
    copy: t("quick.createCopy"),
    icon: Sparkles,
    tone: "violet",
  },
  {
    id: "video" as PageId,
    title: t("quick.videoTitle"),
    copy: t("quick.videoCopy"),
    icon: Film,
    tone: "blue",
  },
  {
    id: "time-lens" as PageId,
    title: t("quick.timeLensTitle"),
    copy: t("quick.timeLensCopy"),
    icon: History,
    tone: "magenta",
  },
  ];
}

function buildModelCards(t: Translate) {
  return [
  {
    name: "SeedVR2 3B",
    role: t("models.seed3Role"),
    copy: t("models.seed3Copy"),
    tags: [t("common.restore"), "3B", t("models.promptless")],
    status: t("common.recommended"),
    tone: "cyan",
    kind: "Photo",
  },
  {
    name: "SeedVR2 7B",
    role: t("models.seed7Role"),
    copy: t("models.seed7Copy"),
    tags: [t("common.restore"), "7B", t("models.highMemory")],
    status: t("common.available"),
    tone: "violet",
    kind: "Photo",
  },
  {
    name: "FLUX.2 Klein",
    role: t("models.fluxRole"),
    copy: t("models.fluxCopy"),
    tags: [t("models.generate"), t("common.edit"), "4B / 9B"],
    status: t("common.catalog"),
    tone: "blue",
    kind: "Photo",
  },
  {
    name: "Krea 2 Turbo",
    role: t("models.kreaRole"),
    copy: t("models.kreaCopy"),
    tags: [t("models.generate"), t("models.turbo"), t("models.gated")],
    status: t("common.catalog"),
    tone: "orange",
    kind: "Photo",
  },
  {
    name: "Ideogram 4",
    role: t("models.ideogramRole"),
    copy: t("models.ideogramCopy"),
    tags: [t("models.text"), t("models.layout"), t("models.structured")],
    status: t("common.catalog"),
    tone: "magenta",
    kind: "Photo",
  },
  {
    name: "Qwen Image Edit",
    role: t("models.qwenRole"),
    copy: t("models.qwenCopy"),
    tags: [t("common.edit"), t("models.multiImage"), "20B"],
    status: t("common.catalog"),
    tone: "green",
    kind: "Photo",
  },
  ];
}

function buildLibraryItems(t: Translate) {
  return [
    { title: t("library.goldenTitle"), meta: t("library.goldenMeta"), tag: t("library.travel"), art: "sunset" },
    { title: t("library.familyTitle"), meta: t("library.familyMeta"), tag: t("library.archive"), art: "archive" },
    { title: t("library.coastTitle"), meta: t("library.coastMeta"), tag: t("library.create"), art: "coast" },
    { title: t("library.nightTitle"), meta: t("library.nightMeta"), tag: t("library.travel"), art: "night" },
    { title: t("library.posterTitle"), meta: t("library.posterMeta"), tag: t("library.typography"), art: "poster" },
    { title: t("library.printTitle"), meta: t("library.printMeta"), tag: t("common.restore"), art: "print" },
  ];
}

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
  getLabel = (option) => option,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (value: T) => void;
  getLabel?: (value: T) => string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          type="button"
          key={option}
          className={value === option ? "is-active" : ""}
          aria-pressed={value === option}
          onClick={() => onChange(option)}
        >
          {getLabel(option)}
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
  const { t } = useI18n();
  const clipStyle = { "--compare": `${value}%` } as CSSProperties;

  return (
    <div className={`comparison ${compact ? "comparison--compact" : ""}`}>
      <div className="comparison-stage" style={clipStyle}>
        <div className="comparison-layer comparison-layer--before">
          <PhotoLayer src={src} variant={variant} alt={t("comparison.originalPreview")} />
        </div>
        <div className="comparison-layer comparison-layer--after">
          <PhotoLayer src={src} variant={variant} alt={t("comparison.restoredPreview")} />
        </div>
        <span className="comparison-label comparison-label--left">{t("comparison.original")}</span>
        <span className="comparison-label comparison-label--right">{t("comparison.restored")}</span>
        <span className="comparison-line" aria-hidden="true">
          <span><ChevronDown size={14} /></span>
        </span>
      </div>
      <label className="comparison-control">
        <span className="sr-only">{t("comparison.aria")}</span>
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
  const { t } = useI18n();
  const quickActions = useMemo(() => buildQuickActions(t), [t]);
  const libraryItems = useMemo(() => buildLibraryItems(t), [t]);
  return (
    <div className="page-stack home-page">
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="release-pill"><span /> {t("home.release")}</span>
          <h1>{t("home.titlePhotoVideo")}<br />{t("home.titleOneStudio")}</h1>
          <p>
            {t("home.intro")}
          </p>
          <div className="button-row">
            <button className="button button--primary" type="button" onClick={() => onNavigate("restore")}>
              {t("home.restorePhoto")} <ArrowRight size={16} />
            </button>
            <button className="button button--ghost" type="button" onClick={() => onNavigate("create")}>
              {t("home.createImage")}
            </button>
            <button className="button button--ghost" type="button" onClick={() => onNavigate("video")}>
              {t("home.exploreVideo")}
            </button>
          </div>
        </div>
        <div className="hero-visual" aria-label={t("home.visualAria")}>
          <img className="hero-emblem" src="/assets/mlx-media-emblem.png" alt="" />
          <div className="hero-orbit hero-orbit--one" />
          <div className="hero-orbit hero-orbit--two" />
          <div className="floating-output floating-output--back"><MockPhoto variant="coast" /></div>
          <div className="floating-output floating-output--front"><MockPhoto variant="sunset" /></div>
          <div className="hero-status"><ShieldCheck size={15} /> {t("home.localCapabilityAware")}</div>
        </div>
      </section>

      <section>
        <SectionHeading
          eyebrow={t("home.startHere")}
          title={t("home.whatMake")}
          copy={t("home.intentCopy")}
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
            eyebrow={t("home.continue")}
            title={t("home.recentWork")}
            action={<button className="text-button" type="button" onClick={() => onNavigate("library")}>{t("home.openLibrary")}</button>}
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
            <div><span className="eyebrow">{t("home.appleSilicon")}</span><h3>{t("home.privateWorkspace")}</h3></div>
            <span className="status-dot" aria-label={t("home.localWorkspace")} />
          </div>
          <div className="system-meter"><span style={{ width: "32%" }} /></div>
          <div className="system-stats">
            <span><small>{t("home.memoryProfile")}</small><strong>{t("common.balanced")}</strong></span>
            <span><small>{t("home.queue")}</small><strong>{t("home.serverManaged")}</strong></span>
          </div>
          <p><CloudOff size={14} /> {t("home.mediaStays")}</p>
        </div>
      </section>
    </div>
  );
}

function CreatePage({ notify }: { notify: (message: string) => void }) {
  const { t } = useI18n();
  const [quality, setQuality] = useState<"speed" | "balanced" | "quality">("balanced");
  const [prompt, setPrompt] = useState("");

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow={t("library.create")}
        title={t("create.title")}
        copy={t("create.copy")}
        action={<span className="context-pill"><Sparkles size={14} /> {t("create.autoModel")}</span>}
      />
      <div className="studio-grid">
        <section className="panel prompt-panel" aria-label={t("create.creationBrief")}>
          <label className="field-label" htmlFor="create-prompt">{t("create.creativeBrief")}</label>
          <textarea id="create-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder={t("create.promptPlaceholder")} rows={6} />
          <div className="prompt-tools">
            <button type="button" className="tool-button"><Plus size={15} /> {t("create.reference")}</button>
            <button type="button" className="tool-button"><Tag size={15} /> {t("create.style")}</button>
            <button type="button" className="tool-button" onClick={() => notify(t("create.refineUnavailable"))}><RefreshCw size={15} /> {t("create.localRefine")}</button>
          </div>
          <div className="field-stack">
            <span className="field-label">{t("create.outputIntent")}</span>
            <div className="intent-list">
              <button type="button" className="intent-chip is-active">{t("create.photographic")}</button>
              <button type="button" className="intent-chip">{t("create.illustrative")}</button>
              <button type="button" className="intent-chip">{t("create.textLayout")}</button>
            </div>
          </div>
          <div className="panel-divider" />
          <div className="field-stack">
            <span className="field-label">{t("create.performance")}</span>
            <Segmented label={t("create.generationQuality")} options={["speed", "balanced", "quality"]} value={quality} onChange={setQuality} getLabel={(value) => t(value === "speed" ? "create.speed" : value === "quality" ? "create.quality" : "common.balanced")} />
          </div>
          <button className="button button--primary button--wide" type="button" onClick={() => notify(t("create.preparedNotice"))}>
            <Sparkles size={17} /> {t("create.prepare")}
          </button>
        </section>

        <section className="panel canvas-panel" aria-label={t("create.imagePreview")}>
          <div className="canvas-toolbar">
            <span><span className="status-dot" /> {t("create.previewCanvas")}</span>
            <div><button type="button" aria-label={t("create.fitPreview")}><Eye size={16} /></button><button type="button" aria-label={t("create.morePreview")}><MoreHorizontal size={17} /></button></div>
          </div>
          <div className="create-canvas"><MockPhoto variant="night" /></div>
          <div className="canvas-caption"><span>{t("create.conceptPreview")}</span><span>{t("create.outputAuto")}</span></div>
        </section>

        <aside className="panel inspector-panel" aria-label={t("create.inspectorAria")}>
          <div className="inspector-title"><SlidersHorizontal size={17} /><strong>{t("create.inspector")}</strong><span>{t("create.simple")}</span></div>
          <div className="inspector-group"><small>{t("create.modelRoute")}</small><button type="button" className="select-button"><span><strong>{t("common.auto")}</strong><small>{t("create.bestFit")}</small></span><ChevronDown size={16} /></button></div>
          <div className="inspector-group"><small>{t("create.canvas")}</small><div className="ratio-grid"><button className="is-active" type="button">1:1</button><button type="button">4:5</button><button type="button">3:2</button></div></div>
          <div className="inspector-group"><small>{t("create.variations")}</small><div className="stepper"><button type="button" aria-label={t("create.decreaseVariations")}>−</button><strong>4</strong><button type="button" aria-label={t("create.increaseVariations")}>+</button></div></div>
          <div className="info-callout"><CircleGauge size={17} /><span><strong>{t("create.balancedRoute")}</strong><small>{t("create.balancedRouteCopy")}</small></span></div>
          <button type="button" className="text-button expert-link"><Settings size={14} /> {t("create.expertControls")}</button>
        </aside>
      </div>
    </div>
  );
}

function useVideoWorkspace(): VideoWorkspaceState {
  const { t } = useI18n();
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
      setError(localizedVideoError(t, nextError, "video.errorService"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void fetchVideoStatus()
        .then((nextRuntime) => {
          setRuntime(nextRuntime);
          setError("");
        })
        .catch((nextError) => {
          setError(localizedVideoError(t, nextError, "video.errorService"));
        });
    }, 8000);
    return () => window.clearInterval(timer);
  }, [refresh, t]);

  return { registry, runtime, loading, error, refresh };
}

function translateTechnicalValue(
  t: Translate,
  value: string,
  fallback?: Parameters<Translate>[0],
): string {
  const normalized = value.replaceAll("_", "-").toLowerCase();
  const keys: Partial<Record<string, Parameters<Translate>[0]>> = {
    loading: "status.loading",
    validating: "status.validating",
    denoising: "status.denoising",
    decoding: "status.decoding",
    muxing: "status.muxing",
    completed: "status.completed",
    queued: "status.queued",
    running: "status.running",
    failed: "status.failed",
    cancelled: "status.cancelled",
    auto: "status.auto",
    default: "status.default",
    "separate-subprocess": "status.separateSubprocess",
    "serialized-with-media-queue": "status.serializedQueue",
    "process-group-with-stage-checks": "status.stageCancel",
    "text-to-video": "status.textToVideo",
  };
  const key = keys[normalized];
  return key ? t(key) : fallback ? t(fallback) : value;
}

function localizedVideoError(
  t: Translate,
  error: unknown,
  fallback: Parameters<Translate>[0],
): string {
  if (error instanceof VideoApiError) {
    return t("video.httpError", { status: error.status });
  }
  return t(fallback);
}

function videoRuntimeLabel(video: VideoWorkspaceState, t: Translate): string {
  if (video.loading && !video.runtime) return t("common.checking");
  if (video.error && !video.runtime) return t("common.unavailable");
  if (video.runtime?.state === "ready") return t("common.ready");
  if (video.runtime?.state === "busy") return t("common.busy");
  return t("common.setupRequired");
}

function translateRuntimeReason(t: Translate, code: string): string {
  if (code === "unsupported-platform") return t("status.setupReasonPlatform");
  if (code === "runner-python-not-configured" || code === "runner-python-unavailable") return t("status.setupReasonPython");
  if (code === "engine-pin-invalid") return t("status.setupReasonEngine");
  if (code === "model-not-configured" || code === "model-unavailable" || code === "model-incomplete") return t("status.setupReasonModel");
  if (code === "tokenizer-unavailable") return t("status.setupReasonTokenizer");
  if (code === "smoke-provenance-missing" || code === "smoke-provenance-invalid") return t("status.setupReasonSmoke");
  return t("settings.runtimeNotReady");
}

function VideoPage({ video, notify }: { video: VideoWorkspaceState; notify: (message: string) => void }) {
  const { locale, t } = useI18n();
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
        setJobError(next.error ? t("video.failedCopy") : "");
        if (isVideoJobTerminal(next.status)) void video.refresh();
      } catch (nextError) {
        if (!disposed) setJobError(localizedVideoError(t, nextError, "video.errorRead"));
      }
    };

    void update();
    const timer = window.setInterval(() => void update(), 1200);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [job?.status, jobId, t, video.refresh]);

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
  const duration = capability
    ? new Intl.NumberFormat(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      .format(frames / capability.parameters.fps.fixed)
    : "—";
  const runtimeLabel = videoRuntimeLabel(video, t);

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
      notify(t("video.noticeQueued"));
      void video.refresh();
    } catch (nextError) {
      setJobError(localizedVideoError(t, nextError, "video.errorSubmit"));
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
      notify(t("video.noticeCancel"));
      void video.refresh();
    } catch (nextError) {
      setJobError(localizedVideoError(t, nextError, "video.errorCancel"));
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
        eyebrow={t("nav.video")}
        title={t("video.title")}
        copy={t("video.copy")}
        action={(
          <span className={`context-pill video-state-pill video-state-pill--${video.runtime?.state ?? "setup-required"}`}>
            {video.loading && !video.runtime ? <LoaderCircle className="is-spinning" size={14} /> : <MonitorPlay size={14} />}
            {runtimeLabel}
          </span>
        )}
      />

      <div className="video-studio-grid">
        <section className="panel video-brief-panel" aria-label={t("video.controlsAria")}>
          <div className="video-live-header">
            <div><span className="eyebrow">{t("video.serverCapability")}</span><h3>{capability?.label ?? t("video.noCapability")}</h3></div>
            <button type="button" className="button button--ghost button--compact" onClick={() => void video.refresh()} disabled={video.loading}>
              <RefreshCw className={video.loading ? "is-spinning" : ""} size={14} /> {t("common.refresh")}
            </button>
          </div>

          {capabilities.length > 1 && (
            <label className="field-stack" htmlFor="video-capability">
              <span className="field-label">{t("video.capability")}</span>
              <select id="video-capability" value={capabilityId} onChange={(event) => setCapabilityId(event.target.value)}>
                {capabilities.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
              </select>
            </label>
          )}

          <label className="field-stack" htmlFor="video-prompt">
            <span className="field-label">{t("video.motionBrief")}</span>
            <textarea
              id="video-prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={6}
              maxLength={capability?.parameters.prompt.max_length}
              placeholder={t("video.promptPlaceholder")}
              disabled={!capability || jobActive}
            />
            <small className="field-help">{prompt.length}{capability ? ` / ${capability.parameters.prompt.max_length}` : ""} {t("video.characters")}</small>
          </label>

          <div className="video-output-grid video-output-grid--live">
            <label className="field-stack">
              <span className="field-label">{t("video.frames")}</span>
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
              <small>{capability ? `${capability.parameters.num_frames.minimum}–${capability.parameters.num_frames.maximum} · ${capability.parameters.num_frames.rule}` : t("common.fromServer")}</small>
            </label>
            <label className="field-stack">
              <span className="field-label">{t("video.steps")}</span>
              <input
                type="number"
                min={capability?.parameters.steps.minimum}
                max={capability?.parameters.steps.maximum}
                value={steps}
                onChange={(event) => setSteps(Number(event.target.value))}
                disabled={!capability || jobActive}
              />
              <small>{capability ? `${capability.parameters.steps.minimum}–${capability.parameters.steps.maximum}` : t("common.fromServer")}</small>
            </label>
            <label className="field-stack">
              <span className="field-label">{t("video.tiling")}</span>
              <select value={tiling} onChange={(event) => setTiling(event.target.value)} disabled={!capability || jobActive}>
                {(capability?.parameters.tiling.allowed ?? []).map((item) => <option key={item} value={item}>{translateTechnicalValue(t, item)}</option>)}
              </select>
              <small>{t("video.allowedModes")}</small>
            </label>
            <label className="field-stack">
              <span className="field-label">{t("video.seed")} <small>{t("common.optional")}</small></span>
              <input
                className={capability && !seedValid ? "is-invalid" : ""}
                type="number"
                min={capability?.parameters.seed.minimum}
                max={capability?.parameters.seed.maximum}
                value={seed}
                placeholder={t("status.auto")}
                onChange={(event) => setSeed(event.target.value)}
                disabled={!capability || jobActive}
              />
              <small>{t("video.randomSeed")}</small>
            </label>
          </div>

          {capability && (
            <div className="video-spec-strip">
              <span><small>{t("video.output")}</small><strong>{capability.parameters.width.fixed} × {capability.parameters.height.fixed}</strong></span>
              <span><small>{t("video.frameRate")}</small><strong>{capability.parameters.fps.fixed} {t("video.fpsUnit")}</strong></span>
              <span><small>{t("video.duration")}</small><strong>{duration} {t("video.secondsUnit")}</strong></span>
              <span><small>{t("video.scheduler")}</small><strong>{capability.parameters.scheduler.fixed}</strong></span>
              <span><small>{t("video.container")}</small><strong>{capability.output.container.toUpperCase()}</strong></span>
            </div>
          )}

          {capability && !frameValid && <p className="validation-note"><TriangleAlert size={14} /> {t("video.framesError", { rule: capability.parameters.num_frames.rule })}</p>}
          {capability && !seedValid && <p className="validation-note"><TriangleAlert size={14} /> {t("video.seedError")}</p>}
          {video.error && <p className="video-inline-error" role="alert"><TriangleAlert size={15} /> {video.error}</p>}
          {jobError && <p className="video-inline-error" role="alert"><TriangleAlert size={15} /> {jobError}</p>}

          <div className="button-row video-submit-row">
            <button className="button button--primary button--grow" type="button" disabled={!canSubmit} onClick={() => void submit()}>
              {submitting ? <LoaderCircle className="is-spinning" size={17} /> : <Film size={17} />}
              {submitting ? t("video.submitting") : runtimeReady ? t("video.submit") : runtimeLabel}
            </button>
            {jobActive && (
              <button className="button button--danger" type="button" onClick={() => void cancel()} disabled={cancelling}>
                {cancelling ? <LoaderCircle className="is-spinning" size={15} /> : <Square size={14} />} {t("common.cancel")}
              </button>
            )}
          </div>
        </section>

        <section className="panel video-preview-panel video-preview-panel--live" aria-label={t("video.outputAria")}>
          <div className="canvas-toolbar">
            <span><span className={`status-dot status-dot--${job?.status ?? video.runtime?.state ?? "setup-required"}`} /> {t("video.localOutput")}</span>
            <span className={`preview-badge preview-badge--${job?.status ?? video.runtime?.state ?? "setup-required"}`}>{job ? translateTechnicalValue(t, job.status) : runtimeLabel}</span>
          </div>

          {videoUrl ? (
            <div className="video-result">
              <video controls preload="metadata" src={videoUrl} aria-label={t("video.generatedAria")} />
              <div className="video-result-meta">
                <span><strong>{t("video.complete")}</strong><small>{capability?.label ?? job?.result?.capability_id}</small></span>
                <a className="button button--ghost button--compact" href={videoUrl} download><Download size={14} /> {t("video.download")}</a>
              </div>
              <div className="artifact-links" aria-label={t("video.artifactAria")}>
                {localArtifactUrl(artifactUrls?.provenance) && <a href={localArtifactUrl(artifactUrls?.provenance)} target="_blank" rel="noreferrer">{t("video.provenance")} <ExternalLink size={12} /></a>}
                {localArtifactUrl(artifactUrls?.request) && <a href={localArtifactUrl(artifactUrls?.request)} target="_blank" rel="noreferrer">{t("video.request")} <ExternalLink size={12} /></a>}
              </div>
            </div>
          ) : (
            <div className={`video-job-stage video-job-stage--${job?.status ?? video.runtime?.state ?? "setup-required"}`}>
              <span className="video-job-stage__icon">
                {jobActive || video.loading ? <LoaderCircle className="is-spinning" size={28} /> : video.runtime?.state === "ready" ? <Check size={28} /> : video.runtime?.state === "busy" ? <Clock3 size={28} /> : <TriangleAlert size={28} />}
              </span>
              <span className="eyebrow">{job ? t("video.mediaQueue") : t("video.runtime")}</span>
              <h3>{job ? translateTechnicalValue(t, job.progress.stage || job.status, "status.running") : runtimeLabel}</h3>
              <p>{job
                ? job.status === "queued"
                  ? t("video.queuedCopy")
                  : job.status === "cancelled"
                    ? t("video.cancelledCopy")
                    : job.status === "failed"
                      ? t("video.failedCopy")
                      : t("video.runningCopy")
                : video.runtime?.state === "ready"
                  ? t("video.readyCopy")
                  : video.runtime?.state === "busy"
                    ? t(video.runtime.active_media_job?.type === "photo_batch" ? "video.busyPhoto" : "video.busyVideo")
                    : t("video.connectBackend")}</p>

              {job && (
                <div className="video-progress" aria-label={`${Math.round(progress)}%`}>
                  <div><span style={{ width: `${progress}%` }} /></div>
                  <span><strong>{Math.round(progress)}%</strong><small>{job.progress.total_steps ? t("video.stepProgress", { current: job.progress.current_step ?? 0, total: job.progress.total_steps }) : translateTechnicalValue(t, job.progress.stage, "status.running")}</small></span>
                </div>
              )}

              {job && isVideoJobTerminal(job.status) && (
                <button type="button" className="button button--ghost" onClick={clearJob}>{t("video.startNew")}</button>
              )}
            </div>
          )}

          <div className="video-readiness">
            <div className="video-readiness__head"><div><span className="eyebrow">{t("video.liveChecks")}</span><h3>{t("video.readiness")}</h3></div><span className={`preview-badge preview-badge--${video.runtime?.state ?? "setup-required"}`}>{runtimeLabel}</span></div>
            <div className="readiness-list">
              <span>{video.runtime?.engine.tested ? <Check size={14} /> : <TriangleAlert size={14} />}<span><strong>{t("video.engine")}</strong><small>{video.runtime ? t(video.runtime.engine.tested ? "video.engineTested" : "video.engineIncomplete", { name: video.runtime.engine.name }) : t("video.waitServer")}</small></span></span>
              <span>{video.runtime?.model.smoke_tested ? <Check size={14} /> : <Cpu size={14} />}<span><strong>{t("video.convertedModel")}</strong><small>{video.runtime ? t(video.runtime.model.cached ? "video.modelCached" : "video.modelNotCached", { name: video.runtime.model.source }) : t("video.waitServer")}</small></span></span>
              <span><ShieldCheck size={14} /><span><strong>{t("video.licenseIsolation")}</strong><small>{capability ? `${capability.model.license} · ${translateTechnicalValue(t, capability.isolation)}` : t("video.registryRead")}</small></span></span>
            </div>
            {video.runtime?.reasons?.length ? <ul className="video-reason-list">{video.runtime.reasons.map((reason) => <li key={reason.code}>{translateRuntimeReason(t, reason.code)}</li>)}</ul> : null}
          </div>
        </section>
      </div>

      <section className="panel video-boundary-strip">
        <span className="video-boundary-icon"><ShieldCheck size={18} /></span>
        <span><strong>{t("video.oneQueue")}</strong><small>{video.runtime?.active_media_job ? t("video.activeQueueCopy") : t("video.clearQueueCopy")}</small></span>
        <button type="button" className="button button--ghost" onClick={() => void video.refresh()} disabled={video.loading}><RefreshCw className={video.loading ? "is-spinning" : ""} size={14} /> {t("video.refreshStatus")}</button>
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
  const { t } = useI18n();
  const [compare, setCompare] = useState(58);
  const [preset, setPreset] = useState<"faithful" | "soft" | "compression" | "print">("faithful");
  const [model, setModel] = useState("3B");
  const [target, setTarget] = useState("2×");
  const [preserveMetadata, setPreserveMetadata] = useState(true);
  const [locationMode, setLocationMode] = useState<"private" | "exif" | "manual">("private");
  const [manualLocation, setManualLocation] = useState("");

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow={t("nav.restore")}
        title={t("restore.title")}
        copy={t("restore.copy")}
        action={<span className="context-pill context-pill--safe"><ShieldCheck size={14} /> {t("restore.originalProtected")}</span>}
      />
      <div className="restore-grid">
        <section className="panel compare-panel">
          <div className="canvas-toolbar">
            <span><WandSparkles size={15} /> {t("restore.beforeAfter")}</span>
            <div><button type="button" aria-label={t("restore.view100")}>100%</button><button type="button" aria-label={t("restore.moreComparison")}><MoreHorizontal size={17} /></button></div>
          </div>
          <ComparisonStage src={restoreImage} value={compare} onChange={setCompare} />
          <div className="photo-meta">
            <span><ImageIcon size={15} /><span><strong>{restoreName || t("restore.defaultName")}</strong><small>{t("restore.originalPreview")}</small></span></span>
            <span className="photo-meta-values"><small>{t("restore.target")}</small><strong>{target} {t("restore.upscale")}</strong></span>
          </div>
        </section>

        <aside className="panel restore-controls" aria-label={t("restore.controlsAria")}>
          <label className="upload-tile" htmlFor="restore-upload">
            <Upload size={19} />
            <span><strong>{restoreName || t("restore.choosePhoto")}</strong><small>{t("restore.fileTypes")}</small></span>
            <input id="restore-upload" type="file" accept="image/*" onChange={onFile} />
          </label>

          <div className="field-stack">
            <span className="field-label">{t("restore.recipe")}</span>
            <div className="recipe-grid">
              {(["faithful", "soft", "compression", "print"] as const).map((item) => (
                <button type="button" key={item} className={preset === item ? "is-active" : ""} onClick={() => setPreset(item)}>{t(item === "faithful" ? "restore.faithful" : item === "soft" ? "restore.softPhoto" : item === "compression" ? "restore.compression" : "restore.printReady")}{item === "faithful" && <Check size={14} />}</button>
              ))}
            </div>
          </div>

          <div className="control-pair">
            <div className="field-stack"><span className="field-label">{t("restore.seedModel")}</span><Segmented label={t("restore.seedModel")} options={["3B", "7B"]} value={model} onChange={setModel} /></div>
            <div className="field-stack"><span className="field-label">{t("restore.targetSize")}</span><Segmented label={t("restore.targetSize")} options={["2×", "3×", "2160p"]} value={target} onChange={setTarget} /></div>
          </div>

          <div className="safety-box">
            <div className="setting-row"><span><strong>{t("restore.keepCamera")}</strong><small>{t("restore.keepCameraCopy")}</small></span><Toggle checked={preserveMetadata} onChange={setPreserveMetadata} label={t("restore.keepCamera")} /></div>
          </div>

          <div className="location-card">
            <div className="location-card__head">
              <span><MapPin size={16} /></span>
              <div><strong>{t("restore.placeInfo")}</strong><small>{t("restore.placeInfoCopy")}</small></div>
            </div>
            <Segmented
              label={t("restore.locationPrivacy")}
              options={["private", "exif", "manual"]}
              value={locationMode}
              onChange={setLocationMode}
              getLabel={(value) => t(value === "private" ? "restore.private" : value === "exif" ? "restore.exifReview" : "restore.manual")}
            />
            {locationMode === "private" && (
              <p className="location-note"><LockKeyhole size={14} /> {t("restore.privateCopy")}</p>
            )}
            {locationMode === "exif" && (
              <p className="location-note"><Info size={14} /> {t("restore.exifCopy")}</p>
            )}
            {locationMode === "manual" && (
              <label className="manual-location">
                <span>{t("restore.placeLabel")} <small>{t("common.optional")}</small></span>
                <input value={manualLocation} onChange={(event) => setManualLocation(event.target.value)} placeholder={t("restore.placePlaceholder")} />
                <small>{t("restore.placeNote")}</small>
              </label>
            )}
          </div>

          <div className="restore-summary"><span><Gauge size={16} /><span><strong>{model === "3B" ? t("restore.balancedMemory") : t("common.maximumDetail")}</strong><small>{model === "3B" ? t("restore.bestBatch") : t("restore.slower")}</small></span></span><button type="button" aria-label={t("restore.memoryDetails")}><ChevronDown size={15} /></button></div>
          <div className="button-row button-row--stack-mobile">
            <button className="button button--primary button--grow" type="button" onClick={() => notify(t("restore.noticePrepared"))}><Play size={16} /> {t("restore.preview")}</button>
            <button className="button button--icon" type="button" aria-label={t("restore.addFolder")}><FolderOpen size={18} /></button>
          </div>
        </aside>
      </div>

      <section className="batch-strip panel">
        <div><span className="batch-icon"><GalleryVerticalEnd size={19} /></span><span><strong>{t("restore.folderTitle")}</strong><small>{t("restore.folderCopy")}</small></span></div>
        <button type="button" className="button button--ghost" onClick={() => notify(t("restore.folderNotice"))}>{t("restore.buildBatch")} <ArrowRight size={15} /></button>
      </section>
    </div>
  );
}

function TimeLensPage({ restoreImage }: { restoreImage: string | null }) {
  const { t } = useI18n();
  const [compare, setCompare] = useState(52);
  const [year, setYear] = useState(1974);
  const [color, setColor] = useState(true);
  const [grain, setGrain] = useState(true);

  return (
    <div className="page-stack time-lens-page">
      <SectionHeading
        eyebrow={t("nav.timeLens")}
        title={t("timeLens.title")}
        copy={t("timeLens.copy")}
        action={<span className="context-pill context-pill--unavailable"><LockKeyhole size={14} /> {t("common.notAvailable")}</span>}
      />
      <section className="panel feature-unavailable" role="status">
        <span><Clock3 size={18} /></span>
        <div>
          <strong>{t("timeLens.designOnly")}</strong>
          <p>{t("timeLens.disabledCopy")}</p>
        </div>
      </section>
      <section className="time-hero panel time-lens-disabled" aria-disabled="true">
        <div className="time-visual">
          <ComparisonStage src={restoreImage} value={compare} onChange={setCompare} variant="archive" compact disabled />
          <div className="time-year"><small>{t("timeLens.workingEra")}</small><strong>{year}</strong><span>{t("timeLens.manual")}</span></div>
        </div>
        <div className="time-story">
          <span className="eyebrow">{t("timeLens.guided")}</span>
          <h3>{t("timeLens.keepTexture")}</h3>
          <p>{t("timeLens.story")}</p>
          <label className="year-control"><span><strong>{t("timeLens.possibleEra")}</strong><small>{year < 1980 ? t("timeLens.filmArchive") : year < 2000 ? t("timeLens.lateAnalogue") : t("timeLens.earlyDigital")}</small></span><input type="range" min="1940" max="2026" value={year} onChange={(event) => setYear(Number(event.target.value))} disabled /></label>
          <p className="truth-note"><Info size={14} /> {t("timeLens.truth")}</p>
          <div className="time-toggles">
            <div className="setting-row"><span><strong>{t("timeLens.naturalColour")}</strong><small>{t("timeLens.naturalColourCopy")}</small></span><Toggle checked={color} onChange={setColor} label={t("timeLens.naturalColour")} disabled /></div>
            <div className="setting-row"><span><strong>{t("timeLens.keepGrain")}</strong><small>{t("timeLens.keepGrainCopy")}</small></span><Toggle checked={grain} onChange={setGrain} label={t("timeLens.keepGrain")} disabled /></div>
          </div>
        </div>
      </section>
      <section className="timeline panel time-lens-disabled" aria-label={t("timeLens.workflowAria")} aria-disabled="true">
        {[t("timeLens.originalScan"), t("timeLens.repair"), t("timeLens.tone"), t("timeLens.storyExport")].map((step, index) => (
          <div className={`timeline-step ${index === 0 ? "is-active" : ""}`} key={step}>
            <span>{index === 0 ? <Check size={14} /> : index + 1}</span>
            <div><strong>{step}</strong><small>{[t("timeLens.protectedMaster"), t("timeLens.repairCopy"), t("timeLens.toneCopy"), t("timeLens.exportCopy")][index]}</small></div>
          </div>
        ))}
      </section>
    </div>
  );
}

function LibraryPage() {
  const { t } = useI18n();
  const libraryItems = useMemo(() => buildLibraryItems(t), [t]);
  const [filter, setFilter] = useState<"all" | "restored" | "created" | "favourites">("all");
  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow={t("nav.library")}
        title={t("library.title")}
        copy={t("library.copy")}
        action={<button className="button button--ghost" type="button"><Plus size={16} /> {t("library.newCollection")}</button>}
      />
      <div className="library-tools panel">
        <label className="search-field"><Search size={16} /><span className="sr-only">{t("library.searchAria")}</span><input placeholder={t("library.searchPlaceholder")} /></label>
        <div className="filter-tabs" role="group" aria-label={t("library.filterAria")}>
          {(["all", "restored", "created", "favourites"] as const).map((item) => <button type="button" key={item} className={filter === item ? "is-active" : ""} onClick={() => setFilter(item)}>{t(item === "all" ? "library.allWork" : item === "restored" ? "library.restored" : item === "created" ? "library.created" : "library.favourites")}</button>)}
        </div>
        <button type="button" className="button button--icon" aria-label={t("library.displaySettings")}><SlidersHorizontal size={17} /></button>
      </div>
      <div className="library-grid">
        {libraryItems.map((item) => (
          <article className="library-card" key={item.title}>
            <div className={`library-art library-art--${item.art}`}><MockPhoto variant={item.art} /><button type="button" aria-label={t("library.openOptions", { title: item.title })}><MoreHorizontal size={17} /></button><span>{item.tag}</span></div>
            <div className="library-card-copy"><div><h3>{item.title}</h3><p>{item.meta}</p></div><ArrowRight size={17} /></div>
          </article>
        ))}
      </div>
    </div>
  );
}

function ModelsPage({ video }: { video: VideoWorkspaceState }) {
  const { locale, t } = useI18n();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "photo" | "video" | "restore" | "edit">("all");
  const modelCards = useMemo(() => buildModelCards(t), [t]);
  const videoCards = useMemo(() => (video.registry?.capabilities ?? []).map((capability) => ({
    name: capability.label,
    role: capability.operations.map((operation) => translateTechnicalValue(t, operation)).join(" · "),
    copy: capability.availability === "ready" ? t("video.readyCopy") : t("settings.runtimeNotReady"),
    tags: [
      capability.model.license,
      `${capability.parameters.width.fixed} × ${capability.parameters.height.fixed}`,
      `${capability.parameters.fps.fixed} ${t("video.fpsUnit")}`,
    ],
    status: capability.availability === "ready" && video.runtime?.ready
      ? video.runtime.state === "busy" ? t("common.busy") : t("common.ready")
      : t("common.setupRequired"),
    tone: "blue",
    kind: "video",
  })), [t, video.registry?.capabilities, video.runtime]);
  const catalogCards = useMemo(() => [...modelCards, ...videoCards], [videoCards]);
  const filteredModels = useMemo(() => catalogCards.filter((model) => {
    const matchesQuery = `${model.name} ${model.role} ${model.tags.join(" ")}`.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === "all" || model.kind.toLowerCase() === filter || (filter === "restore" && model.tags.includes(t("common.restore"))) || (filter === "edit" && model.tags.includes(t("common.edit")));
    return matchesQuery && matchesFilter;
  }), [catalogCards, filter, query]);
  const showVideoConnection = videoCards.length === 0
    && (filter === "all" || filter === "video")
    && `${t("common.video")} ${t("modelPage.runtime")}`.toLowerCase().includes(query.trim().toLowerCase());
  const runtimeLabel = videoRuntimeLabel(video, t);

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow={t("nav.models")}
        title={t("modelPage.title")}
        copy={t("modelPage.copy")}
        action={<span className={`context-pill video-state-pill video-state-pill--${video.runtime?.state ?? "setup-required"}`}><MonitorPlay size={14} /> {t("modelPage.videoState", { state: runtimeLabel.toLocaleLowerCase(locale) })}</span>}
      />
      <div className="model-toolbar panel">
        <label className="search-field"><Search size={16} /><span className="sr-only">{t("modelPage.searchAria")}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("modelPage.searchPlaceholder")} /></label>
        <div className="filter-tabs" role="group" aria-label={t("modelPage.typeAria")}>
          {(["all", "photo", "video", "restore", "edit"] as const).map((item) => <button type="button" key={item} className={filter === item ? "is-active" : ""} onClick={() => setFilter(item)}>{t(item === "all" ? "common.all" : item === "photo" ? "common.photo" : item === "video" ? "common.video" : item === "restore" ? "modelPage.restoreFilter" : "modelPage.editFilter")}</button>)}
        </div>
      </div>
      <div className="model-grid">
        {showVideoConnection && (
          <article className="model-card model-card--blue model-card--connection">
            <div className="model-card-top"><span className="model-glyph"><span /></span><span className="model-status">{runtimeLabel}</span></div>
            <span className="eyebrow">{t("modelPage.runtime")}</span>
            <h3>{video.loading ? t("modelPage.reading") : t("modelPage.none")}</h3>
            <p>{video.error || t("modelPage.noneCopy")}</p>
            <button className="model-action" type="button" onClick={() => void video.refresh()}><RefreshCw className={video.loading ? "is-spinning" : ""} size={15} /> {t("modelPage.refresh")}</button>
          </article>
        )}
        {filteredModels.map((model) => (
          <article className={`model-card model-card--${model.tone}`} key={model.name}>
            <div className="model-card-top"><span className="model-glyph"><span /></span><span className="model-status">{model.status}</span></div>
            <span className="eyebrow">{model.role}</span>
            <h3>{model.name}</h3>
            <p>{model.copy}</p>
            <div className="model-tags">{model.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
            <button className="model-action" type="button">{model.kind === "video" ? t("modelPage.serverCapability") : t("modelPage.viewCapability")} <ArrowRight size={15} /></button>
          </article>
        ))}
      </div>
    </div>
  );
}

function ActivityPage({ video }: { video: VideoWorkspaceState }) {
  const { t } = useI18n();
  const activeJob = video.runtime?.active_media_job;
  const runtimeLabel = videoRuntimeLabel(video, t);
  return (
    <div className="page-stack">
      <SectionHeading eyebrow={t("nav.activity")} title={t("activity.title")} copy={t("activity.copy")} />
      <section className="activity-hero panel">
        <div className="activity-empty">
          <span>{video.runtime?.state === "busy" ? <LoaderCircle className="is-spinning" size={21} /> : video.runtime?.ready ? <Check size={21} /> : <TriangleAlert size={21} />}</span>
          <div>
            <h3>{activeJob ? t(activeJob.type === "photo_batch" ? "activity.photoActive" : "activity.videoActive") : video.runtime?.ready ? t("activity.queueClear") : runtimeLabel}</h3>
            <p>{activeJob ? t("activity.waitCopy") : video.error || (video.runtime?.reasons?.[0] ? translateRuntimeReason(t, video.runtime.reasons[0].code) : "") || t("activity.serverCopy")}</p>
          </div>
        </div>
        <div className="activity-metrics"><span><small>{t("activity.queue")}</small><strong>{activeJob ? t("common.occupied") : video.runtime?.ready ? t("common.clear") : t("common.unknown")}</strong></span><span><small>{t("activity.videoRuntime")}</small><strong>{runtimeLabel}</strong></span><span><small>{t("activity.isolation")}</small><strong>{video.runtime ? translateTechnicalValue(t, video.runtime.isolation) : "—"}</strong></span></div>
      </section>
      <section className="panel history-panel">
        <SectionHeading eyebrow={t("activity.history")} title={t("activity.noHistory")} copy={t("activity.noHistoryCopy")} />
      </section>
    </div>
  );
}

function SettingsPage({ video }: { video: VideoWorkspaceState }) {
  const { locale, setLocale, t } = useI18n();
  const [preserveOriginals, setPreserveOriginals] = useState(true);
  const [localOnly, setLocalOnly] = useState(true);
  const [batteryAware, setBatteryAware] = useState(true);
  const [stepPreviews, setStepPreviews] = useState(false);
  const runtimeLabel = videoRuntimeLabel(video, t);
  const runtimeDetail = video.error
    || (video.runtime?.reasons?.[0] ? translateRuntimeReason(t, video.runtime.reasons[0].code) : "")
    || (video.runtime?.state === "busy"
      ? t("settings.queueBusy")
      : video.runtime?.ready
        ? t("settings.runtimeReady")
        : t("settings.runtimeNotReady"));
  return (
    <div className="page-stack settings-page">
      <SectionHeading eyebrow={t("nav.settings")} title={t("settings.title")} copy={t("settings.copy")} />
      <div className="settings-grid">
        <section className="panel settings-section"><div className="settings-section-head"><span><ShieldCheck size={18} /></span><div><h3>{t("settings.privacy")}</h3><p>{t("settings.privacyCopy")}</p></div></div><div className="settings-list"><div className="setting-row"><span><strong>{t("settings.localServer")}</strong><small>{t("settings.localServerCopy")}</small></span><Toggle checked={localOnly} onChange={setLocalOnly} label={t("settings.localServer")} /></div><div className="setting-row"><span><strong>{t("settings.neverOverwrite")}</strong><small>{t("settings.neverOverwriteCopy")}</small></span><Toggle checked={preserveOriginals} onChange={setPreserveOriginals} label={t("settings.neverOverwrite")} /></div></div></section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Gauge size={18} /></span><div><h3>{t("settings.performance")}</h3><p>{t("settings.performanceCopy")}</p></div></div><div className="settings-list"><div className="setting-row"><span><strong>{t("settings.battery")}</strong><small>{t("settings.batteryCopy")}</small></span><Toggle checked={batteryAware} onChange={setBatteryAware} label={t("settings.battery")} /></div><div className="setting-row"><span><strong>{t("settings.stepPreviews")}</strong><small>{t("settings.stepPreviewsCopy")}</small></span><Toggle checked={stepPreviews} onChange={setStepPreviews} label={t("settings.stepPreviews")} /></div></div></section>
        <section className="panel settings-section language-section">
          <div className="settings-section-head"><span><GalleryVerticalEnd size={18} /></span><div><h3>{t("settings.language")}</h3><p>{t("settings.languageCopy")}</p></div></div>
          <label className="language-field" htmlFor="interface-language"><span>{t("settings.interfaceLanguage")}</span><select id="interface-language" value={locale} onChange={(event) => setLocale(event.target.value as typeof locale)}>{languageOptions.map((language) => <option key={language.code} value={language.code}>{language.label}</option>)}</select><small>{t("settings.englishDefault")}</small></label>
        </section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Sparkles size={18} /></span><div><h3>{t("settings.aiConnections")}</h3><p>{t("settings.aiConnectionsCopy")}</p></div><span className="preview-badge">{t("common.preview")}</span></div><div className="settings-list"><div className="setting-row"><span><strong>{t("settings.nativ")}</strong><small>{t("settings.nativCopy")}</small></span><span className="setting-value">{t("common.statusOnly")}</span></div><div className="setting-row"><span><strong>{t("settings.promptRefine")}</strong><small>{t("settings.promptRefineCopy")}</small></span><span className="setting-value">{t("common.notConnected")}</span></div></div></section>
        <section className="panel settings-section">
          <div className="settings-section-head"><span><Film size={18} /></span><div><h3>{t("settings.videoRuntime")}</h3><p>{runtimeDetail}</p></div><span className={`preview-badge preview-badge--${video.runtime?.state ?? "setup-required"}`}>{runtimeLabel}</span></div>
          <div className="settings-list">
            <div className="setting-row"><span><strong>{t("settings.engine")}</strong><small>{t("settings.engineCopy")}</small></span><span className={`setting-value ${video.runtime?.engine.tested ? "setting-value--safe" : ""}`}>{video.runtime?.engine.tested ? t("common.smokeTested") : video.runtime?.engine.configured ? t("settings.configured") : t("common.setupNeeded")}</span></div>
            <div className="setting-row"><span><strong>{t("settings.convertedModel")}</strong><small>{t("settings.convertedModelCopy")}</small></span><span className={`setting-value ${video.runtime?.model.smoke_tested ? "setting-value--safe" : ""}`}>{video.runtime?.model.smoke_tested ? t("common.ready") : video.runtime?.model.cached ? t("settings.cachedUnverified") : t("common.notReady")}</span></div>
            <div className="setting-row"><span><strong>{t("settings.jobIsolation")}</strong><small>{t("settings.jobIsolationCopy")}</small></span><span className="setting-value">{video.runtime ? translateTechnicalValue(t, video.runtime.concurrency) : t("common.waiting")}</span></div>
          </div>
          <button type="button" className="button button--ghost button--compact settings-refresh" onClick={() => void video.refresh()} disabled={video.loading}><RefreshCw className={video.loading ? "is-spinning" : ""} size={14} /> {t("settings.refreshRuntime")}</button>
        </section>
        <section className="panel settings-section settings-section--wide"><div className="settings-section-head"><span><HardDrive size={18} /></span><div><h3>{t("settings.storage")}</h3><p>{t("settings.storageCopy")}</p></div><button className="button button--ghost" type="button">{t("settings.chooseFolders")}</button></div><div className="storage-bar"><span style={{ width: "37%" }} /></div><div className="storage-legend"><span><i className="legend-dot legend-dot--models" />{t("settings.models")}</span><span><i className="legend-dot legend-dot--outputs" />{t("settings.outputs")}</span><span>{t("settings.liveTotals")}</span></div></section>
      </div>
    </div>
  );
}

export default function App() {
  const { t } = useI18n();
  const video = useVideoWorkspace();
  const navigation = useMemo(() => buildNavigation(t), [t]);
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
    setNotice(t("shell.photoLoaded"));
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
      <a className="skip-link" href="#studio-main">{t("shell.skip")}</a>
      <aside className="sidebar">
        <button type="button" className="brand" onClick={() => navigate("home")} aria-label={t("shell.home")}>
          <BrandMark />
          <span><strong>MLX</strong><small>MEDIA</small></span>
        </button>
        <nav className="primary-nav" aria-label={t("shell.studioNav")}>
          <span className="nav-label">{t("shell.workspace")}</span>
          {navigation.slice(0, 6).map((item) => {
            const Icon = item.icon;
            return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={18} /><span>{item.label}</span>{item.id === "restore" && <i>{t("common.new")}</i>}</button>;
          })}
          <span className="nav-label nav-label--utility">{t("shell.system")}</span>
          {navigation.slice(6).map((item) => {
            const Icon = item.icon;
            return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={18} /><span>{item.label}</span></button>;
          })}
        </nav>
        <div className="sidebar-footer"><span className="mini-status"><i /> {t("shell.localMedia")}</span><button type="button" aria-label={t("shell.workspaceMenu")}><MoreHorizontal size={17} /></button></div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className="mobile-brand"><BrandMark /><strong>MLX MEDIA</strong></div>
          <div className="page-identity"><span>{activeMeta.label}</span><small>{activeMeta.description}</small></div>
          <div className="topbar-actions">
            <button ref={searchRef} type="button" className="search-button" aria-label={t("shell.commandSearch")}><Search size={16} /><span>{t("shell.searchStudio")}</span><kbd><Command size={11} />K</kbd></button>
            <span className="local-pill"><span /> {t("shell.localOnly")}</span>
            <button
              type="button"
              className="icon-button"
              aria-label={moreOpen ? t("shell.closeQuick") : t("shell.openQuick")}
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

      <nav className="mobile-nav" aria-label={t("shell.mobileNav")}>
        {navigation.slice(0, 5).map((item) => {
          const Icon = item.icon;
          return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-label={item.label} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={19} /><span>{item.label}</span></button>;
        })}
        <button
          type="button"
          className={activePage === "library" || activePage === "settings" || activePage === "models" || activePage === "activity" || moreOpen ? "is-active" : ""}
          aria-label={t("shell.moreTools")}
          aria-expanded={moreOpen}
          aria-controls="studio-quick-menu"
          onClick={() => setMoreOpen((open) => !open)}
        >
          <Menu size={19} /><span>{t("shell.more")}</span>
        </button>
      </nav>
      {moreOpen && (
        <>
          <button className="quick-menu-scrim" type="button" aria-label={t("shell.closeMenu")} onClick={() => setMoreOpen(false)} />
          <aside className="quick-menu" id="studio-quick-menu" aria-label={t("shell.quickMenu")}>
            <div className="quick-menu__header"><span className="eyebrow">{t("shell.studio")}</span><strong>{t("shell.moreTools")}</strong></div>
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
