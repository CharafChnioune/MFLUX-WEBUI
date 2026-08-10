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
  Eye,
  Film,
  FolderOpen,
  GalleryVerticalEnd,
  Gauge,
  HardDrive,
  History,
  Image as ImageIcon,
  ImagePlus,
  Info,
  Layers3,
  Library,
  LockKeyhole,
  MapPin,
  Menu,
  MoreHorizontal,
  MonitorPlay,
  Music2,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
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
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  VIDEO_CAPABILITIES,
  isFrameCountValid,
  tasksForModel,
  type VideoModelFamilyId,
  type VideoTask,
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
  { id: "video", label: "Video", description: "Design motion workflows", icon: Film },
  { id: "restore", label: "Restore", description: "Faithful photo enhancement", icon: WandSparkles },
  { id: "time-lens", label: "Time Lens", description: "Bring memories forward", icon: History },
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
    title: "Design a video",
    copy: "Shape a local text, image or audio-to-video job with verified model constraints.",
    icon: Film,
    tone: "blue",
  },
  {
    id: "time-lens" as PageId,
    title: "Open Time Lens",
    copy: "Recover old photographs with a gentle, story-first workflow.",
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
  {
    name: "LTX-2 / 2.3",
    role: "Joint audio & video",
    copy: "Text, image and audio-conditioned generation with distilled and dev pipelines.",
    tags: ["T2V", "I2V", "A2V", "19B"],
    status: "Contract preview",
    tone: "cyan",
    kind: "Video",
  },
  {
    name: "Wan 2.1",
    role: "Converted local video",
    copy: "Single-model text-to-video and model-dependent image-to-video on Apple silicon.",
    tags: ["T2V", "I2V", "1.3B / 14B"],
    status: "Setup required",
    tone: "blue",
    kind: "Video",
  },
  {
    name: "Wan 2.2",
    role: "Single & dual-model video",
    copy: "Text and image workflows with local converted weights, scheduler and LoRA controls.",
    tags: ["T2V", "I2V", "5B / 14B"],
    status: "Setup required",
    tone: "violet",
    kind: "Video",
  },
];

const libraryItems = [
  { title: "Golden-hour walk", meta: "18 photographs", tag: "Travel", art: "sunset" },
  { title: "Family archive", meta: "42 restored", tag: "Time Lens", art: "archive" },
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
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`toggle ${checked ? "is-on" : ""}`}
      role="switch"
      aria-checked={checked}
      aria-label={label}
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
}: {
  src: string | null;
  value: number;
  onChange: (value: number) => void;
  variant?: string;
  compact?: boolean;
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
          <div className="hero-status"><ShieldCheck size={15} /> Local only · Ready</div>
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
            <div><span className="eyebrow">Apple silicon</span><h3>Studio ready</h3></div>
            <span className="status-dot" aria-label="System is ready" />
          </div>
          <div className="system-meter"><span style={{ width: "32%" }} /></div>
          <div className="system-stats">
            <span><small>Memory profile</small><strong>Balanced</strong></span>
            <span><small>Queue</small><strong>Clear</strong></span>
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

const videoTaskLabels: Record<VideoTask, string> = {
  "text-to-video": "Text to video",
  "image-to-video": "Image to video",
  "audio-to-video": "Audio to video",
};

const videoTaskIcons: Record<VideoTask, LucideIcon> = {
  "text-to-video": Film,
  "image-to-video": ImagePlus,
  "audio-to-video": Music2,
};

function VideoPage({ notify }: { notify: (message: string) => void }) {
  const [model, setModel] = useState<VideoModelFamilyId>("ltx-2");
  const [task, setTask] = useState<VideoTask>("text-to-video");
  const [prompt, setPrompt] = useState("A slow camera move through soft window light, natural motion, cinematic framing");
  const [frames, setFrames] = useState(33);
  const [sourceName, setSourceName] = useState("");
  const capability = VIDEO_CAPABILITIES.find((item) => item.id === model) ?? VIDEO_CAPABILITIES[0];
  const availableTasks = tasksForModel(model);
  const frameValid = isFrameCountValid(model, frames);
  const fps = 24;
  const duration = (frames / fps).toFixed(1);

  useEffect(() => {
    if (!availableTasks.includes(task)) setTask("text-to-video");
    setSourceName("");
  }, [availableTasks, model, task]);

  const onConditioningFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSourceName(file.name);
    notify(`${file.name} added to the local workflow draft.`);
  };

  return (
    <div className="page-stack video-page">
      <SectionHeading
        eyebrow="Video"
        title="Direct motion without hiding the machinery"
        copy="Build a validated video job for mlx-video. The runner stays disabled until its isolated environment, model license and exact checkpoint pass local checks."
        action={<span className="context-pill context-pill--preview"><MonitorPlay size={14} /> Integration preview</span>}
      />

      <div className="video-studio-grid">
        <section className="panel video-brief-panel" aria-label="Video workflow brief">
          <div className="video-mode-list" aria-label="Video task">
            {availableTasks.map((item) => {
              const Icon = videoTaskIcons[item];
              return (
                <button type="button" key={item} className={task === item ? "is-active" : ""} onClick={() => setTask(item)}>
                  <Icon size={16} /><span>{videoTaskLabels[item]}</span>
                </button>
              );
            })}
          </div>

          <label className="field-stack" htmlFor="video-prompt">
            <span className="field-label">Motion brief</span>
            <textarea id="video-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={5} />
          </label>

          <div className="field-stack">
            <span className="field-label">MLX video family</span>
            <div className="video-model-picker">
              {VIDEO_CAPABILITIES.map((item) => (
                <button type="button" key={item.id} className={model === item.id ? "is-active" : ""} onClick={() => setModel(item.id)}>
                  <span><strong>{item.label}</strong><small>{item.source_kind === "hugging-face" ? "Repository weights" : "Converted local folder"}</small></span>
                  {model === item.id && <Check size={15} />}
                </button>
              ))}
            </div>
          </div>

          {task !== "text-to-video" && (
            <label className="video-source-tile" htmlFor="video-conditioning-file">
              {task === "audio-to-video" ? <Music2 size={19} /> : <ImagePlus size={19} />}
              <span>
                <strong>{sourceName || (task === "audio-to-video" ? "Choose source audio" : "Choose a first frame")}</strong>
                <small>{task === "audio-to-video" ? "Local audio file · LTX only" : "Local image · never uploaded by the shell"}</small>
              </span>
              <Upload size={16} />
              <input
                id="video-conditioning-file"
                type="file"
                accept={task === "audio-to-video" ? "audio/*" : "image/*"}
                onChange={onConditioningFile}
              />
            </label>
          )}

          <div className="video-output-grid">
            <label className="field-stack">
              <span className="field-label">Frames</span>
              <input className={frameValid ? "" : "is-invalid"} type="number" min="1" step={model === "ltx-2" ? 8 : 4} value={frames} onChange={(event) => setFrames(Number(event.target.value))} />
              <small>{capability.frame_rule}</small>
            </label>
            <div className="field-stack"><span className="field-label">Output</span><span className="video-readout">{model === "ltx-2" ? "512 × 512" : "1280 × 704"}<small>{fps} fps · {duration}s draft</small></span></div>
            <div className="field-stack"><span className="field-label">Pipeline</span><span className="video-readout">{model === "ltx-2" ? "Distilled" : "UniPC"}<small>Conservative default</small></span></div>
          </div>

          {!frameValid && <p className="validation-note"><TriangleAlert size={14} /> Frame count must follow {capability.frame_rule} for this family.</p>}

          <div className="video-contract-note">
            <Info size={16} />
            <span><strong>Draft only</strong><small>This screen creates no process and downloads no model. Submission unlocks only after backend and checkpoint validation.</small></span>
          </div>

          <button className="button button--primary button--wide" type="button" disabled>
            <Film size={17} /> Video runner not connected
          </button>
        </section>

        <section className="panel video-preview-panel" aria-label="Video job preview">
          <div className="canvas-toolbar"><span><span className="status-dot status-dot--preview" /> Storyboard preview</span><span className="preview-badge">No render started</span></div>
          <div className="video-canvas">
            <MockPhoto variant="night" />
            <div className="video-safe-frame" />
            <span className="video-play"><Play size={22} /></span>
            <span className="video-timecode">00:00 / 00:{duration.padStart(4, "0")}</span>
          </div>
          <div className="video-timeline" aria-hidden="true">
            {Array.from({ length: 8 }, (_, index) => <span key={index}><MockPhoto variant={index % 3 === 0 ? "night" : "coast"} /></span>)}
          </div>

          <div className="video-readiness">
            <div className="video-readiness__head"><div><span className="eyebrow">Readiness</span><h3>Safe integration gate</h3></div><span className="preview-badge preview-badge--warning">Runner offline</span></div>
            <div className="readiness-list">
              <span><Check size={14} /><span><strong>Verified capability map</strong><small>{capability.label} · {availableTasks.map((item) => videoTaskLabels[item]).join(" · ")}</small></span></span>
              <span><Cpu size={14} /><span><strong>Separate Python environment</strong><small>Required to protect the existing photo stack.</small></span></span>
              <span><TriangleAlert size={14} /><span><strong>Model license review</strong><small>{capability.license}</small></span></span>
            </div>
            <p>{capability.caution}</p>
          </div>
        </section>
      </div>

      <section className="panel video-boundary-strip">
        <span className="video-boundary-icon"><ShieldCheck size={18} /></span>
        <span><strong>Photo jobs keep priority</strong><small>MLX Media will serialize GPU-heavy photo and video work instead of letting two large models fight over unified memory.</small></span>
        <button type="button" className="button button--ghost" onClick={() => notify("Integration notes are documented in docs/MLX_VIDEO_INTEGRATION.md.")}>Read integration notes</button>
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
        title="Meet the photograph where it began"
        copy="Restore the image first, then decide how much of its age and atmosphere should remain."
        action={<span className="context-pill"><Clock3 size={14} /> Faithful by design</span>}
      />
      <section className="time-hero panel">
        <div className="time-visual">
          <ComparisonStage src={restoreImage} value={compare} onChange={setCompare} variant="archive" compact />
          <div className="time-year"><small>Working era</small><strong>{year}</strong><span>Set manually</span></div>
        </div>
        <div className="time-story">
          <span className="eyebrow">A guided restoration</span>
          <h3>Keep the texture of the moment.</h3>
          <p>Time Lens separates repair from interpretation, so faces, places and the character of the original remain yours.</p>
          <label className="year-control"><span><strong>Possible era</strong><small>{year < 1980 ? "Film archive" : year < 2000 ? "Late analogue" : "Early digital"}</small></span><input type="range" min="1940" max="2026" value={year} onChange={(event) => setYear(Number(event.target.value))} /></label>
          <p className="truth-note"><Info size={14} /> This era is a creative input, not a verified capture date. Confirm it from your own records before saving.</p>
          <div className="time-toggles">
            <div className="setting-row"><span><strong>Natural colour recovery</strong><small>Balanced skin tones and faded dyes.</small></span><Toggle checked={color} onChange={setColor} label="Natural colour recovery" /></div>
            <div className="setting-row"><span><strong>Keep original grain</strong><small>Preserves the medium instead of polishing it away.</small></span><Toggle checked={grain} onChange={setGrain} label="Keep original grain" /></div>
          </div>
        </div>
      </section>
      <section className="timeline panel" aria-label="Time Lens workflow">
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

function ModelsPage() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const filteredModels = useMemo(() => modelCards.filter((model) => {
    const matchesQuery = `${model.name} ${model.role} ${model.tags.join(" ")}`.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === "All" || model.kind === filter || model.tags.includes(filter);
    return matchesQuery && matchesFilter;
  }), [filter, query]);

  return (
    <div className="page-stack">
      <SectionHeading
        eyebrow="Models"
        title="A catalog that explains itself"
        copy="Choose photo and video models by capability, memory, maturity and license—not by cryptic checkpoint names."
        action={<span className="context-pill"><CloudOff size={14} /> Local catalog preview</span>}
      />
      <div className="model-toolbar panel">
        <label className="search-field"><Search size={16} /><span className="sr-only">Search models</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models or capabilities" /></label>
        <div className="filter-tabs" aria-label="Model type">
          {["All", "Photo", "Video", "Restore", "Edit"].map((item) => <button type="button" key={item} className={filter === item ? "is-active" : ""} onClick={() => setFilter(item)}>{item}</button>)}
        </div>
      </div>
      <div className="model-grid">
        {filteredModels.map((model) => (
          <article className={`model-card model-card--${model.tone}`} key={model.name}>
            <div className="model-card-top"><span className="model-glyph"><span /></span><span className="model-status">{model.status}</span></div>
            <span className="eyebrow">{model.role}</span>
            <h3>{model.name}</h3>
            <p>{model.copy}</p>
            <div className="model-tags">{model.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
            <button className="model-action" type="button">{model.kind === "Video" ? "Review integration card" : "View capability card"} <ArrowRight size={15} /></button>
          </article>
        ))}
      </div>
    </div>
  );
}

function ActivityPage() {
  const jobs = [
    { title: "Golden-hour walk · restore", meta: "18 photos · SeedVR2 3B", time: "Today, 15:42", status: "Completed" },
    { title: "Travel concept", meta: "4 variations · Auto", time: "Today, 14:18", status: "Completed" },
    { title: "Archive scan", meta: "1 photo · Time Lens", time: "Yesterday", status: "Completed" },
  ];
  return (
    <div className="page-stack">
      <SectionHeading eyebrow="Activity" title="A truthful view of what your Mac is doing" copy="One clear queue for downloads, photo work, future video jobs and exports." />
      <section className="activity-hero panel"><div className="activity-empty"><span><Check size={21} /></span><div><h3>The queue is clear</h3><p>New work will appear here with real stage-by-stage progress.</p></div></div><div className="activity-metrics"><span><small>Active memory</small><strong>—</strong></span><span><small>Loaded model</small><strong>None</strong></span><span><small>Local status</small><strong>Ready</strong></span></div></section>
      <section className="panel history-panel">
        <SectionHeading eyebrow="History preview" title="Recent sessions" action={<button className="text-button" type="button">Clear filters</button>} />
        <div className="job-list">
          {jobs.map((job) => <div className="job-row" key={job.title}><span className="job-icon"><Check size={15} /></span><span><strong>{job.title}</strong><small>{job.meta}</small></span><time>{job.time}</time><span className="success-pill">{job.status}</span><button type="button" aria-label={`More options for ${job.title}`}><MoreHorizontal size={17} /></button></div>)}
        </div>
      </section>
    </div>
  );
}

function SettingsPage() {
  const [preserveOriginals, setPreserveOriginals] = useState(true);
  const [localOnly, setLocalOnly] = useState(true);
  const [batteryAware, setBatteryAware] = useState(true);
  const [stepPreviews, setStepPreviews] = useState(false);
  return (
    <div className="page-stack settings-page">
      <SectionHeading eyebrow="Settings" title="Make local AI feel predictable" copy="Defaults are conservative, private and reversible." />
      <div className="settings-grid">
        <section className="panel settings-section"><div className="settings-section-head"><span><ShieldCheck size={18} /></span><div><h3>Privacy & originals</h3><p>Control what leaves this Mac and what can be changed.</p></div></div><div className="settings-list"><div className="setting-row"><span><strong>Local-only server</strong><small>Blocks network access unless you explicitly enable sharing.</small></span><Toggle checked={localOnly} onChange={setLocalOnly} label="Local-only server" /></div><div className="setting-row"><span><strong>Never overwrite originals</strong><small>Every edit becomes a new, traceable version.</small></span><Toggle checked={preserveOriginals} onChange={setPreserveOriginals} label="Never overwrite originals" /></div></div></section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Gauge size={18} /></span><div><h3>Apple silicon performance</h3><p>Balance speed, memory and battery use.</p></div></div><div className="settings-list"><div className="setting-row"><span><strong>Battery-aware mode</strong><small>Uses gentler defaults while disconnected from power.</small></span><Toggle checked={batteryAware} onChange={setBatteryAware} label="Battery-aware mode" /></div><div className="setting-row"><span><strong>Step previews</strong><small>Show intermediate images during longer generations.</small></span><Toggle checked={stepPreviews} onChange={setStepPreviews} label="Step previews" /></div></div></section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Sparkles size={18} /></span><div><h3>AI assist connections</h3><p>Provider state is shown separately from creative capabilities.</p></div><span className="preview-badge">Preview</span></div><div className="settings-list"><div className="setting-row"><span><strong>Nativ local server</strong><small>Status and detected-model discovery only; no prompt or vision calls.</small></span><span className="setting-value">Status only</span></div><div className="setting-row"><span><strong>Local prompt refinement</strong><small>Existing Ollama or MLX path; not wired into this React preview.</small></span><span className="setting-value">Not connected</span></div></div></section>
        <section className="panel settings-section"><div className="settings-section-head"><span><Film size={18} /></span><div><h3>Video runtime</h3><p>Keep experimental video dependencies isolated from proven photo workflows.</p></div><span className="preview-badge preview-badge--warning">Offline</span></div><div className="settings-list"><div className="setting-row"><span><strong>mlx-video environment</strong><small>Separate process and dependency lock required before activation.</small></span><span className="setting-value">Not installed</span></div><div className="setting-row"><span><strong>Media job isolation</strong><small>Photo and video jobs will never compete for unified memory.</small></span><span className="setting-value setting-value--safe">Required</span></div></div></section>
        <section className="panel settings-section settings-section--wide"><div className="settings-section-head"><span><HardDrive size={18} /></span><div><h3>Storage</h3><p>Keep large local models and outputs understandable.</p></div><button className="button button--ghost" type="button">Choose folders</button></div><div className="storage-bar"><span style={{ width: "37%" }} /></div><div className="storage-legend"><span><i className="legend-dot legend-dot--models" />Models · —</span><span><i className="legend-dot legend-dot--outputs" />Outputs · —</span><span>Connect backend for live totals</span></div></section>
      </div>
    </div>
  );
}

export default function App() {
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
      case "video": return <VideoPage notify={setNotice} />;
      case "restore": return <RestorePage restoreImage={restoreImage} restoreName={restoreName} onFile={onFile} notify={setNotice} />;
      case "time-lens": return <TimeLensPage restoreImage={restoreImage} />;
      case "library": return <LibraryPage />;
      case "models": return <ModelsPage />;
      case "activity": return <ActivityPage />;
      case "settings": return <SettingsPage />;
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
          return <button type="button" key={item.id} className={activePage === item.id ? "is-active" : ""} aria-current={activePage === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={19} /><span>{item.label}</span></button>;
        })}
        <button
          type="button"
          className={activePage === "library" || activePage === "settings" || activePage === "models" || activePage === "activity" || moreOpen ? "is-active" : ""}
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
