"""
Job lifecycle and queue management for the async API.

Provides a singleton JobManager that:
- Accepts job submissions and returns immediately with a job ID
- Runs a single worker thread (GPU is serialized via MLX memory)
- Dispatches to existing generation functions with progress callbacks
- Manages SSE subscribers per job
- Cleans up completed jobs after a TTL
"""

import base64
import collections
import threading
import time
import traceback
from io import BytesIO
from typing import Dict, List, Optional

from backend.api_models import APIError, Job, JobProgress, JobStatus, JobType
from backend.log_capture import JobLogCapture


_JOB_TTL = 3600  # 1 hour
_MAX_COMPLETED = 100
_CLEANUP_INTERVAL = 300  # 5 minutes


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._queue: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._queue_event = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="job-worker", daemon=True
        )
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="job-cleanup", daemon=True
        )
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._worker_thread.start()
        self._cleanup_thread.start()

    # ── Job submission ──────────────────────────────────────────────

    def submit_job(self, job_type: JobType, params: dict) -> Job:
        job = Job(job_type=job_type, params=params)
        job.progress.total_images = int(params.get("num_images", 1))
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
        self._queue_event.set()
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> Optional[Job]:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
            return job
        job.status = JobStatus.cancelled
        job.completed_at = time.time()
        job.notify("status", {"job_id": job.id, "status": job.status.value})
        job.notify("error", {"code": "CANCELLED", "message": "Job was cancelled"})
        # Signal workflow to stop
        try:
            from backend.generation_workflow import get_generation_workflow
            workflow = get_generation_workflow()
            workflow.request_cancel(job_id)
        except Exception:
            pass
        return job

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)

    # ── Worker loop ─────────────────────────────────────────────────

    def _worker_loop(self):
        while True:
            self._queue_event.wait()
            while True:
                job_id = None
                with self._lock:
                    if self._queue:
                        job_id = self._queue.popleft()
                    else:
                        self._queue_event.clear()
                        break
                if job_id is None:
                    break
                job = self.get_job(job_id)
                if job is None or job.status == JobStatus.cancelled:
                    continue
                self._execute_job(job)

    def _execute_job(self, job: Job):
        job.status = JobStatus.running
        job.started_at = time.time()
        job.notify("status", {"job_id": job.id, "status": job.status.value})

        def on_log_line(entry: dict):
            job.log_lines.append(entry)
            job.notify("log", entry)

        def progress_callback(event: str, data=None):
            if event == "stage":
                job.progress.stage = data or ""
                job.notify("progress", {
                    "current_image": job.progress.current_image,
                    "total_images": job.progress.total_images,
                    "percent": job.progress.percent,
                    "stage": job.progress.stage,
                })
            elif event == "image_start":
                if isinstance(data, dict):
                    job.progress.current_image = data.get("current_image", 0)
                    job.progress.total_images = data.get("total_images", job.progress.total_images)
                n = job.progress.total_images or 1
                job.progress.percent = round(
                    (job.progress.current_image - 1) / n * 100, 1
                )
                job.progress.stage = "generating"
                job.notify("progress", {
                    "current_image": job.progress.current_image,
                    "total_images": job.progress.total_images,
                    "percent": job.progress.percent,
                    "stage": job.progress.stage,
                })
            elif event == "image_complete":
                if isinstance(data, dict):
                    job.progress.current_image = data.get("current_image", job.progress.current_image)
                n = job.progress.total_images or 1
                job.progress.percent = round(
                    job.progress.current_image / n * 100, 1
                )
                job.progress.stage = "saving"
                job.notify("progress", {
                    "current_image": job.progress.current_image,
                    "total_images": job.progress.total_images,
                    "percent": job.progress.percent,
                    "stage": job.progress.stage,
                })
            elif event == "image_error":
                msg = data.get("error", "unknown") if isinstance(data, dict) else str(data)
                job.notify("log", {
                    "timestamp": time.time(),
                    "level": "error",
                    "message": f"Image error: {msg}",
                })

        with self._generation_lock:
            try:
                with JobLogCapture(on_log_line):
                    if job.status == JobStatus.cancelled:
                        return
                    self._dispatch(job, progress_callback)
            except Exception as exc:
                tb = traceback.format_exc()
                error_code = APIError.GENERATION_FAILED
                if "memory" in str(exc).lower():
                    error_code = APIError.OUT_OF_MEMORY
                elif "model" in str(exc).lower() and "load" in str(exc).lower():
                    error_code = APIError.MODEL_LOAD_FAILED
                job.status = JobStatus.failed
                job.completed_at = time.time()
                job.error = APIError(
                    code=error_code,
                    message=str(exc),
                    details=tb,
                    stage=job.progress.stage,
                )
                job.notify("status", {"job_id": job.id, "status": job.status.value})
                job.notify("error", job.error.to_dict())

    def _dispatch(self, job: Job, progress_callback):
        params = job.params
        job_type = job.job_type

        if job_type == JobType.txt2img:
            self._run_txt2img(job, params, progress_callback)
        elif job_type == JobType.img2img:
            self._run_img2img(job, params, progress_callback)
        elif job_type == JobType.controlnet:
            self._run_controlnet(job, params, progress_callback)
        elif job_type == JobType.upscale:
            self._run_upscale(job, params, progress_callback)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    def _run_txt2img(self, job: Job, params: dict, progress_callback):
        from backend.flux_manager import generate_image_gradio
        from backend.api_server import _resolve_model_from_payload

        model = _resolve_model_from_payload(params)
        prompt = params.get("prompt", "")
        seed = params.get("seed")
        width = int(params.get("width", 576))
        height = int(params.get("height", 1024))
        steps = str(params.get("steps", ""))
        guidance = float(params.get("guidance", 3.5))
        num_images = int(params.get("num_images", 1))
        auto_seeds = bool(params.get("auto_seeds", False))
        lora_files = params.get("lora_files") or None
        low_ram = bool(params.get("low_ram", False))

        progress_callback("stage", "loading_model")

        images, info, used_prompt = generate_image_gradio(
            prompt, model, None, seed, width, height, steps, guidance,
            lora_files, False, None, None, None, None, None, False, None,
            num_images=num_images, low_ram=low_ram, auto_seeds=auto_seeds,
            progress_callback=progress_callback,
        )

        if job.status == JobStatus.cancelled:
            return

        self._finalize_job(job, images, info, used_prompt)

    def _run_img2img(self, job: Job, params: dict, progress_callback):
        from backend.flux_manager import generate_image_i2i_gradio
        from backend.api_server import _resolve_model_from_payload, _decode_base64_image

        model = _resolve_model_from_payload(params)
        prompt = params.get("prompt", "")
        init_images = params.get("init_images") or params.get("images") or []
        if not init_images:
            raise ValueError("init_images is required for img2img")
        init_img = _decode_base64_image(init_images[0])

        seed = params.get("seed")
        width = int(params.get("width", init_img.width))
        height = int(params.get("height", init_img.height))
        steps = str(params.get("steps", ""))
        guidance = float(params.get("guidance", 3.5))
        num_images = int(params.get("num_images", 1))
        image_strength = float(params.get("image_strength", 0.4))
        lora_files = params.get("lora_files") or None
        low_ram = bool(params.get("low_ram", False))

        progress_callback("stage", "loading_model")

        images, info, used_prompt = generate_image_i2i_gradio(
            prompt, init_img, model, None, seed, height, width, steps, guidance,
            image_strength, lora_files, False, None, None, None, False, None,
            num_images=num_images, low_ram=low_ram,
            progress_callback=progress_callback,
        )

        if job.status == JobStatus.cancelled:
            return

        self._finalize_job(job, images, info, used_prompt)

    def _run_controlnet(self, job: Job, params: dict, progress_callback):
        from backend.flux_manager import generate_image_controlnet_gradio
        from backend.api_server import _resolve_model_from_payload, _decode_base64_image

        model = _resolve_model_from_payload(params)
        prompt = params.get("prompt", "")
        cn_images = (
            params.get("controlnet_image")
            or params.get("controlnet_images")
            or params.get("init_images")
            or []
        )
        if not cn_images:
            raise ValueError("controlnet_image is required")
        controlnet_img = _decode_base64_image(cn_images[0])

        seed = params.get("seed")
        width = int(params.get("width", controlnet_img.width))
        height = int(params.get("height", controlnet_img.height))
        steps = str(params.get("steps", ""))
        guidance = float(params.get("guidance", 3.5))
        controlnet_strength = float(params.get("controlnet_strength", 0.4))
        lora_files = params.get("lora_files") or None
        low_ram = bool(params.get("low_ram", False))

        progress_callback("stage", "loading_model")

        images, info, used_prompt = generate_image_controlnet_gradio(
            prompt, controlnet_img, model, None, seed, height, width, steps,
            guidance, controlnet_strength, lora_files, False, False,
            None, None, None, False, "horizontal",
            num_images=1, low_ram=low_ram,
            progress_callback=progress_callback,
        )

        if job.status == JobStatus.cancelled:
            return

        self._finalize_job(job, images, info, used_prompt)

    def _run_upscale(self, job: Job, params: dict, progress_callback):
        from backend.api_server import _decode_base64_image
        from backend import upscale_manager

        image_b64 = params.get("image")
        if not image_b64:
            raise ValueError("image is required for upscale")
        img = _decode_base64_image(image_b64)

        factor = params.get("upscale_factor", 2)
        output_format = params.get("output_format", "PNG").upper()
        metadata = bool(params.get("metadata", False))

        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as f:
            img.save(f, format="PNG")

        progress_callback("stage", "upscaling")

        upscaled, status = upscale_manager.upscale_image_gradio(
            input_image=path,
            upscale_factor=factor,
            output_format=output_format,
            metadata=metadata,
        )

        if upscaled is None:
            raise RuntimeError(status or "Upscale failed")

        self._finalize_job(job, [upscaled], status or "", "")

    def _finalize_job(self, job: Job, images, info, used_prompt):
        encoded_images = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            encoded_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        job.progress.percent = 100.0
        job.progress.stage = "completed"
        job.status = JobStatus.completed
        job.completed_at = time.time()
        job.result = {
            "images": encoded_images,
            "info": info or "",
            "prompt": used_prompt if used_prompt else "",
        }
        job.notify("status", {"job_id": job.id, "status": job.status.value})
        job.notify("result", job.result)

    # ── Cleanup ─────────────────────────────────────────────────────

    def _cleanup_loop(self):
        while True:
            time.sleep(_CLEANUP_INTERVAL)
            self._cleanup_old_jobs()

    def _cleanup_old_jobs(self):
        now = time.time()
        with self._lock:
            terminal = [
                jid
                for jid, j in self._jobs.items()
                if j.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled)
                and j.completed_at
                and (now - j.completed_at) > _JOB_TTL
            ]
            for jid in terminal:
                del self._jobs[jid]

            # Enforce max completed
            completed = sorted(
                [
                    (jid, j)
                    for jid, j in self._jobs.items()
                    if j.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled)
                ],
                key=lambda x: x[1].completed_at or 0,
            )
            while len(completed) > _MAX_COMPLETED:
                jid, _ = completed.pop(0)
                del self._jobs[jid]

    # ── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            jobs = list(self._jobs.values())
        total = len(jobs)
        by_status = {}
        for j in jobs:
            by_status[j.status.value] = by_status.get(j.status.value, 0) + 1
        completed_jobs = [j for j in jobs if j.status == JobStatus.completed and j.started_at and j.completed_at]
        avg_duration = 0.0
        if completed_jobs:
            avg_duration = sum(j.completed_at - j.started_at for j in completed_jobs) / len(completed_jobs)
        return {
            "total_jobs": total,
            "by_status": by_status,
            "queue_depth": len(self._queue),
            "avg_duration": round(avg_duration, 2),
        }


# ── Global singleton ────────────────────────────────────────────────

_instance: Optional[JobManager] = None
_instance_lock = threading.Lock()


def get_job_manager() -> JobManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = JobManager()
                _instance.start()
    return _instance
