"""
Shared data structures for the MFLUX-WEBUI async API.
"""

import time
import uuid
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobType(str, Enum):
    txt2img = "txt2img"
    img2img = "img2img"
    controlnet = "controlnet"
    upscale = "upscale"


@dataclass
class JobProgress:
    current_image: int = 0
    total_images: int = 1
    percent: float = 0.0
    stage: str = ""


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_type: JobType = JobType.txt2img
    status: JobStatus = JobStatus.queued
    params: Dict[str, Any] = field(default_factory=dict)
    progress: JobProgress = field(default_factory=JobProgress)
    result: Optional[Dict[str, Any]] = None
    error: Optional["APIError"] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    log_lines: List[Dict[str, Any]] = field(default_factory=list)
    _subscribers: List[queue.Queue] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_subscriber(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def remove_subscriber(self, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def notify(self, event_type: str, data: Dict[str, Any]):
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait((event_type, data))
            except queue.Full:
                pass

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "job_id": self.id,
            "type": self.job_type.value,
            "status": self.status.value,
            "progress": {
                "current_image": self.progress.current_image,
                "total_images": self.progress.total_images,
                "percent": self.progress.percent,
                "stage": self.progress.stage,
            },
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error.to_dict()
        return d


@dataclass
class APIError:
    code: str
    message: str
    details: Optional[str] = None
    stage: Optional[str] = None

    # Standard error codes
    INVALID_JSON = "INVALID_JSON"
    MISSING_PARAM = "MISSING_PARAM"
    INVALID_PARAM = "INVALID_PARAM"
    GENERATION_FAILED = "GENERATION_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_ALREADY_COMPLETED = "JOB_ALREADY_COMPLETED"
    CANCELLED = "CANCELLED"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            d["details"] = self.details
        if self.stage:
            d["stage"] = self.stage
        return d
