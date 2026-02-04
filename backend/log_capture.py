"""
Terminal output interception for job log capture.

TeeStream writes to both the original stdout and a callback so that
print() statements inside flux_manager.py are captured per-job while
the terminal still receives all output.

Safe because the job worker executes one job at a time.
"""

import re
import sys
import time
from typing import Callable, Optional


_LEVEL_PATTERNS = [
    (re.compile(r"\berror\b", re.IGNORECASE), "error"),
    (re.compile(r"\bwarning\b", re.IGNORECASE), "warning"),
    (re.compile(r"\bmemory\b", re.IGNORECASE), "memory"),
    (re.compile(r"\bprogress\b|\bstep\b|\bgenerating\b", re.IGNORECASE), "progress"),
]


def classify_log_level(message: str) -> str:
    for pattern, level in _LEVEL_PATTERNS:
        if pattern.search(message):
            return level
    return "info"


class TeeStream:
    """File-like object that writes to the original stream AND a callback."""

    def __init__(self, original, callback: Callable[[str], None]):
        self._original = original
        self._callback = callback

    def write(self, data: str):
        if data and data.strip():
            try:
                self._callback(data)
            except Exception:
                pass
        return self._original.write(data)

    def flush(self):
        return self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    # Forward any other attribute access to the original stream
    def __getattr__(self, name):
        return getattr(self._original, name)


class JobLogCapture:
    """
    Context manager that temporarily tees sys.stdout during job execution.

    Usage::

        def on_line(line_dict):
            job.log_lines.append(line_dict)
            job.notify("log", line_dict)

        with JobLogCapture(on_line):
            generate_image_gradio(...)
    """

    def __init__(self, line_callback: Callable[[dict], None]):
        self._line_callback = line_callback
        self._original_stdout: Optional[object] = None
        self._original_stderr: Optional[object] = None

    def _handle_output(self, text: str):
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            entry = {
                "timestamp": time.time(),
                "level": classify_log_level(line),
                "message": line,
            }
            self._line_callback(entry)

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = TeeStream(self._original_stdout, self._handle_output)
        sys.stderr = TeeStream(self._original_stderr, self._handle_output)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        return False
