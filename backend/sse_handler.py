"""
Server-Sent Events (SSE) helpers for the async API.

Event types:
  status   - Job status changes
  progress - Generation progress updates
  log      - Captured print() lines
  result   - Final result with base64 images
  error    - Error details
"""

import json
import queue
import time


def send_sse_headers(handler):
    """Send the required headers for an SSE stream."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def send_sse_event(wfile, event_type: str, data: dict):
    """Write a single SSE event to the stream."""
    payload = json.dumps(data, default=str)
    chunk = f"event: {event_type}\ndata: {payload}\n\n"
    wfile.write(chunk.encode("utf-8"))
    wfile.flush()


def stream_job_events(handler, job):
    """
    Main SSE loop: subscribe to job events and stream them to the client.

    Sends a heartbeat comment every 15 seconds to keep the connection alive.
    Terminates when the job reaches a terminal state (completed/failed/cancelled)
    and the subscriber queue is drained.
    """
    from backend.api_models import JobStatus

    send_sse_headers(handler)
    sub_queue = job.add_subscriber()

    try:
        # Send current status immediately so the client knows the starting state
        send_sse_event(handler.wfile, "status", {
            "job_id": job.id,
            "status": job.status.value,
        })

        # If the job already has log lines, replay them
        for log_entry in list(job.log_lines):
            send_sse_event(handler.wfile, "log", log_entry)

        # If already completed, send result and close
        if job.status == JobStatus.completed and job.result is not None:
            send_sse_event(handler.wfile, "result", job.result)
            return
        if job.status == JobStatus.failed and job.error is not None:
            send_sse_event(handler.wfile, "error", job.error.to_dict())
            return
        if job.status == JobStatus.cancelled:
            send_sse_event(handler.wfile, "error", {
                "code": "CANCELLED",
                "message": "Job was cancelled",
            })
            return

        heartbeat_interval = 15
        while True:
            try:
                event_type, data = sub_queue.get(timeout=heartbeat_interval)
            except queue.Empty:
                # Send heartbeat to keep connection alive
                try:
                    handler.wfile.write(b": heartbeat\n\n")
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                continue

            try:
                send_sse_event(handler.wfile, event_type, data)
            except (BrokenPipeError, ConnectionResetError):
                return

            # Terminal events close the stream
            if event_type in ("result", "error"):
                return

    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        job.remove_subscriber(sub_queue)
