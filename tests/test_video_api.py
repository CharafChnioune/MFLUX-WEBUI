import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from backend.api_models import Job, JobStatus, JobType
from backend.api_server import APIServer
from backend.job_manager import JobManager
from backend.job_manager import get_media_generation_lock


class VideoAPITest(unittest.TestCase):
    def setUp(self):
        self.runtime = tempfile.TemporaryDirectory()
        self.outputs = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {
                "MFLUX_VIDEO_RUNTIME_ROOT": self.runtime.name,
                "MFLUX_VIDEO_OUTPUT_ROOT": self.outputs.name,
            },
        )
        self.env.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), APIServer)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.env.stop()
        self.runtime.cleanup()
        self.outputs.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=2
        )
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"Content-Type": "application/json"} if body is not None else {}
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        raw = response.read()
        data = json.loads(raw) if response.getheader("Content-Type") == "application/json" else raw
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, response_headers, data

    def test_capability_and_status_are_local_and_do_not_claim_readiness(self):
        status, _, capabilities = self.request("GET", "/api/v1/video/capabilities")
        self.assertEqual(status, 200)
        serialized = json.dumps(capabilities)
        self.assertIn("wan-2.1-t2v-1.3b", serialized)
        self.assertNotIn(str(Path.home()), serialized)

        status, _, runtime = self.request("GET", "/api/v1/video/status")
        self.assertEqual(status, 200)
        self.assertNotEqual(runtime["state"], "ready")
        self.assertFalse(runtime["engine"]["tested"])

        status, headers, _ = self.request(
            "GET",
            "/api/v1/video/status",
            headers={"Origin": "https://example.com"},
        )
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_video_submission_uses_only_a_server_validated_plan(self):
        plan = {
            "schema_version": "1",
            "capability_id": "wan-2.1-t2v-1.3b",
            "operation": "text-to-video",
        }

        class FakeManager:
            submitted = None

            def submit_job(self, job_type, params):
                self.submitted = (job_type, params)
                return Job(job_type=job_type, status=JobStatus.queued, params=params)

        manager = FakeManager()
        with (
            patch("backend.video_runner.prepare_video_request", return_value=plan),
            patch("backend.job_manager.get_job_manager", return_value=manager),
        ):
            status, _, submitted = self.request(
                "POST",
                "/api/v1/generate",
                {
                    "schema_version": "1",
                    "type": "video",
                    "capability_id": "wan-2.1-t2v-1.3b",
                    "operation": "text-to-video",
                    "prompt": "A quiet sunrise over a still lake.",
                },
            )
        self.assertEqual(status, 202)
        self.assertEqual(submitted["type"], "video")
        self.assertEqual(manager.submitted[0], JobType.video)
        self.assertEqual(manager.submitted[1]["_video_plan"], plan)

    def test_artifact_route_returns_only_resolved_local_file(self):
        artifact = Path(self.outputs.name) / "output.mp4"
        artifact.write_bytes(b"local-video")
        with patch("backend.video_runner.resolve_video_artifact", return_value=artifact):
            status, headers, payload = self.request(
                "GET", "/api/v1/video/artifacts/abcdef123456/output.mp4"
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"local-video")
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_artifact_route_supports_single_byte_ranges_for_video_playback(self):
        artifact = Path(self.outputs.name) / "output.mp4"
        artifact.write_bytes(b"local-video")
        with patch("backend.video_runner.resolve_video_artifact", return_value=artifact):
            status, headers, payload = self.request(
                "GET",
                "/api/v1/video/artifacts/abcdef123456/video.mp4",
                headers={"Range": "bytes=1-3"},
            )
            self.assertEqual(status, 206)
            self.assertEqual(payload, b"oca")
            self.assertEqual(headers["Content-Range"], "bytes 1-3/11")
            self.assertEqual(headers["Accept-Ranges"], "bytes")

            status, headers, payload = self.request(
                "GET",
                "/api/v1/video/artifacts/abcdef123456/video.mp4",
                headers={"Range": "bytes=99-100"},
            )
            self.assertEqual(status, 416)
            self.assertEqual(payload, b"")
            self.assertEqual(headers["Content-Range"], "bytes */11")


class VideoJobManagerTest(unittest.TestCase):
    def test_async_managers_share_the_process_wide_media_lock(self):
        first = JobManager()
        second = JobManager()
        self.assertIs(first._generation_lock, get_media_generation_lock())
        self.assertIs(second._generation_lock, get_media_generation_lock())

    def test_video_result_is_published(self):
        manager = JobManager()
        job = Job(
            job_type=JobType.video,
            status=JobStatus.running,
            params={"_video_plan": {"capability_id": "wan-2.1-t2v-1.3b"}},
        )
        result = {"status": "completed", "artifact_url": "/api/v1/video/artifacts/example/output.mp4"}
        with patch("backend.video_runner.run_video_job", return_value=result):
            manager._run_video(job, job.params, lambda *_args: None)
        self.assertEqual(job.status, JobStatus.completed)
        self.assertEqual(job.progress.percent, 100.0)
        self.assertEqual(job.result, result)

    def test_running_video_cancel_reaches_isolated_process_registry(self):
        manager = JobManager()
        job = Job(job_type=JobType.video, status=JobStatus.running)
        manager._jobs[job.id] = job
        with patch("backend.video_runner.request_video_cancel", return_value=True) as cancel:
            manager.cancel_job(job.id)
        cancel.assert_called_once_with(job.id)
        self.assertEqual(job.status, JobStatus.cancelled)

    def test_cancellation_wins_a_final_video_publication_race(self):
        manager = JobManager()
        job = Job(
            job_type=JobType.video,
            status=JobStatus.running,
            params={"_video_plan": {"capability_id": "wan-2.1-t2v-1.3b"}},
        )
        completed = {"status": "completed", "artifact_urls": {"video": "/video.mp4"}}
        cancelled = {
            "status": "cancelled",
            "artifact_urls": {"provenance": "/provenance.json"},
        }

        def finish_after_cancel(*_args, **_kwargs):
            job.status = JobStatus.cancelled
            return completed

        with patch(
            "backend.video_runner.run_video_job", side_effect=finish_after_cancel
        ), patch(
            "backend.video_runner.discard_video_job_output", return_value=cancelled
        ) as discard:
            manager._run_video(job, job.params, lambda *_args: None)

        self.assertEqual(job.status, JobStatus.cancelled)
        self.assertEqual(job.result, cancelled)
        discard.assert_called_once_with(job.id)

    def test_cancelled_queued_job_cannot_transition_back_to_running(self):
        manager = JobManager()
        job = Job(
            job_type=JobType.video,
            status=JobStatus.queued,
            params={"_video_plan": {"capability_id": "wan-2.1-t2v-1.3b"}},
        )
        manager._jobs[job.id] = job

        with patch("backend.video_runner.request_video_cancel", return_value=False), patch.object(
            manager, "_dispatch"
        ) as dispatch:
            manager.cancel_job(job.id)
            manager._execute_job(job)

        self.assertEqual(job.status, JobStatus.cancelled)
        self.assertIsNone(job.started_at)
        dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
