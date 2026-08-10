import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.video_runner import (
    VIDEO_CAPABILITY_ID,
    VIDEO_ENGINE_LICENSE,
    VIDEO_ENGINE_REVISION,
    VIDEO_ENGINE_SOURCE,
    VIDEO_MODEL_LICENSE,
    VIDEO_MODEL_REVISION,
    VIDEO_MODEL_SOURCE,
    VideoValidationError,
    get_video_capabilities,
    get_video_runtime_status,
    prepare_video_request,
    public_video_plan,
    record_video_smoke_success,
    request_video_cancel,
    resolve_video_artifact,
    run_video_job,
)


def _request(**overrides):
    request = {
        "schema_version": 1,
        "type": "video",
        "operation": "text-to-video",
        "capability_id": VIDEO_CAPABILITY_ID,
        "prompt": "A quiet Andalusian courtyard at sunrise",
        "output": {
            "width": 832,
            "height": 480,
            "num_frames": 5,
            "fps": 16,
            "container": "mp4",
        },
        "sampling": {
            "steps": 2,
            "scheduler": "unipc",
            "tiling": "auto",
            "seed": 7,
        },
    }
    request.update(overrides)
    return request


class _FakeProcess:
    def __init__(self, lines, *, return_code=0, pid=999999):
        self.stdout = io.StringIO("".join(lines))
        self.returncode = return_code
        self.pid = pid
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class VideoRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.python = self.root / "video-env" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text("#!/bin/sh\n", encoding="utf-8")
        self.python.chmod(0o700)
        self.model = self.root / "hf" / "mlx-media-video" / "wan-1.3b"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "dim": 1536,
                    "num_layers": 30,
                    "model_type": "t2v",
                    "sample_fps": 16,
                }
            ),
            encoding="utf-8",
        )
        for filename in ("model.safetensors", "t5_encoder.safetensors", "vae.safetensors"):
            (self.model / filename).write_bytes(b"weights")
        manifest_files = {}
        for filename in ("config.json", "model.safetensors", "t5_encoder.safetensors", "vae.safetensors"):
            artifact = self.model / filename
            manifest_files[filename] = {
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        (self.model / "conversion-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "mlx-video-conversion-manifest",
                    "engine_revision": VIDEO_ENGINE_REVISION,
                    "source": {
                        "repository": VIDEO_MODEL_SOURCE,
                        "revision": VIDEO_MODEL_REVISION,
                    },
                    "conversion": {"model_version": "2.1", "dtype": "bfloat16"},
                    "files": manifest_files,
                }
            ),
            encoding="utf-8",
        )
        self.smoke = self.model / "smoke-provenance.json"
        self.output = self.root / "app-output"
        self.hf_home = self.root / "hf"
        self.runtime_root = self.root / "video-runtime"
        (self.runtime_root / "google" / "umt5-xxl").mkdir(parents=True)
        self.smoke_artifact = self.runtime_root / "smoke" / "proof.mp4"
        self.smoke_artifact.parent.mkdir()
        self.smoke_artifact.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"s" * 64)
        self.environment = {
            "MFLUX_VIDEO_RUNNER_PYTHON": str(self.python),
            "MFLUX_VIDEO_MODEL_DIR": str(self.model),
            "MFLUX_VIDEO_SMOKE_PROVENANCE": str(self.smoke),
            "MFLUX_VIDEO_OUTPUT_ROOT": str(self.output),
            "MFLUX_VIDEO_RUNTIME_ROOT": str(self.runtime_root),
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_home / "hub"),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write_smoke(self, **changes):
        import backend.video_runner as runner

        proof = {
            "schema_version": 1,
            "purpose": "smoke-test",
            "status": "completed",
            "capability_id": VIDEO_CAPABILITY_ID,
            "engine": {
                "source": VIDEO_ENGINE_SOURCE,
                "revision": VIDEO_ENGINE_REVISION,
                "license": VIDEO_ENGINE_LICENSE,
            },
            "model": {
                "source": VIDEO_MODEL_SOURCE,
                "revision": VIDEO_MODEL_REVISION,
                "license": VIDEO_MODEL_LICENSE,
            },
            "profile": {
                "width": 832,
                "height": 480,
                "num_frames": 5,
                "fps": 16,
                "steps": 10,
                "scheduler": "unipc",
                "seed": 42,
            },
            "generation_seconds": 36.7,
            "attestation": runner._current_runtime_attestation(
                self.python,
                self.model,
                self.runtime_root,
            ),
            "output": {
                "artifact": self.smoke_artifact.relative_to(self.runtime_root).as_posix(),
                "sha256": hashlib.sha256(self.smoke_artifact.read_bytes()).hexdigest(),
                "size_bytes": self.smoke_artifact.stat().st_size,
                "verification": {
                    "container": "mp4",
                    "width": 832,
                    "height": 480,
                    "fps": 16,
                    "num_frames": 5,
                },
            },
        }
        proof.update(changes)
        self.smoke.write_text(json.dumps(proof), encoding="utf-8")

    def _ready_context(self):
        self._write_smoke()
        return patch.multiple(
            "backend.video_runner",
            _is_apple_silicon=lambda: True,
            _validate_engine_checkout=lambda _python: (True, ""),
            _validate_tokenizer=lambda _root: (True, ""),
        )

    def _runtime_context(self):
        return patch.multiple(
            "backend.video_runner",
            _is_apple_silicon=lambda: True,
            _validate_engine_checkout=lambda _python: (True, ""),
            _validate_tokenizer=lambda _root: (True, ""),
        )

    def test_status_stays_setup_required_until_exact_smoke_proof_exists(self):
        with patch.dict(os.environ, self.environment, clear=False), self._runtime_context():
            status = get_video_runtime_status()
            self.assertFalse(status["ready"])
            self.assertEqual(status["state"], "setup-required")
            self.assertIn("smoke-provenance-missing", {item["code"] for item in status["reasons"]})

            self._write_smoke(
                engine={
                    "source": VIDEO_ENGINE_SOURCE,
                    "revision": "wrong",
                    "license": VIDEO_ENGINE_LICENSE,
                }
            )
            status = get_video_runtime_status()
            self.assertFalse(status["ready"])
            self.assertIn("smoke-provenance-invalid", {item["code"] for item in status["reasons"]})

            self._write_smoke(
                model={
                    "source": VIDEO_MODEL_SOURCE,
                    "revision": "wrong",
                    "license": VIDEO_MODEL_LICENSE,
                }
            )
            status = get_video_runtime_status()
            self.assertFalse(status["ready"])
            self.assertIn("smoke-provenance-invalid", {item["code"] for item in status["reasons"]})

            self._write_smoke()
            status = get_video_runtime_status({"id": "photo-job", "type": "photo_batch"})
            self.assertTrue(status["ready"])
            self.assertEqual(status["state"], "busy")
            self.assertTrue(status["engine"]["tested"])

    def test_capability_and_status_never_publish_absolute_paths(self):
        self._write_smoke()
        with patch.dict(os.environ, self.environment, clear=False), self._runtime_context():
            payloads = [get_video_capabilities(), get_video_runtime_status()]
        serialized = json.dumps(payloads)
        self.assertNotIn(str(self.root), serialized)
        capability = payloads[0]["capabilities"][0]
        self.assertEqual(capability["id"], VIDEO_CAPABILITY_ID)
        self.assertEqual(capability["availability"], "ready")
        self.assertEqual(capability["parameters"]["num_frames"]["rule"], "4n+1")

    def test_request_validation_is_strict_and_plan_has_public_projection(self):
        self._write_smoke()
        with patch.dict(os.environ, self.environment, clear=False), self._runtime_context():
            plan = prepare_video_request(_request())
            public = public_video_plan(plan)
            self.assertEqual(public["sampling"]["scheduler"], "unipc")
            self.assertNotIn("_server", public)
            self.assertNotIn(str(self.root), json.dumps(public))

            invalid_requests = []
            bad_frames = _request()
            bad_frames["output"] = {**bad_frames["output"], "num_frames": 6}
            invalid_requests.append(bad_frames)
            bad_size = _request()
            bad_size["output"] = {**bad_size["output"], "width": 1024}
            invalid_requests.append(bad_size)
            bad_scheduler = _request()
            bad_scheduler["sampling"] = {**bad_scheduler["sampling"], "scheduler": "euler"}
            invalid_requests.append(bad_scheduler)
            extra_path = _request(model_dir="/tmp/untrusted")
            invalid_requests.append(extra_path)
            empty_prompt = _request(prompt="  ")
            invalid_requests.append(empty_prompt)
            too_many_steps = _request()
            too_many_steps["sampling"] = {**too_many_steps["sampling"], "steps": 51}
            invalid_requests.append(too_many_steps)
            for invalid in invalid_requests:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(VideoValidationError):
                        prepare_video_request(invalid)

    def test_subprocess_uses_pinned_module_offline_cache_and_writes_provenance(self):
        self._write_smoke()
        calls = []

        def launch(argv, **kwargs):
            calls.append((argv, kwargs))
            output_path = Path(argv[argv.index("--output-path") + 1])
            output_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 64)
            return _FakeProcess(
                [
                    "Loading T5 encoder...\n",
                    "Denoising (2 steps)...\n",
                    "Diffusion: 100%|########| 2/2\n",
                    "Decoding with VAE...\n",
                    f"Video saved to {output_path}\n",
                ]
            )

        verification = {
            "container": "mp4",
            "container_verified": True,
            "inspector": "ffprobe",
            "codec": "h264",
            "width": 832,
            "height": 480,
            "fps": 16,
            "num_frames": 5,
            "duration_seconds": 0.3125,
        }
        events = []
        with patch.dict(os.environ, self.environment, clear=False), self._runtime_context(), patch("backend.video_runner.subprocess.Popen", side_effect=launch), patch(
            "backend.video_runner._inspect_mp4", return_value=verification
        ):
            plan = prepare_video_request(_request())
            result = run_video_job(
                plan,
                job_id="video-job-1",
                progress_callback=lambda event, data=None: events.append((event, data)),
            )

        self.assertEqual(result["status"], "completed")
        argv, kwargs = calls[0]
        self.assertEqual(
            argv[:5],
            [str(self.python), "-I", "-u", "-m", "mlx_video.models.wan_2.generate"],
        )
        self.assertIn("--model-dir", argv)
        self.assertIn(f"--prompt={_request()['prompt']}", argv)
        self.assertNotIn("--prompt", argv)
        self.assertEqual(argv[argv.index("--scheduler") + 1], "unipc")
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(kwargs["env"]["HF_HOME"], str(self.hf_home.resolve()))
        self.assertNotIn("HF_TOKEN", kwargs["env"])
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertTrue(kwargs["start_new_session"])
        stages = [data["stage"] for event, data in events if event == "video_progress"]
        self.assertIn("loading", stages)
        self.assertIn("denoising", stages)
        self.assertIn("decoding", stages)
        self.assertEqual(stages[-1], "completed")

        job_dir = self.output / "video-jobs" / "video-job-1"
        self.assertTrue((job_dir / "video.mp4").is_file())
        provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["status"], "completed")
        self.assertEqual(provenance["engine"]["revision"], VIDEO_ENGINE_REVISION)
        self.assertEqual(provenance["model"]["revision"], VIDEO_MODEL_REVISION)
        self.assertNotIn(str(self.root), json.dumps(provenance))
        self.assertNotIn("_server", request)

    def test_upstream_tail_is_reencoded_to_the_exact_requested_frame_count(self):
        import backend.video_runner as runner

        raw_path = self.runtime_root / "raw.mp4"
        normalized_path = self.runtime_root / "normalized.mp4"
        raw_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 64)
        raw_verification = {
            "container": "mp4",
            "container_verified": True,
            "inspector": "isolated-opencv",
            "width": 832,
            "height": 480,
            "fps": 16,
            "num_frames": 8,
            "duration_seconds": 0.5,
            "first_frame_stddev": 12.0,
        }
        final_verification = {
            **raw_verification,
            "num_frames": 5,
            "duration_seconds": 0.3125,
        }
        plan = {
            "output": _request()["output"],
            "_server": {
                "runtime_python": str(self.python.resolve()),
                "runtime_root": str(self.runtime_root.resolve()),
                "hf_home": str(self.hf_home.resolve()),
                "hf_hub_cache": str((self.hf_home / "hub").resolve()),
            },
        }

        def transcode(argv, **kwargs):
            self.assertIn("select='lt(n,5)',setpts=N/(16*TB)", argv)
            self.assertEqual(argv[argv.index("-frames:v") + 1], "5")
            self.assertEqual(argv[argv.index("-c:v") + 1], "libx264")
            self.assertEqual(argv[argv.index("-pix_fmt") + 1], "yuv420p")
            self.assertEqual(kwargs["cwd"], str(self.runtime_root.resolve()))
            normalized_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"1" * 64)

        with patch(
            "backend.video_runner._inspect_mp4",
            side_effect=[raw_verification, final_verification],
        ), patch("backend.video_runner._isolated_ffmpeg", return_value="/isolated/ffmpeg"), patch(
            "backend.video_runner.subprocess.run", side_effect=transcode
        ):
            verified = runner._normalize_and_verify_mp4(raw_path, normalized_path, plan)

        self.assertEqual(verified["raw_num_frames"], 8)
        self.assertEqual(verified["num_frames"], 5)
        self.assertEqual(verified["raw_frame_count"], 8)
        self.assertEqual(verified["final_frame_count"], 5)
        self.assertEqual(verified["tail_frames_removed"], 3)
        self.assertTrue(normalized_path.is_file())

    def test_record_smoke_success_atomically_unlocks_only_the_exact_pins(self):
        artifact = self.runtime_root / "smoke" / "output.mp4"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"2" * 64)
        verification = {
            "container": "mp4",
            "container_verified": True,
            "inspector": "isolated-opencv",
            "width": 832,
            "height": 480,
            "fps": 16,
            "num_frames": 5,
            "duration_seconds": 0.3125,
            "first_frame_stddev": 9.0,
        }
        with patch.dict(os.environ, self.environment, clear=False), patch(
            "backend.video_runner._inspect_mp4", return_value=verification
        ), self._runtime_context():
            proof = record_video_smoke_success(
                artifact,
                generation_seconds=36.7,
                raw_num_frames=8,
            )
            status = get_video_runtime_status()

        self.assertEqual(proof["engine"]["revision"], VIDEO_ENGINE_REVISION)
        self.assertEqual(proof["model"]["revision"], VIDEO_MODEL_REVISION)
        self.assertEqual(proof["output"]["verification"]["raw_num_frames"], 8)
        self.assertEqual(proof["output"]["verification"]["raw_frame_count"], 8)
        self.assertEqual(proof["output"]["verification"]["final_frame_count"], 5)
        self.assertEqual(proof["output"]["verification"]["tail_frames_removed"], 3)
        self.assertTrue(status["ready"])
        written = json.loads(self.smoke.read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "completed")

    def test_cancel_terminates_process_and_cleans_partial_output(self):
        self._write_smoke()
        process_holder = []

        def launch(argv, **_kwargs):
            output_path = Path(argv[argv.index("--output-path") + 1])
            output_path.write_bytes(b"partial")
            process = _FakeProcess(["Loading T5 encoder...\n"], return_code=None)
            process_holder.append(process)
            return process

        cancel_calls = 0

        def cancel_check():
            nonlocal cancel_calls
            cancel_calls += 1
            return cancel_calls >= 2

        with patch.dict(os.environ, self.environment, clear=False), self._runtime_context(), patch(
            "backend.video_runner.subprocess.Popen", side_effect=launch
        ):
            plan = prepare_video_request(_request())
            result = run_video_job(plan, job_id="cancel-job", cancel_check=cancel_check)

        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(process_holder[0].terminated)
        job_dir = self.output / "video-jobs" / "cancel-job"
        self.assertFalse((job_dir / "video.mp4").exists())
        self.assertFalse((job_dir / ".work").exists())
        provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["status"], "cancelled")

    def test_cancel_registry_and_artifact_resolution_are_contained(self):
        process = _FakeProcess([], return_code=None)
        import backend.video_runner as runner

        running = runner._RunningProcess(process=process, cancelled=runner.threading.Event())
        with runner._RUNNING_PROCESSES_LOCK:
            runner._RUNNING_PROCESSES["active-job"] = running
        try:
            self.assertTrue(request_video_cancel("active-job"))
            self.assertTrue(running.cancelled.is_set())
            self.assertTrue(process.terminated)
        finally:
            with runner._RUNNING_PROCESSES_LOCK:
                runner._RUNNING_PROCESSES.pop("active-job", None)

        artifact_dir = self.output / "video-jobs" / "safe-job"
        artifact_dir.mkdir(parents=True)
        video = artifact_dir / "video.mp4"
        video.write_bytes(b"video")
        with patch.dict(os.environ, self.environment, clear=False):
            self.assertEqual(resolve_video_artifact("safe-job", "video.mp4"), video.resolve())
            for unsafe_job, unsafe_file in (
                ("../escape", "video.mp4"),
                ("safe-job", "../video.mp4"),
                ("safe-job", "secret.txt"),
            ):
                with self.assertRaises(VideoValidationError):
                    resolve_video_artifact(unsafe_job, unsafe_file)

    def test_tampered_prepared_plan_is_rejected(self):
        self._write_smoke()
        with patch.dict(os.environ, self.environment, clear=False), self._runtime_context():
            plan = prepare_video_request(_request())
            tampered = copy.deepcopy(plan)
            tampered["sampling"]["steps"] = 1
            with self.assertRaises(VideoValidationError):
                public_video_plan(tampered)

            changed_environment = {
                **self.environment,
                "MFLUX_VIDEO_OUTPUT_ROOT": str(self.root / "changed-output"),
            }
            with patch.dict(os.environ, changed_environment, clear=False):
                with self.assertRaisesRegex(VideoValidationError, "changed while this job was queued"):
                    run_video_job(plan, job_id="stale-runtime-job")


if __name__ == "__main__":
    unittest.main()
