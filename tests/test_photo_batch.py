import copy
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.api_models import Job, JobStatus, JobType
from backend.job_manager import JobManager
from backend.photo_batch import (
    PhotoBatchValidationError,
    _register_heif_opener_required,
    prepare_photo_batch,
    preprocess_photo_source,
    run_photo_batch,
    validate_photo_batch_plan,
)
from backend.seedvr2_manager import load_seedvr2_model

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependency-free API smoke environment
    Image = None

try:
    import pillow_heif
except ImportError:  # pragma: no cover - dependency-free API smoke environment
    pillow_heif = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(Image is not None, "Pillow is required for image pipeline tests")
class PhotoBatchPipelineTest(unittest.TestCase):
    def _write_oriented_jpeg(self, path: Path) -> None:
        image = Image.new("RGB", (2, 3), color=(20, 40, 60))
        exif = Image.Exif()
        exif[274] = 6
        exif[271] = "Example"
        exif[272] = "Travel Phone"
        exif[306] = "2026:08:09 12:34:56"
        image.save(path, format="JPEG", quality=100, subsampling=0, exif=exif)

    def _environment(self, import_root: Path, output_root: Path):
        return patch.dict(
            os.environ,
            {
                "MFLUX_PHOTO_IMPORT_ROOT": str(import_root),
                "MFLUX_PHOTO_BATCH_OUTPUT_ROOT": str(output_root),
            },
        )

    def test_preprocessing_transposes_exif_orientation_and_strips_output_exif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "oriented.jpg"
            destination = root / "normalized.png"
            self._write_oriented_jpeg(source_path)

            with source_path.open("rb") as source:
                details = preprocess_photo_source(
                    source,
                    filename=source_path.name,
                    destination=destination,
                )

            self.assertEqual(details["source_dimensions"], {"width": 2, "height": 3})
            self.assertEqual(details["normalized_dimensions"], {"width": 3, "height": 2})
            self.assertEqual(details["orientation_before"], 6)
            self.assertTrue(details["exif_transposed"])
            with Image.open(destination) as normalized:
                self.assertEqual(normalized.size, (3, 2))
                self.assertNotIn(274, normalized.getexif())

    def test_complete_copy_only_batch_preserves_original_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as import_dir, tempfile.TemporaryDirectory() as output_dir:
            import_root = Path(import_dir)
            output_root = Path(output_dir)
            album = import_root / "trip"
            album.mkdir()
            photo = album / "IMG_0001.JPG"
            self._write_oriented_jpeg(photo)
            (album / "IMG_0001.AAE").write_text("adjustment", encoding="utf-8")
            (album / "IMG_0002.MOV").write_bytes(b"video")
            original_hash = _sha256(photo)

            with self._environment(import_root, output_root):
                plan = prepare_photo_batch(
                    {
                        "directory": "trip",
                        "resolution": "2x",
                        "gps_mode": "suggest",
                        "location_overrides": {
                            "trip/IMG_0001.JPG": {"label": "Private trip place"}
                        },
                    }
                )

                def fake_enhancer(input_path, output_path, _seed, _settings):
                    with Image.open(input_path) as source:
                        source.save(output_path, format="PNG")

                events = []
                result = run_photo_batch(
                    plan,
                    enhance_one=fake_enhancer,
                    progress_callback=lambda event, data=None: events.append((event, data)),
                )

                batch_dir = output_root / plan["output_relative_directory"]
                manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
                sidecar_path = batch_dir / plan["files"][0]["sidecar_relative_path"]
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                enhanced = batch_dir / plan["files"][0]["output_relative_path"]

                self.assertEqual(plan["num_images"], 1)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(manifest["completed_count"], 1)
                self.assertEqual(_sha256(photo), original_hash)
                self.assertEqual(sidecar["source"]["captured_at_source"], "exif")
                self.assertEqual(
                    sidecar["source"]["location_candidate"]["label"],
                    "Private trip place",
                )
                self.assertEqual(sidecar["enhancement"]["network_access"], "disabled")
                self.assertFalse(sidecar["output"]["metadata_embedded"])
                self.assertEqual(sidecar["preprocessing"]["orientation_before"], 6)
                self.assertIsNotNone(sidecar["source"]["raw_exif_sidecar"])
                self.assertTrue((batch_dir / sidecar["source"]["raw_exif_sidecar"]).is_file())
                with Image.open(enhanced) as image:
                    self.assertEqual(image.size, (3, 2))
                    self.assertNotIn(274, image.getexif())
                self.assertTrue(any(event == "image_complete" for event, _ in events))

                with self.assertRaises(PhotoBatchValidationError):
                    run_photo_batch(plan, enhance_one=fake_enhancer)

    def test_output_root_inside_import_tree_is_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as import_dir:
            import_root = Path(import_dir)
            album = import_root / "trip"
            album.mkdir()
            self._write_oriented_jpeg(album / "photo.jpg")
            unsafe_output = import_root / "must-not-be-created"

            with self._environment(import_root, unsafe_output):
                plan = prepare_photo_batch({"directory": "trip"})
                with self.assertRaises(PhotoBatchValidationError):
                    run_photo_batch(plan, enhance_one=lambda *_args: None)
            self.assertFalse(unsafe_output.exists())

    def test_cancellation_writes_manifest_without_calling_model(self):
        with tempfile.TemporaryDirectory() as import_dir, tempfile.TemporaryDirectory() as output_dir:
            import_root = Path(import_dir)
            output_root = Path(output_dir)
            album = import_root / "trip"
            album.mkdir()
            self._write_oriented_jpeg(album / "photo.jpg")

            def must_not_run(*_args):
                raise AssertionError("enhancer should not run after cancellation")

            with self._environment(import_root, output_root):
                plan = prepare_photo_batch({"directory": "trip"})
                result = run_photo_batch(
                    plan,
                    enhance_one=must_not_run,
                    cancel_check=lambda: True,
                )

            batch_dir = output_root / plan["output_relative_directory"]
            manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(list((batch_dir / "images").iterdir()), [])

    def test_tampered_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as import_dir:
            import_root = Path(import_dir)
            album = import_root / "trip"
            album.mkdir()
            self._write_oriented_jpeg(album / "photo.jpg")
            with patch.dict(os.environ, {"MFLUX_PHOTO_IMPORT_ROOT": str(import_root)}):
                plan = prepare_photo_batch({"directory": "trip"})
            tampered = copy.deepcopy(plan)
            tampered["settings"]["softness"] = 0.9
            with self.assertRaises(PhotoBatchValidationError):
                validate_photo_batch_plan(tampered)

    @unittest.skipUnless(pillow_heif is not None, "pillow-heif is required for HEIF decoding")
    def test_heif_is_decoded_locally(self):
        pillow_heif.register_heif_opener(thumbnails=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "photo.heic"
            destination = root / "photo.png"
            Image.new("RGB", (4, 3), color=(10, 20, 30)).save(source_path, format="HEIF")
            with source_path.open("rb") as source:
                details = preprocess_photo_source(
                    source,
                    filename=source_path.name,
                    destination=destination,
                )
            self.assertEqual(details["decoder"], "pillow-heif")
            self.assertEqual(details["normalized_dimensions"], {"width": 4, "height": 3})


class PhotoBatchContractTest(unittest.TestCase):
    def test_heif_registration_disables_thumbnail_substitution(self):
        calls = []
        fake_module = types.ModuleType("pillow_heif")
        fake_module.register_heif_opener = lambda **kwargs: calls.append(kwargs)
        with patch.dict(sys.modules, {"pillow_heif": fake_module}):
            _register_heif_opener_required()
        self.assertEqual(calls, [{"thumbnails": False}])

    def test_job_manager_publishes_batch_result(self):
        manager = JobManager()
        job = Job(
            job_type=JobType.photo_batch,
            status=JobStatus.running,
            params={"_photo_batch_plan": {"batch_id": "example"}},
        )
        result = {
            "batch_id": "example",
            "status": "completed",
            "completed_count": 1,
        }
        with patch("backend.photo_batch.run_photo_batch", return_value=result):
            manager._run_photo_batch(job, job.params, lambda *_args: None)
        self.assertEqual(job.status, JobStatus.completed)
        self.assertEqual(job.progress.percent, 100.0)
        self.assertEqual(job.result, result)

    def test_seedvr2_loader_resolves_shared_cache_without_network(self):
        calls = {}

        class FakeModelConfig:
            @staticmethod
            def seedvr2_3b():
                return types.SimpleNamespace(model_name="example/seedvr2")

        def fake_seedvr2(**kwargs):
            calls["constructor"] = kwargs
            return "loaded-model"

        fake_huggingface = types.ModuleType("huggingface_hub")

        def fake_snapshot_download(**kwargs):
            calls["snapshot"] = kwargs
            return "/shared/cache/snapshot"

        fake_huggingface.snapshot_download = fake_snapshot_download
        with (
            patch(
                "backend.seedvr2_manager._seedvr2_runtime",
                return_value=(fake_seedvr2, FakeModelConfig, object()),
            ),
            patch.dict(sys.modules, {"huggingface_hub": fake_huggingface}),
        ):
            loaded = load_seedvr2_model()

        self.assertEqual(loaded, "loaded-model")
        self.assertEqual(
            calls["snapshot"],
            {"repo_id": "example/seedvr2", "local_files_only": True},
        )
        self.assertEqual(calls["constructor"]["model_path"], "/shared/cache/snapshot")


if __name__ == "__main__":
    unittest.main()
