import hashlib
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from backend.api_server import APIServer
from backend.api_models import Job, JobStatus, JobType
from backend.photo_imports import (
    PhotoImportValidationError,
    _gps_to_decimal,
    _normalize_capture_date,
    extract_photo_metadata,
    inventory_photos,
    resolve_import_directory,
)


class PhotoImportUnitTest(unittest.TestCase):
    def test_directory_cannot_escape_root_or_follow_outside_symlink(self):
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(root_dir)
            album = root / "album"
            album.mkdir()

            self.assertEqual(resolve_import_directory("album", root=root), album.resolve())
            with self.assertRaises(PhotoImportValidationError):
                resolve_import_directory("../outside", root=root)

            link = root / "outside-link"
            link.symlink_to(Path(outside_dir), target_is_directory=True)
            with self.assertRaises(PhotoImportValidationError):
                resolve_import_directory("outside-link", root=root)

    def test_inventory_preserves_original_and_uses_local_gps_candidate(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            album = root / "trip"
            album.mkdir()
            photo = album / "IMG_0001.JPG"
            original = b"original-photo-bytes"
            photo.write_bytes(original)
            before = hashlib.sha256(photo.read_bytes()).hexdigest()

            extracted = {
                "image": {"width": 4032, "height": 3024, "format": "JPEG", "mode": "RGB"},
                "captured_at": "2026-08-09T12:00:00+02:00",
                "captured_at_source": "exif",
                "timezone_known": True,
                "device": {"make": "Example", "model": "Phone", "lens_model": None, "software": None},
                "orientation": 1,
                "_gps": {"latitude": 37.8882, "longitude": -4.7794, "altitude_m": 120.0},
            }
            with patch("backend.photo_imports.extract_photo_metadata", return_value=extracted.copy()):
                result = inventory_photos("trip", root=root)

            item = result["items"][0]
            self.assertEqual(item["location_candidate"]["source"], "exif_gps")
            self.assertFalse(item["location_candidate"]["external_lookup_performed"])
            self.assertEqual(result["privacy"]["external_requests_performed"], 0)
            self.assertEqual(result["originals"]["write_operations_performed"], 0)
            self.assertEqual(hashlib.sha256(photo.read_bytes()).hexdigest(), before)

    def test_inventory_skips_symlinked_image_and_enforces_limit(self):
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(root_dir)
            album = root / "trip"
            album.mkdir()
            (album / "one.jpg").write_bytes(b"one")
            (album / "two.jpg").write_bytes(b"two")
            outside = Path(outside_dir) / "private.jpg"
            outside.write_bytes(b"outside")
            (album / "linked.jpg").symlink_to(outside)

            extracted = {
                "image": {"width": 1, "height": 1, "format": "JPEG", "mode": "RGB"},
                "captured_at": "2026-08-09T12:00:00",
                "captured_at_source": "exif",
                "timezone_known": False,
                "device": {"make": None, "model": None, "lens_model": None, "software": None},
                "orientation": None,
                "_gps": None,
            }
            with (
                patch.dict(os.environ, {"MFLUX_PHOTO_IMPORT_MAX_FILES": "1"}),
                patch("backend.photo_imports.extract_photo_metadata", return_value=extracted.copy()),
            ):
                result = inventory_photos("trip", root=root)

            self.assertEqual(result["summary"]["image_count"], 1)
            self.assertTrue(result["summary"]["truncated"])
            self.assertNotIn("linked.jpg", {item["name"] for item in result["items"]})

    def test_gps_can_be_disabled_or_replaced_by_user_override(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            album = root / "trip"
            album.mkdir()
            (album / "photo.jpg").write_bytes(b"photo")
            extracted = {
                "image": {"width": 1, "height": 1, "format": "JPEG", "mode": "RGB"},
                "captured_at": "2026-08-09T12:00:00",
                "captured_at_source": "exif",
                "timezone_known": False,
                "device": {"make": None, "model": None, "lens_model": None, "software": None},
                "orientation": None,
                "_gps": {"latitude": 1.0, "longitude": 2.0, "altitude_m": None},
            }
            with patch("backend.photo_imports.extract_photo_metadata", return_value=extracted.copy()):
                disabled = inventory_photos("trip", root=root, gps_mode="disabled")
            self.assertIsNone(disabled["items"][0]["location_candidate"])

            override = {
                "trip/photo.jpg": {
                    "label": "My chosen place",
                    "latitude": 3,
                    "longitude": 4,
                }
            }
            with patch("backend.photo_imports.extract_photo_metadata", return_value=extracted.copy()):
                overridden = inventory_photos("trip", root=root, location_overrides=override)
            candidate = overridden["items"][0]["location_candidate"]
            self.assertEqual(candidate["source"], "user_override")
            self.assertEqual(candidate["label"], "My chosen place")
            self.assertEqual((candidate["latitude"], candidate["longitude"]), (3.0, 4.0))

            with self.assertRaises(PhotoImportValidationError):
                inventory_photos(
                    "trip",
                    root=root,
                    location_overrides={"trip/missing.jpg": {"disabled": True}},
                )

    def test_gps_and_capture_date_normalization(self):
        self.assertAlmostEqual(_gps_to_decimal((37, 53, 17.52), "N"), 37.8882, places=4)
        self.assertAlmostEqual(_gps_to_decimal((4, 46, 45.84), "W"), -4.7794, places=4)
        captured, timezone_known = _normalize_capture_date(
            "2026:08:09 12:34:56", "+02:00", "25"
        )
        self.assertEqual(captured, "2026-08-09T12:34:56.250000+02:00")
        self.assertTrue(timezone_known)

    def test_exif_date_device_and_gps_are_normalized(self):
        class FakeExif(dict):
            def get_ifd(self, tag):
                return {
                    34665: {
                        36867: "2026:08:09 12:34:56",
                        36881: "+02:00",
                        42036: "Example Lens",
                    },
                    34853: {
                        1: "N",
                        2: (37, 53, 17.52),
                        3: "W",
                        4: (4, 46, 45.84),
                        5: 0,
                        6: 120,
                    },
                }.get(tag, {})

        class FakeImage:
            width = 4032
            height = 3024
            format = "JPEG"
            mode = "RGB"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getexif(self):
                return FakeExif({271: "Example", 272: "Travel Phone", 274: 1})

        fake_image_module = types.SimpleNamespace(open=lambda source: FakeImage())
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = fake_image_module

        with tempfile.NamedTemporaryFile() as source_file:
            source_stat = os.fstat(source_file.fileno())
            with patch.dict(sys.modules, {"PIL": fake_pil}):
                metadata = extract_photo_metadata(
                    io.BytesIO(b"photo"),
                    stat=source_stat,
                    filename="photo.jpg",
                )

        self.assertEqual(metadata["captured_at"], "2026-08-09T12:34:56+02:00")
        self.assertEqual(metadata["device"]["model"], "Travel Phone")
        self.assertEqual(metadata["device"]["lens_model"], "Example Lens")
        self.assertEqual(metadata["_gps"]["latitude"], 37.8882)
        self.assertEqual(metadata["_gps"]["longitude"], -4.7794)


class PhotoImportAPITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        album = self.root / "album"
        album.mkdir()
        (album / "photo.jpg").write_bytes(b"not-a-real-jpeg")
        self.env = patch.dict(os.environ, {"MFLUX_PHOTO_IMPORT_ROOT": str(self.root)})
        self.env.start()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), APIServer)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.env.stop()
        self.temp_dir.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=2
        )
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"Content-Type": "application/json"} if body is not None else {}
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        data = json.loads(response.read())
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, response_headers, data

    def test_config_and_inventory_endpoints(self):
        status, _, config = self.request("GET", "/api/v1/photo-imports/config")
        self.assertEqual(status, 200)
        self.assertTrue(config["privacy"]["local_only"])
        self.assertFalse(config["privacy"]["external_services_used"])

        status, _, inventory = self.request(
            "POST",
            "/api/v1/photo-imports/inventory",
            {"directory": "album", "gps_mode": "disabled"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(inventory["summary"]["image_count"], 1)
        self.assertEqual(inventory["items"][0]["relative_path"], "photo.jpg")
        self.assertNotIn(str(self.root), json.dumps(inventory))

    def test_traversal_and_non_local_origin_are_rejected(self):
        status, _, _ = self.request(
            "POST", "/api/v1/photo-imports/inventory", {"directory": "../escape"}
        )
        self.assertEqual(status, 400)

        status, headers, _ = self.request(
            "GET",
            "/api/v1/photo-imports/config",
            headers={"Origin": "https://example.com"},
        )
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

        status, _, _ = self.request(
            "GET",
            "/api/v1/photo-imports/config",
            headers={"Host": "evil.example"},
        )
        self.assertEqual(status, 403)

    def test_photo_batch_plan_and_submission_are_local_and_server_created(self):
        status, _, plan = self.request(
            "POST",
            "/api/v1/photo-batches/plan",
            {"directory": "album", "gps_mode": "disabled"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(plan["num_images"], 1)
        self.assertTrue(plan["originals_preserved"])
        self.assertFalse(plan["external_services_used"])
        self.assertNotIn(str(self.root), json.dumps(plan))

        class FakeManager:
            submitted = None

            def submit_job(self, job_type, params):
                self.submitted = (job_type, params)
                return Job(job_type=job_type, status=JobStatus.queued, params=params)

        manager = FakeManager()
        with patch("backend.job_manager.get_job_manager", return_value=manager):
            status, _, submitted = self.request(
                "POST",
                "/api/v1/generate",
                {"type": "photo_batch", "directory": "album", "gps_mode": "disabled"},
            )
        self.assertEqual(status, 202)
        self.assertEqual(submitted["type"], "photo_batch")
        self.assertEqual(manager.submitted[0], JobType.photo_batch)
        self.assertIn("_photo_batch_plan", manager.submitted[1])
        self.assertEqual(manager.submitted[1]["num_images"], 1)

if __name__ == "__main__":
    unittest.main()
