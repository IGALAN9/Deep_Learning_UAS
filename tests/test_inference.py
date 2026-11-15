"""Uji model dan fungsi API tanpa socket/event loop HTTP."""

from io import BytesIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / ".ultralytics"
CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))
os.environ.setdefault("YOLO_DEVICE", "cpu")

from fastapi import HTTPException, Response, UploadFile
from PIL import Image

import api


def upload(name="blank.png", size=(128, 96), mode="RGB", format="PNG", orientation=None):
    stream = BytesIO()
    with Image.new(mode, size) as image:
        options = {}
        if orientation:
            exif = image.getexif()
            exif[274] = orientation
            options["exif"] = exif
        image.save(stream, format=format, **options)
    stream.seek(0)
    return UploadFile(filename=name, file=stream)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api._load_model()

    @classmethod
    def tearDownClass(cls):
        api.app_state.clear()

    def test_model_identity_and_health(self):
        info = api.model_info()
        self.assertEqual(info.model, "YOLO26s")
        self.assertEqual(info.scheme, "A")
        self.assertEqual(info.classes, {0: "pothole"})
        self.assertEqual(info.ultralytics_version, "8.4.202")
        self.assertEqual(info.checkpoint_sha256, api.app_state["metadata"]["sha256"])
        self.assertTrue(api.health(Response()).model_loaded)

    def test_empty_detections_and_serialization(self):
        result = api.predict(upload())
        self.assertEqual((result.width, result.height), (128, 96))
        self.assertFalse(result.detected)
        self.assertEqual(result.count, 0)
        self.assertEqual(result.detections, [])
        self.assertEqual(api.PredictResponse.model_validate_json(result.model_dump_json()), result)

    def test_actual_detections_and_normalized_coordinates(self):
        fixture = ROOT / "Hasil Run/yolo26s_A_kaggle_pothole_01/test/val_batch0_labels.jpg"
        if not fixture.is_file():
            self.skipTest("Montage hasil eksperimen tidak tersedia.")
        with fixture.open("rb") as stream:
            result = api.predict(UploadFile(filename=fixture.name, file=stream))
        self.assertGreater(result.count, 0)
        for item in result.detections:
            self.assertEqual(item.class_id, 0)
            self.assertEqual(item.class_name, "pothole")
            self.assertGreaterEqual(item.confidence, 0.25)
            for key in ("x1", "y1", "x2", "y2"):
                normalized = getattr(item.bbox_normalized, key)
                dimension = result.width if key.startswith("x") else result.height
                self.assertGreaterEqual(normalized, 0)
                self.assertLessEqual(normalized, 1)
                self.assertAlmostEqual(getattr(item.bbox, key) / dimension, normalized)

    def test_batch_grayscale_and_rgba(self):
        result = api.predict_batch([upload(mode="L"), upload(mode="RGBA")], confidence=0.4)
        self.assertEqual(result.total, 2)
        self.assertEqual([item.confidence_threshold for item in result.results], [0.4, 0.4])

    def test_jpeg_output_and_exif_orientation(self):
        result = api.predict_image(upload(size=(128, 64), format="JPEG", orientation=6))
        self.assertEqual(result.media_type, "image/jpeg")
        with Image.open(BytesIO(result.body)) as image:
            self.assertEqual(image.size, (64, 128))
        self.assertIn("x-detection-count", result.headers)

    def test_invalid_empty_and_unsupported_input(self):
        files = [
            (UploadFile(filename="empty.png", file=BytesIO()), 400),
            (UploadFile(filename="bad.png", file=BytesIO(b"invalid")), 400),
            (UploadFile(filename="truncated.png", file=BytesIO(b"\x89PNG\r\n\x1a\nbroken")), 400),
            (upload(format="GIF"), 415),
        ]
        for file, status in files:
            with self.subTest(status=status):
                with self.assertRaises(HTTPException) as raised:
                    api.predict(file)
                self.assertEqual(raised.exception.status_code, status)

    def test_image_limits_and_batch_limit(self):
        files = [
            UploadFile(filename="big.png", file=BytesIO(b"0" * (api.MAX_UPLOAD_BYTES + 1))),
            upload(size=(4001, 4000), mode="L"),
        ]
        for file in files:
            with self.assertRaises(HTTPException) as raised:
                api.predict(file)
            self.assertEqual(raised.exception.status_code, 413)
        with self.assertRaises(HTTPException) as raised:
            api.predict_batch([upload() for _ in range(11)])
        self.assertEqual(raised.exception.status_code, 422)

    def test_readiness_failure(self):
        with patch.dict(api.app_state, {}, clear=True):
            response = Response()
            self.assertFalse(api.health(response).model_loaded)
            self.assertEqual(response.status_code, 503)
            with self.assertRaises(HTTPException) as raised:
                api.model_info()
            self.assertEqual(raised.exception.status_code, 503)

    def test_openapi_upload_and_image_content_types(self):
        spec = api.app.openapi()
        self.assertIn("multipart/form-data", spec["paths"]["/predict"]["post"]["requestBody"]["content"])
        self.assertIn("image/jpeg", spec["paths"]["/predict/image"]["post"]["responses"]["200"]["content"])


if __name__ == "__main__":
    unittest.main()
