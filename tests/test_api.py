"""Uji integrasi memakai bobot asli; tidak memalsukan output detektor."""

from io import BytesIO
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / ".ultralytics"
CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))
os.environ.setdefault("YOLO_DEVICE", "cpu")

from fastapi.testclient import TestClient
from PIL import Image

import api


def image_bytes(size=(128, 96), mode="RGB", format="PNG", orientation=None):
    with Image.new(mode, size) as image:
        output = BytesIO()
        options = {}
        if orientation:
            exif = image.getexif()
            exif[274] = orientation
            options["exif"] = exif
        image.save(output, format=format, **options)
        return output.getvalue()


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_and_model_identity(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "model_loaded": True})
        info = self.client.get("/model").json()
        self.assertEqual(info["model"], "YOLO26s")
        self.assertEqual(info["scheme"], "A")
        self.assertEqual(info["classes"], {"0": "pothole"})
        self.assertEqual(info["image_size"], 640)
        self.assertEqual(info["training_ultralytics_version"], "8.4.202")
        self.assertEqual(info["checkpoint_sha256"], api.app_state["metadata"]["sha256"])

    def test_predict_blank_image(self):
        response = self.client.post("/predict", files={"file": ("blank.png", image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual((result["width"], result["height"]), (128, 96))
        self.assertEqual(result["count"], len(result["detections"]))
        self.assertEqual(result["detected"], bool(result["detections"]))
        self.assertFalse(result["detected"])
        self.assertGreater(result["processing_time_ms"], 0)

    def test_predict_real_run_image_and_box_coordinates(self):
        fixture = ROOT / "Hasil Run/yolo26s_A_kaggle_pothole_01/test/val_batch0_labels.jpg"
        if not fixture.is_file():
            self.skipTest("Montage hasil eksperimen tidak tersedia.")
        response = self.client.post("/predict", files={"file": (fixture.name, fixture.read_bytes(), "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertGreater(result["count"], 0)
        for detection in result["detections"]:
            self.assertEqual(detection["class_id"], 0)
            self.assertEqual(detection["class_name"], "pothole")
            self.assertGreaterEqual(detection["confidence"], 0.25)
            for key in ("x1", "y1", "x2", "y2"):
                coordinate = detection["bbox_normalized"][key]
                self.assertGreaterEqual(coordinate, 0)
                self.assertLessEqual(coordinate, 1)
                size = result["width"] if key.startswith("x") else result["height"]
                self.assertAlmostEqual(detection["bbox"][key] / size, coordinate)

    def test_batch_grayscale_and_rgba(self):
        response = self.client.post("/predict/batch?confidence=0.4", files=[
            ("files", ("gray.png", image_bytes(mode="L"), "image/png")),
            ("files", ("rgba.png", image_bytes(size=(96, 128), mode="RGBA"), "image/png")),
        ])
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual([row["filename"] for row in data["results"]], ["gray.png", "rgba.png"])
        self.assertEqual([row["confidence_threshold"] for row in data["results"]], [0.4, 0.4])

    def test_annotated_jpeg_and_exif_orientation(self):
        content = image_bytes(size=(128, 64), format="JPEG", orientation=6)
        response = self.client.post("/predict/image", files={"file": ("rotate.jpg", content, "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertIn("x-detection-count", response.headers)
        with Image.open(BytesIO(response.content)) as result:
            self.assertEqual(result.format, "JPEG")
            self.assertEqual(result.size, (64, 128))

    def test_empty_corrupt_and_unsupported_images(self):
        cases = [
            ("empty.png", b"", "image/png", 400),
            ("bad.png", b"this is not a picture", "image/png", 400),
            ("truncated.png", b"\x89PNG\r\n\x1a\nbroken", "image/png", 400),
            ("unsupported.gif", image_bytes(format="GIF"), "image/gif", 415),
        ]
        for name, content, content_type, status in cases:
            with self.subTest(name=name):
                response = self.client.post("/predict", files={"file": (name, content, content_type)})
                self.assertEqual(response.status_code, status, response.text)

    def test_file_size_and_resolution_limits(self):
        response = self.client.post("/predict", files={"file": (
            "large.png", b"0" * (api.MAX_UPLOAD_BYTES + 1), "image/png",
        )})
        self.assertEqual(response.status_code, 413)
        response = self.client.post("/predict", files={"file": (
            "large-pixels.png", image_bytes(size=(4001, 4000), mode="L"), "image/png",
        )})
        self.assertEqual(response.status_code, 413)

    def test_invalid_parameters_missing_file_and_batch_limit(self):
        for query in ("confidence=0", "confidence=1.1", "iou=-1", "max_det=301"):
            with self.subTest(query=query):
                response = self.client.post("/predict?" + query, files={"file": ("x.png", image_bytes(), "image/png")})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.post("/predict").status_code, 422)
        content = image_bytes()
        response = self.client.post("/predict/batch", files=[
            ("files", (f"{index}.png", content, "image/png")) for index in range(11)
        ])
        self.assertEqual(response.status_code, 422)

    def test_openapi_documents_upload_and_image_response(self):
        spec = self.client.get("/openapi.json").json()
        self.assertIn("multipart/form-data", spec["paths"]["/predict"]["post"]["requestBody"]["content"])
        image_response = spec["paths"]["/predict/image"]["post"]["responses"]["200"]["content"]
        self.assertIn("image/jpeg", image_response)


if __name__ == "__main__":
    unittest.main()
