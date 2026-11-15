"""Integrasi dua bobot asli dan regresi pemetaan kelas pada prediksi gabungan."""

from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_DEVICE", "cpu")

from fastapi.testclient import TestClient
from PIL import Image

import api


def png_bytes(size=(128, 96), mode="RGB", orientation=None):
    with Image.new(mode, size) as image:
        buffer = BytesIO()
        options = {}
        if orientation:
            exif = image.getexif()
            exif[274] = orientation
            options["exif"] = exif
        image.save(buffer, format="PNG", **options)
        return buffer.getvalue()


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def fake_model(name, bbox=(8, 24, 56, 70), confidence=0.9):
    result = SimpleNamespace(names={0: name}, boxes=SimpleNamespace(
        xyxy=Values([list(bbox)]), conf=Values([confidence]), cls=Values([0])))
    return SimpleNamespace(predict=Mock(return_value=[result]))


class CombinedApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def post(self, path="/predict/combined", content=None, name="road.png"):
        return self.client.post(path, files={"file": (name, png_bytes() if content is None else content, "image/png")})

    def test_actual_checkpoints_health_and_identity(self):
        response = self.client.get("/health/combined")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dict(status="ok", models_loaded={"pothole": True, "cracks": True}, combined_ready=True))
        info = self.client.get("/models").json()
        self.assertEqual(info["classes"], {"0": "pothole", "1": "cracks"})
        self.assertEqual(info["models"]["pothole"]["classes"], {"0": "pothole"})
        self.assertEqual(info["models"]["cracks"]["classes"], {"0": "cracks"})
        for key in ["pothole", "cracks"]:
            self.assertEqual(info["models"][key]["model"], "YOLO26s")
            self.assertEqual(info["models"][key]["scheme"], "A")
        self.assertEqual(info["models"]["cracks"]["checkpoint_sha256"], api.app_state["crack_metadata"]["sha256"])

    def test_actual_combined_inference_counts_and_coordinates(self):
        fixture = ROOT / "Hasil Run/yolo26s_A_roads_bridges_cracks_01/eval_test/val_batch0_labels.jpg"
        self.assertTrue(fixture.is_file(), "Fixture hasil training crack tidak tersedia")
        data = self.post(content=fixture.read_bytes(), name=fixture.name).json()
        self.assertGreater(data["counts"]["cracks"], 0)
        self.assertEqual(data["count"], sum(data["counts"].values()))
        self.assertEqual(data["count"], len(data["detections"]))
        self.assertEqual(data["models_used"], ["pothole", "cracks"])
        for item in data["detections"]:
            self.assertEqual(item["source_class_id"], 0)
            self.assertEqual(item["class_id"], 0 if item["class_name"] == "pothole" else 1)
            self.assertEqual(item["source_model"], item["class_name"])
            self.assertGreaterEqual(item["confidence"], 0.25)
            for key in ["x1", "y1", "x2", "y2"]:
                size = data["width"] if key.startswith("x") else data["height"]
                self.assertAlmostEqual(item["bbox"][key] / size, item["bbox_normalized"][key])
                self.assertGreaterEqual(item["bbox_normalized"][key], 0)
                self.assertLessEqual(item["bbox_normalized"][key], 1)

    def test_two_local_class_zero_outputs_are_distinct_and_overlap_is_retained(self):
        pothole = fake_model("pothole", confidence=0.95)
        crack = fake_model("cracks", confidence=0.85)
        with patch.dict(api.app_state, model=pothole, crack_model=crack), patch.object(api, "_read_image", wraps=api._read_image) as decode:
            response = self.post("/predict/combined?max_det=1&confidence=0.4")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(decode.call_count, 1)
        data = response.json()
        self.assertEqual(data["counts"], {"pothole": 1, "cracks": 1})
        self.assertEqual([item["class_id"] for item in data["detections"]], [0, 1])
        self.assertEqual([item["source_class_id"] for item in data["detections"]], [0, 0])
        self.assertEqual(data["detections"][0]["bbox"], data["detections"][1]["bbox"])
        self.assertEqual(pothole.predict.call_count, 1)
        self.assertEqual(crack.predict.call_count, 1)
        self.assertIs(pothole.predict.call_args.kwargs["source"], crack.predict.call_args.kwargs["source"])
        for model in [pothole, crack]:
            self.assertEqual(model.predict.call_args.kwargs["conf"], 0.4)
            self.assertEqual(model.predict.call_args.kwargs["max_det"], 1)
        self.assertEqual(data["max_detections_per_model"], 1)

    def test_combined_jpeg_contains_both_annotation_colors(self):
        pothole = fake_model("pothole", bbox=(5, 30, 35, 60))
        crack = fake_model("cracks", bbox=(70, 30, 105, 60))
        with patch.dict(api.app_state, model=pothole, crack_model=crack):
            response = self.post("/predict/combined/image")
        self.assertEqual(response.status_code, 200, response.text[:100])
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-pothole-count"], "1")
        self.assertEqual(response.headers["x-crack-count"], "1")
        with Image.open(BytesIO(response.content)) as image:
            self.assertEqual(image.size, (128, 96))
            red = image.getpixel((6, 45))
            blue = image.getpixel((71, 45))
            self.assertGreater(red[0], red[2] + 50)
            self.assertGreater(blue[2], blue[0] + 50)

    def test_exif_orientation_and_rgb_conversion_are_shared(self):
        response = self.post(content=png_bytes(size=(128, 64), orientation=6))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.json()["width"], response.json()["height"]), (64, 128))
        for mode in ["L", "RGBA"]:
            response = self.post(content=png_bytes(mode=mode))
            self.assertEqual(response.status_code, 200, response.text)

    def test_missing_model_never_returns_partial_success(self):
        with patch.dict(api.app_state, {"model": api.app_state["model"]}, clear=True):
            response = self.client.get("/health/combined")
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json()["combined_ready"])
            self.assertEqual(self.post().status_code, 503)

    def test_detector_failure_never_returns_partial_success(self):
        broken = SimpleNamespace(predict=Mock(side_effect=RuntimeError("internal checkpoint error")))
        with patch.dict(api.app_state, crack_model=broken), patch.object(api.logger, "exception"):
            response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal checkpoint error", response.text)

    def test_combined_upload_validation_and_limit(self):
        self.assertEqual(self.post(content=b"").status_code, 400)
        self.assertEqual(self.post(content=b"not an image").status_code, 400)
        for query in ["confidence=0", "iou=1.1", "max_det=301"]:
            self.assertEqual(self.post("/predict/combined?" + query).status_code, 422)
        response = self.post("/predict/combined?max_det=1")
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(response.json()["count"], 2)

    def test_ui_and_openapi_use_combined_route(self):
        response = self.client.get("/ui")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/predict/combined?confidence=", response.text)
        self.assertIn("/health/combined", response.text)
        spec = self.client.get("/openapi.json").json()
        self.assertIn("multipart/form-data", spec["paths"]["/predict/combined"]["post"]["requestBody"]["content"])
        self.assertIn("image/jpeg", spec["paths"]["/predict/combined/image"]["post"]["responses"]["200"]["content"])


if __name__ == "__main__":
    unittest.main()
