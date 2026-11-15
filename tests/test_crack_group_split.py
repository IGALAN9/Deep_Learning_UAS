"""Regresi pemisahan kelompok, reproduksibilitas, dan koneksi notebook ke data crack."""

import importlib.util
import ast
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("crack_split", ROOT / "scripts/prepare_cracks_group_split.py")
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)


class CrackSplitTests(unittest.TestCase):
    def png(self, array):
        stream = BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        return stream.getvalue()

    def test_rotated_duplicate_and_original_group_are_transitively_joined(self):
        rng = np.random.default_rng(7)
        first = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        other = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        independent = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        values = [first, np.rot90(first), other, independent]
        origins = ["first", "second", "second", "independent"]
        records, hashes, thumbs = [], [], []
        for index, array in enumerate(values):
            _, _, pixel, phashes, thumb = split.image_features(self.png(array))
            records.append(dict(name=f"image{index}.png", pixel_sha256=pixel, origin_group=origins[index]))
            hashes.append(phashes)
            thumbs.append(thumb)
        groups, pairs = split.group_images(records, hashes, thumbs, progress=None)
        self.assertEqual(records[0]["group_id"], records[1]["group_id"])
        self.assertEqual(records[1]["group_id"], records[2]["group_id"])
        self.assertNotEqual(records[0]["group_id"], records[3]["group_id"])
        self.assertTrue(any({p["image_a"], p["image_b"]} == {"image0.png", "image1.png"} for p in pairs))

    def test_same_seed_is_reproducible_and_different_seed_changes_assignment(self):
        records, groups = [], {}
        for group in range(40):
            indices = []
            for variant in range(3):
                indices.append(len(records))
                records.append(dict(name=f"g{group}_{variant}.png", source="archive6"))
            groups[f"group{group:02d}"] = indices
        first = split.allocate_groups(records, groups, seed=42)
        second = split.allocate_groups(records, groups, seed=42)
        third = split.allocate_groups(records, groups, seed=17)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        for indices in groups.values():
            self.assertEqual(len({records[i]["split"] for i in indices}), 1)
        self.assertEqual({r["split"] for r in records}, set(split.SPLIT_NAMES))

    def test_real_zip_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix="crack_split_test_", dir=ROOT / ".venv") as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(ROOT.resolve()))
            source = root / "input.zip"
            rng = np.random.default_rng(19)
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("data.yaml", "train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n")
                for index in range(18):
                    original_split = split.SPLIT_NAMES[index % 3]
                    archive.writestr(f"{original_split}/images/image{index}.png", self.png(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)))
                    archive.writestr(f"{original_split}/labels/image{index}.txt", "0 0.5 0.5 0.3 0.4\n")
            info = dict(dataset_id="round_trip_fixture", description="Dataset pengujian, semua permukaan.")
            a = split.prepare_dataset(source, root / "a", seed=42, ratios=(0.6, 0.2, 0.2), progress=None, dataset_info=info)
            b = split.prepare_dataset(source, root / "b", seed=42, ratios=(0.6, 0.2, 0.2), progress=None, dataset_info=info)
            self.assertEqual(a["fingerprint"], b["fingerprint"])
            self.assertEqual(a["dataset_id"], info["dataset_id"])
            self.assertEqual(json.loads((root / "a/dataset_info.json").read_text()), info)
            self.assertTrue(split.verify_dataset(root / "a")["passed"])
            self.assertEqual(sum(r["images"] for r in a["summary"]), 18)
            label = next((root / "a/train/labels").glob("*.txt"))
            label.write_text("0 0.5 0.5 0.2 0.2\n")
            with self.assertRaisesRegex(ValueError, "berubah"):
                split.verify_dataset(root / "a")

    def test_rejects_wrong_class_and_zero_area_box(self):
        for raw in [b"1 0.5 0.5 0.2 0.2", b"0 0.5 0.5 0 0.2", b"0 0.99 0.5 0.2 0.2"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                split.validate_boxes(raw)

    def test_archive6_only_policy_rejects_archive7_before_writing_output(self):
        with tempfile.TemporaryDirectory(prefix="crack_split_test_", dir=ROOT / ".venv") as tmp:
            folder = Path(tmp)
            self.assertTrue(folder.resolve().is_relative_to(ROOT.resolve()))
            source = folder / "mixed.zip"
            rng = np.random.default_rng(59)
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("data.yaml", "train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n")
                for index in range(6):
                    name = f"image{index}" if index else "archive7__20250216_164325"
                    archive.writestr(f"train/images/{name}.png", self.png(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)))
                    archive.writestr(f"train/labels/{name}.txt", "0 0.5 0.5 0.3 0.4\n")
            output = folder / "output"
            with self.assertRaisesRegex(ValueError, "Sumber dataset"):
                split.prepare_dataset(source, output, progress=None, allowed_sources={"archive6"})
            self.assertFalse(output.exists())

    def test_archive6_notebook_scope_checks_accept_archive6_and_reject_mixed_or_wrong_count(self):
        path = ROOT / "deteksi_crack_YOLO26s_ARCHIVE6.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        config = "".join(notebook["cells"][3]["source"])
        self.assertIn("cracks_archive6_group_split_80_10_10.zip", config)
        self.assertIn("'yolo_cracks_archive6'", config)
        self.assertIn("SKEMA_LIST = ['A']", config)
        loading = "".join(notebook["cells"][6]["source"])
        checks = [node for node in ast.parse(loading).body if isinstance(node, ast.If)
                  and ("EXPECTED_SOURCES" in ast.unparse(node.test) or "EXPECTED_IMAGES" in ast.unparse(node.test))]
        self.assertEqual(len(checks), 2)
        code = compile(ast.Module(body=checks, type_ignores=[]), "archive6_scope_guard", "exec")

        class Column(list):
            @property
            def str(self):
                return self

            def startswith(self, prefix):
                return Column(x.startswith(prefix) for x in self)

            def any(self):
                return any(self)

        class Table(dict):
            def __len__(self):
                return len(self["source"])

        for count, has_archive7 in [(1232, False), (1232, True), (1231, False)]:
            df = Table(source=Column(["archive6"] * count), name=Column([f"frame{i}.jpg" for i in range(count)]))
            if has_archive7:
                df["source"][0] = "archive7"
                df["name"][0] = "archive7__frame.jpg"
            values = dict(df=df, EXPECTED_SOURCES={"archive6"}, EXPECTED_IMAGES=1232)
            if has_archive7 or count != 1232:
                with self.assertRaises(ValueError):
                    exec(code, values)
            else:
                exec(code, values)

    def test_roads_bridges_notebook_requires_full_dataset_identity(self):
        notebook = json.loads((ROOT / "deteksi_crack_YOLO26s_ROADS_BRIDGES.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"full_notebook_cell_{index}", "exec")
        config = "".join(notebook["cells"][3]["source"])
        self.assertIn("roads_bridges_cracks_yolo26.zip", config)
        self.assertIn("EXPECTED_IMAGES = 3439", config)
        self.assertIn("'yolo_roads_bridges_cracks'", config)
        self.assertIn("MODEL = 'yolo26s.pt'", config)
        loading = "".join(notebook["cells"][6]["source"])
        identity_check = next(node for node in ast.parse(loading).body if isinstance(node, ast.With))
        code = compile(ast.Module(body=[identity_check], type_ignores=[]), "full_dataset_identity", "exec")
        with tempfile.TemporaryDirectory(prefix="crack_split_test_", dir=ROOT / ".venv") as tmp:
            folder = Path(tmp)
            self.assertTrue(folder.resolve().is_relative_to(ROOT.resolve()))
            source = folder / "dataset.zip"
            for dataset_id in ["roads_bridges_cracks_full", "asphalt_subset", "archive6_archive7_mixed"]:
                with zipfile.ZipFile(source, "w") as archive:
                    archive.writestr("dataset_info.json", json.dumps(dict(dataset_id=dataset_id)))
                values = dict(zipfile=zipfile, json=json, DATA_ZIP=source, EXPECTED_DATASET_ID="roads_bridges_cracks_full")
                if dataset_id == "roads_bridges_cracks_full":
                    exec(code, values)
                else:
                    with self.assertRaises(ValueError):
                        exec(code, values)

    def test_notebook_training_uses_prepared_crack_dataset_and_supports_resume(self):
        notebook = json.loads((ROOT / "deteksi_crack_YOLO26s_skemaAB_GROUP_SPLIT.ipynb").read_text(encoding="utf-8"))
        code = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "def train_run(skema):" in "".join(c["source"]))
        with tempfile.TemporaryDirectory(prefix="crack_split_test_", dir=ROOT / ".venv") as tmp:
            folder = Path(tmp)
            self.assertTrue(folder.resolve().is_relative_to(ROOT.resolve()))
            calls = []

            class FakeYOLO:
                def __init__(self, source):
                    self.ckpt = {"epoch": 0}

                def info(self, **options):
                    pass

                def train(self, **options):
                    calls.append(options)
                    (folder / "weights").mkdir(exist_ok=True)
                    for name in ["best.pt", "last.pt"]:
                        (folder / "weights" / name).write_bytes(b"test stub, not model weights")

            ns = dict(YOLO=FakeYOLO, MODEL="yolo26s.pt", DATA="/content/cracks/runtime.yaml",
                      EPOCHS=1, BATCH_SIZE=2, IMG_SIZE=640, DEVICE=0, WORKERS=0, TRAIN_SEED=42,
                      PATIENCE=10, CLOSE_MOSAIC=5, RUNS_ROOT=folder, RUN_NAMES={"A": "cracks_A"},
                      SPLIT_SEED=42, run_paths=lambda skema: (folder, folder / "weights/best.pt", folder / "weights/last.pt"))
            exec(compile(code, "notebook_training_cell", "exec"), ns)
            ns["train_run"]("A")
            self.assertEqual(calls[0]["data"], "/content/cracks/runtime.yaml")
            self.assertNotIn("augmentations", calls[0])
            ns["train_run"]("A")
            self.assertTrue(calls[1]["resume"])
            self.assertEqual(calls[1]["data"], calls[0]["data"])


if __name__ == "__main__":
    unittest.main()
