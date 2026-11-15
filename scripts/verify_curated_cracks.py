"""Verifikasi byte gambar, label YOLO, split, dan buat pratinjau hasil kurasi."""

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile
import zlib

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cracks_curated"
SOURCE = ROOT / "data" / "archive (6).zip"


def main():
    with (OUT / "curation_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["output_bucket"] == "asphalt_candidate"]
    assert len(rows) == 3458
    split_counts, box_counts = Counter(), Counter()
    group_sets, hash_splits = defaultdict(set), defaultdict(set)
    with zipfile.ZipFile(SOURCE) as source:
        for row in rows:
            image = OUT / row["output_bucket"] / row["image"]
            assert (zlib.crc32(image.read_bytes()) & 0xffffffff) == source.getinfo(row["image"]).CRC
            original = OUT / row["output_bucket"] / row["split"] / "labels_original" / Path(row["label"]).name
            assert (zlib.crc32(original.read_bytes()) & 0xffffffff) == source.getinfo(row["label"]).CRC
        for row in selected:
            lines = (OUT / "asphalt_candidate" / row["label"]).read_text().strip().splitlines()
            assert lines
            split_counts[row["split"]] += 1
            group_sets[row["split"]].add(row["group_id"])
            hash_splits[row["pixel_sha256"]].add(row["split"])
            for line in lines:
                values = list(map(float, line.split()))
                assert len(values) == 5 and values[0] == 0 and all(map(math.isfinite, values))
                _, x, y, width, height = values
                assert width > 0 and height > 0
                assert min(x - width / 2, y - height / 2) >= -1e-8
                assert max(x + width / 2, y + height / 2) <= 1 + 1e-8
                box_counts[row["split"]] += 1
            assert len(lines) == int(row["boxes_count"])
        for name in ("README.dataset.txt", "README.roboflow.txt", "data.yaml"):
            destination = OUT / "source_metadata" / ("data.yaml.original.txt" if name == "data.yaml" else name)
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(source.read(name))
    assert not (group_sets["train"] & group_sets["valid"] or group_sets["train"] & group_sets["test"]
                or group_sets["valid"] & group_sets["test"])
    assert all(len(splits) == 1 for splits in hash_splits.values())
    config = dict(line.split(":", 1) for line in (OUT / "asphalt_candidate/data.yaml").read_text().splitlines() if ":" in line)
    assert ast.literal_eval(config["names"].strip()) == ["cracks"] and int(config["nc"]) == 1
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        assert len(list((OUT / "asphalt_candidate" / config[key].strip()).glob("*.jpg"))) == split_counts[split]
    report = dict(
        result="passed", source_images=len(rows), selected_images=len(selected),
        selected_boxes=sum(box_counts.values()), selected_original_groups=len(set.union(*group_sets.values())),
        split_images=dict(split_counts), split_boxes=dict(box_counts),
        checks=["all output image bytes match source ZIP CRC32", "all original label bytes match source ZIP CRC32",
                "all selected labels nonempty and class 0", "all selected labels have five numeric bbox fields",
                "all selected boxes have positive area and fit within image",
                "no original filename groups cross selected splits", "no exact-pixel hashes cross selected splits",
                "data.yaml paths and counts valid"],
        surface_classification="visual candidates; material identity not verified")
    (OUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with SOURCE.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = dict(
        source="danishghaffar786/roads-and-bridges-cracks-yolov8-format", source_archive=SOURCE.name,
        archive_sha256=archive_hash,
        review_method="Visual inspection of 21 contact sheets, one representative per 1441 original filename groups; category propagated to source augmentations. Close-up material assignment is provisional.",
        positive_only=True, class_mapping={"0": "cracks"},
        polygon_conversion="axis-aligned bounding box from min/max normalized polygon vertices",
        pixel_policy="original JPEG bytes preserved", splits="original train/valid/test preserved; no resplit",
        code=["scripts/curate_cracks.py", "scripts/audit_crack_similarity.py", "scripts/verify_curated_cracks.py"])
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    samples = [selected[round(i * (len(selected) - 1) / 11)] for i in range(12)]
    canvas = Image.new("RGB", (4 * 252, 3 * 278), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(samples):
        with Image.open(OUT / "asphalt_candidate" / row["image"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (246, 246))
            mark = ImageDraw.Draw(thumb)
            for line in (OUT / "asphalt_candidate" / row["label"]).read_text().splitlines():
                _, x, y, width, height = map(float, line.split())
                mark.rectangle(((x - width / 2) * thumb.width, (y - height / 2) * thumb.height,
                                (x + width / 2) * thumb.width, (y + height / 2) * thumb.height),
                               outline="#ff3b30", width=2)
            x0, y0 = (index % 4) * 252, (index // 4) * 278
            canvas.paste(thumb, (x0, y0))
            draw.text((x0 + 3, y0 + 249), f"group {row['group_id']} | {row['split']} | {row['boxes_count']} box", fill="black")
    canvas.save(OUT / "preview_bbox.jpg", quality=93)
    print(json.dumps(report, indent=2), flush=True)
    print("Bucket counts:", dict(Counter(row["output_bucket"] for row in rows)))
    print("Excluded reasons:", dict(Counter(row["reason"] for row in rows if row["status"] != "valid")))
    print("Source formats among selected:", dict(sum((Counter(ast.literal_eval(row["formats"])) for row in selected), Counter())))


if __name__ == "__main__":
    main()
