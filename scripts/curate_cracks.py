"""Audit ZIP crack, buat contact sheet, lalu ekspor hasil peninjauan visual.

Gambar tidak diubah. Label poligon dikonversi menjadi bbox untuk YOLO detect,
sedangkan label asli disimpan di labels_original. Tidak melakukan training.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "archive (6).zip"
WORK = ROOT / ".venv" / "cracks_review_work"
OUTPUT = ROOT / "data" / "cracks_curated"
PRIORITY = {"test": 0, "valid": 1, "train": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DECISIONS = OUTPUT / "visual_decisions.csv"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_boxes(content):
    boxes, formats, warnings = [], Counter(), []
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if not all(map(math.isfinite, values)) or values[0] != 0:
            raise ValueError(f"baris {number}: angka/kelas tidak valid")
        if len(values) == 5:
            _, x, y, width, height = values
            corners = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
            if width <= 0 or height <= 0:
                raise ValueError(f"baris {number}: bbox kosong")
            formats["bbox"] += 1
        elif len(values) >= 7 and len(values) % 2 == 1:
            points = list(zip(values[1::2], values[2::2]))
            if any(min(point) < -1e-5 or max(point) > 1 + 1e-5 for point in points):
                raise ValueError(f"baris {number}: poligon di luar gambar")
            corners = [min(p[0] for p in points), min(p[1] for p in points),
                       max(p[0] for p in points), max(p[1] for p in points)]
            formats["polygon_to_bbox"] += 1
        else:
            raise ValueError(f"baris {number}: bukan bbox/poligon YOLO")
        if min(corners) < -1e-5 or max(corners) > 1 + 1e-5:
            raise ValueError(f"baris {number}: bbox di luar gambar")
        clipped = [max(0.0, min(1.0, value)) for value in corners]
        if clipped != corners:
            warnings.append("rounding_clipped_to_image")
        x1, y1, x2, y2 = clipped
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"baris {number}: bbox tidak mempunyai area")
        box = (0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        if box in boxes:
            warnings.append("duplicate_box_removed")
        else:
            boxes.append(box)
    return boxes, dict(formats), sorted(set(warnings))


def scan():
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(SOURCE) as archive:
        members = {item.filename: item for item in archive.infolist() if not item.is_dir()}
        if len(members) != sum(not item.is_dir() for item in archive.infolist()):
            raise ValueError("ZIP berisi nama file duplikat")
        for name in members:
            posix = PurePosixPath(name)
            if posix.is_absolute() or ".." in posix.parts or "\\" in name:
                raise ValueError(f"Path ZIP tidak aman: {name}")
        images = sorted(name for name in members if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS)
        group_names = sorted({re.sub(r"\.rf\.[0-9a-f]+$", "", PurePosixPath(name).stem, flags=re.I) for name in images})
        group_ids = {name: i for i, name in enumerate(group_names)}
        for index, name in enumerate(images):
            parts = list(PurePosixPath(name).parts)
            split = parts[0]
            if split not in PRIORITY or parts[1] != "images":
                raise ValueError(f"Struktur split tidak dikenal: {name}")
            stem = re.sub(r"\.rf\.[0-9a-f]+$", "", PurePosixPath(name).stem, flags=re.I)
            label = str(PurePosixPath(split, "labels", PurePosixPath(name).with_suffix(".txt").name))
            record = dict(image=name, label=label, split=split, group_name=stem,
                          group_id=group_ids[stem], status="valid", reason="", warnings=[])
            try:
                content = archive.read(name)
                with Image.open(BytesIO(content)) as opened:
                    oriented = ImageOps.exif_transpose(opened)
                    rgb = oriented.convert("RGB")
                    record.update(width=rgb.width, height=rgb.height,
                                  pixel_sha256=hashlib.sha256(
                                      f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest())
                    if min(rgb.size) < 32:
                        raise ValueError("gambar terlalu kecil")
                    rgb.close()
                    oriented.close()
                boxes, formats, warnings = read_boxes(archive.read(label))
                record.update(boxes=boxes, formats=formats, warnings=warnings)
                if not boxes:
                    record.update(status="excluded", reason="no_crack_annotation")
            except (OSError, ValueError, KeyError, UnicodeError, SyntaxError) as error:
                record.update(status="excluded", reason=str(error))
            records.append(record)
            if (index + 1) % 500 == 0:
                print(f"Audit: {index + 1}/{len(images)}", flush=True)
        groups = defaultdict(list)
        for record in records:
            groups[record["group_id"]].append(record)
        manifest = []
        for group_id, items in sorted(groups.items()):
            representative = next((r for r in items if r["status"] == "valid"), items[0])
            manifest.append(dict(group_id=group_id, group_name=representative["group_name"],
                                 split=representative["split"], variants=len(items),
                                 representative=representative["image"]))
        for start in range(0, len(manifest), 72):
            subset = manifest[start:start + 72]
            canvas = Image.new("RGB", (6 * 184, math.ceil(len(subset) / 6) * 178), "white")
            draw = ImageDraw.Draw(canvas)
            for cell, item in enumerate(subset):
                x, y = (cell % 6) * 184, (cell // 6) * 178
                with archive.open(item["representative"]) as stream, Image.open(stream) as opened:
                    thumb = ImageOps.contain(opened.convert("RGB"), (178, 151))
                    canvas.paste(thumb, (x, y))
                draw.text((x + 2, y + 153), f'{item["group_id"]:04d} {item["split"]} n={item["variants"]}', fill="black")
            canvas.save(WORK / f"sheet_{start // 72:02d}.jpg", quality=92)
    (WORK / "audit.json").write_text(json.dumps(records), encoding="utf-8")
    write_csv(OUTPUT / "group_manifest.csv", manifest,
              ["group_id", "group_name", "split", "variants", "representative"])
    report = dict(images=len(records), groups=len(groups),
                  status=dict(Counter(r["status"] for r in records)),
                  reasons=dict(Counter(r["reason"] for r in records if r["reason"])),
                  annotation_formats=dict(sum((Counter(r.get("formats", {})) for r in records), Counter())))
    (OUTPUT / "source_audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def export():
    records = json.loads((WORK / "audit.json").read_text(encoding="utf-8"))
    with DECISIONS.open(encoding="utf-8", newline="") as stream:
        decisions = {int(row["group_id"]): row for row in csv.DictReader(stream)}
    if set(decisions) != {r["group_id"] for r in records}:
        raise ValueError("Keputusan visual harus mencakup setiap kelompok foto")
    if any(row.get("reviewed") != "true" for row in decisions.values()):
        raise ValueError("Peninjauan visual belum lengkap")
    allowed = {"asphalt_candidate", "other_surface", "needs_review"}
    if any(row["category"] not in allowed for row in decisions.values()):
        raise ValueError("Kategori keputusan tidak dikenal")
    # Hindari kebocoran basename asal, tanpa memindahkan gambar antar split.
    group_split = {}
    for record in records:
        if record["status"] == "valid":
            group = record["group_name"]
            group_split[group] = min(group_split.get(group, record["split"]), record["split"], key=PRIORITY.get)
    hashes = {}
    exported = []
    with zipfile.ZipFile(SOURCE) as archive:
        for record in sorted(records, key=lambda r: (PRIORITY[r["split"]], r["image"])):
            row = dict(record)
            decision = decisions[row["group_id"]]
            row.update(category=decision["category"], visual_reason=decision["reason"])
            if row["status"] == "valid" and group_split[row["group_name"]] != row["split"]:
                row.update(status="excluded", reason="same_original_across_splits")
            if row["status"] == "valid":
                key = row["pixel_sha256"]
                previous = hashes.get(key)
                if previous:
                    # Anotasi berbeda pada piksel identik dikarantina untuk review,
                    # tidak diperlakukan sebagai dua observasi independen.
                    row.update(status="excluded", reason="exact_pixel_duplicate", duplicate_of=previous["image"])
                else:
                    hashes[key] = row
            bucket = row["category"] if row["status"] == "valid" else "excluded_quality"
            row["output_bucket"] = bucket
            image_path = OUTPUT / bucket / row["image"]
            original_path = OUTPUT / bucket / row["split"] / "labels_original" / PurePosixPath(row["label"]).name
            label_path = OUTPUT / bucket / row["label"]
            for target in (image_path, original_path, label_path):
                if target.exists():
                    raise FileExistsError(f"Output sudah ada: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(archive.read(row["image"]))
            if row["label"] in archive.namelist():
                original_path.write_bytes(archive.read(row["label"]))
            boxes = row.get("boxes", [])
            label_path.write_text("".join("0 " + " ".join(f"{v:.10f}" for v in box[1:]) + "\n" for box in boxes), encoding="utf-8")
            exported.append(row)
    for bucket in sorted({r["output_bucket"] for r in exported}):
        data = OUTPUT / bucket
        for split in PRIORITY:
            (data / split / "images").mkdir(parents=True, exist_ok=True)
            (data / split / "labels").mkdir(parents=True, exist_ok=True)
        (data / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n",
            encoding="utf-8")
    fields = ["image", "label", "split", "group_id", "group_name", "category", "visual_reason",
              "status", "reason", "output_bucket", "duplicate_of", "width", "height", "pixel_sha256", "boxes_count", "formats"]
    for row in exported:
        row["boxes_count"] = len(row.get("boxes", []))
    write_csv(OUTPUT / "curation_manifest.csv", exported, fields)
    summary = []
    for bucket in sorted({r["output_bucket"] for r in exported}):
        for split in ("train", "valid", "test"):
            selected = [r for r in exported if r["output_bucket"] == bucket and r["split"] == split]
            summary.append(dict(category=bucket, split=split, images=len(selected),
                                boxes=sum(r["boxes_count"] for r in selected),
                                original_groups=len({r["group_id"] for r in selected})))
    write_csv(OUTPUT / "summary.csv", summary, ["category", "split", "images", "boxes", "original_groups"])
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["scan", "export"])
    arguments = parser.parse_args()
    scan() if arguments.action == "scan" else export()
