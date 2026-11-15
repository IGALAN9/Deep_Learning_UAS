"""Kurasi archive (7), lalu tambahkan crack hanya ke train dataset v2.

Tahap: scan -> review contact sheet/visual_decisions.csv -> export -> verify.
Dataset v1 dan ZIP sumber tidak diubah. Tidak menjalankan training model.
"""

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/archive (7).zip"
BASE = ROOT / "data/cracks_curated/asphalt_candidate"
OUT = ROOT / "data/cracks_curated_v2"
DATASET = OUT / "asphalt_candidate"
WORK = ROOT / ".venv/archive7_review_work"
PACKAGE = ROOT / "data/cracks_asphalt_candidates_yolo_v2.zip"
SPLITS = ("train", "valid", "test")
CATEGORIES = {"asphalt_candidate", "other_surface", "needs_review"}
DECISIONS = OUT / "archive7_visual_decisions.csv"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def features(content):
    with Image.open(BytesIO(content)) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
    gray = image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    thumb = np.asarray(gray, dtype=np.float32)
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    hashes = []
    for base in (small, np.fliplr(small)):
        for turns in range(4):
            coefficients = cv2.dct(np.ascontiguousarray(np.rot90(base, turns)))[:8, :8].flatten()
            bits = coefficients > np.median(coefficients[1:])
            bits[0] = False
            hashes.append(sum(int(value) << index for index, value in enumerate(bits)))
    return image, pixel_hash, hashes, thumb


def parse_cracks(content):
    boxes, objects, clipped, duplicates, excess_max = [], Counter(), 0, 0, 0.0
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 9 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"baris {number}: bukan poligon quadrilateral valid")
        if values[0] not in (0, 1, 2):
            raise ValueError(f"baris {number}: kelas tidak dikenal")
        objects[int(values[0])] += 1
        if values[0] != 1:
            continue
        excess = max(0.0, -min(values[1:]), max(values[1:]) - 1)
        if excess > 0.02:
            raise ValueError(f"baris {number}: koordinat melebihi toleransi 2%")
        clipped += excess > 0
        excess_max = max(excess_max, excess)
        xs, ys = values[1::2], values[2::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"baris {number}: crack tidak punya area setelah clipping")
        box = (0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        if box in boxes:
            duplicates += 1
        else:
            boxes.append(box)
    return boxes, dict(objects), clipped, duplicates, excess_max


def label_bytes(boxes):
    return ("\n".join("0 " + " ".join(f"{v:.10f}" for v in box[1:]) for box in boxes) + "\n").encode()


def sequence_group(stem):
    # Tanggal snapshot adalah petunjuk konservatif, bukan ID video terverifikasi.
    match = re.search(r"(?:vlcsnap[-_])?(\d{4})[-_]?(\d{2})[-_]?(\d{2})", stem)
    return "capture_date_" + "-".join(match.groups()) if match else "unknown_sequence"


def scan():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if DECISIONS.exists():
        raise FileExistsError("Review sudah ada; scan menolak menimpa keputusan visual")
    records, hashes, thumbs = [], [], []
    with zipfile.ZipFile(SOURCE) as archive:
        names = [x.filename for x in archive.infolist() if not x.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Nama anggota ZIP duplikat")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError("Path ZIP tidak aman")
        coco = json.loads(archive.read("data/annotations_coco.json"))
        categories = {x["id"]: x["name"].lower() for x in coco["categories"]}
        if categories != {0: "pothole", 1: "crack", 2: "manhole"}:
            raise ValueError(f"Pemetaan kelas berbeda: {categories}")
        image_names = sorted(x for x in names if x.startswith("data/images/") and x.lower().endswith(".jpg"))
        for index, name in enumerate(image_names):
            stem = PurePosixPath(name).stem
            row = dict(source_index=index, image=name, label=f"data/labels/{stem}.txt",
                       source_group=sequence_group(stem), status="valid", reason="")
            try:
                raw = archive.read(row["label"])
                row["raw_class_counts"] = dict(Counter(int(float(line.split()[0]))
                                                     for line in raw.decode("utf-8-sig").splitlines() if line.strip()))
                boxes, objects, clipped, repeated, excess = parse_cracks(raw)
                image, pixel_hash, phashes, thumb = features(archive.read(name))
                row.update(boxes=boxes, source_class_counts=objects, boxes_count=len(boxes),
                           clipped_crack_rows=clipped, duplicate_boxes_removed=repeated,
                           max_coordinate_excess=excess, width=image.width, height=image.height,
                           pixel_sha256=pixel_hash)
                if not boxes:
                    row.update(status="no_crack", reason="tidak_ada_anotasi_kelas_crack")
            except (ValueError, OSError, KeyError) as error:
                row.update(status="invalid", reason=str(error), boxes=[], boxes_count=0)
                phashes, thumb = [0] * 8, np.zeros((64, 64), dtype=np.float32)
            records.append(row)
            hashes.append(phashes)
            thumbs.append(thumb)
        write_json(WORK / "audit.json", records)
        np.savez_compressed(WORK / "features.npz", hashes=np.asarray(hashes, dtype=np.uint64),
                            thumbs=np.asarray(thumbs, dtype=np.uint8))
        candidates = [r for r in records if r["status"] == "valid"]
        decisions = []
        page_size, columns, cell_width, cell_height = 72, 6, 208, 148
        sheet_dir = OUT / "review_sheets"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(candidates), page_size):
            page = candidates[start:start + page_size]
            sheet = Image.new("RGB", (columns * cell_width, math.ceil(len(page) / columns) * cell_height), "white")
            draw = ImageDraw.Draw(sheet)
            filename = f"archive7_{start // page_size + 1:02d}.jpg"
            for offset, row in enumerate(page):
                with Image.open(BytesIO(archive.read(row["image"]))) as im:
                    image = ImageOps.exif_transpose(im).convert("RGB")
                image.thumbnail((204, 116), Image.Resampling.LANCZOS)
                overlay = ImageDraw.Draw(image)
                for _, xc, yc, w, h in row["boxes"]:
                    overlay.rectangle(((xc - w / 2) * image.width, (yc - h / 2) * image.height,
                                       (xc + w / 2) * image.width, (yc + h / 2) * image.height), outline="cyan", width=1)
                x, y = (offset % columns) * cell_width, (offset // columns) * cell_height
                sheet.paste(image, (x + 2, y))
                draw.text((x + 2, y + 117), f"ID {row['source_index']:04d}  crack={row['boxes_count']}", fill="black")
                draw.text((x + 2, y + 131), PurePosixPath(row["image"]).stem[:30], fill="black")
                decisions.append(dict(source_index=row["source_index"], image=row["image"],
                                      category="", reason="", reviewed="false", contact_sheet=filename))
            sheet.save(sheet_dir / filename, quality=88)
    write_csv(DECISIONS, decisions, ["source_index", "image", "category", "reason", "reviewed", "contact_sheet"])
    summary = dict(source_images=len(records),
                   images_with_cracks=sum(r.get("raw_class_counts", {}).get(1, 0) > 0 for r in records),
                   images_with_valid_crack_labels=len(candidates),
                   status_counts=dict(Counter(r["status"] for r in records)),
                   crack_boxes=sum(r["boxes_count"] for r in candidates),
                   clipped_rows=sum(r.get("clipped_crack_rows", 0) for r in records),
                   contact_sheets=math.ceil(len(candidates) / page_size), categories=categories)
    write_json(OUT / "source_audit_summary.json", summary)
    print(json.dumps(summary), flush=True)


def similarity(a_hashes, a_thumb, b_hashes, b_thumb):
    # DCT hashes dicocokkan pada delapan orientasi, kemudian dicek selisih piksel.
    distance = min(int(a ^ b).bit_count() for a in a_hashes for b in b_hashes)
    if distance > 4:
        return None
    a = a_thumb.astype(np.float32) - a_thumb.mean()
    b = b_thumb.astype(np.float32) - b_thumb.mean()
    mae = min(float(np.abs(a - np.rot90(base, turns)).mean())
              for base in (b, np.fliplr(b)) for turns in range(4))
    return (distance, mae) if mae <= 8 else None


def export():
    if DATASET.exists() or PACKAGE.exists():
        raise FileExistsError("Hasil v2 sudah ada; ekspor menolak menimpa dataset/paket")
    records = json.loads((WORK / "audit.json").read_text(encoding="utf-8"))
    with DECISIONS.open(encoding="utf-8", newline="") as stream:
        decisions = {int(r["source_index"]): r for r in csv.DictReader(stream)}
    expected = {r["source_index"] for r in records if r["status"] == "valid"}
    if set(decisions) != expected:
        raise ValueError("Daftar review tidak sesuai audit")
    for row in decisions.values():
        if row["reviewed"] != "true" or row["category"] not in CATEGORIES or not row["reason"]:
            raise ValueError(f"Review belum lengkap: {row['source_index']}")
    # Lengkapi audit dari label sumber, termasuk foto dengan geometri tidak valid.
    source_counts = Counter()
    source_crack_images = 0
    with zipfile.ZipFile(SOURCE) as archive:
        for row in records:
            counts = Counter(int(float(line.split()[0])) for line in archive.read(row["label"]).decode("utf-8-sig").splitlines() if line.strip())
            row["raw_class_counts"] = dict(counts)
            source_counts.update(counts)
            source_crack_images += counts[1] > 0
    audit_summary = json.loads((OUT / "source_audit_summary.json").read_text(encoding="utf-8"))
    audit_summary.update(images_with_cracks=source_crack_images, images_with_valid_crack_labels=len(expected),
                         raw_object_counts=dict(source_counts))
    write_json(OUT / "source_audit_summary.json", audit_summary)
    cache = np.load(WORK / "features.npz")
    hashes, thumbs = cache["hashes"], cache["thumbs"]
    existing, holdouts, seen = [], [], {}
    for split in ("test", "valid", "train"):
        for path in sorted((BASE / split / "images").iterdir()):
            _, pixel_hash, phashes, thumb = features(path.read_bytes())
            row = dict(image=f"{split}/images/{path.name}", split=split,
                       pixel_sha256=pixel_hash, hashes=phashes, thumb=thumb)
            existing.append(row)
            seen.setdefault(pixel_hash, row["image"])
            if split != "train":
                holdouts.append(row)
    # Base v1 dipertahankan persis, termasuk anotasi dan split aslinya.
    shutil.copytree(BASE, DATASET)
    pairs, new_names = [], []
    with zipfile.ZipFile(SOURCE) as archive:
        for row in records:
            i = row["source_index"]
            row.update(category="excluded_quality", output_bucket="excluded_quality", output_image="",
                       duplicate_of="", visual_reason="")
            if row["status"] == "valid":
                review = decisions[i]
                row.update(category=review["category"], output_bucket=review["category"], visual_reason=review["reason"])
                if row["category"] == "asphalt_candidate":
                    if row["pixel_sha256"] in seen:
                        row.update(output_bucket="excluded_quality", reason="exact_pixel_duplicate",
                                   duplicate_of=seen[row["pixel_sha256"]])
                    else:
                        for old in holdouts:
                            result = similarity(hashes[i], thumbs[i], old["hashes"], old["thumb"])
                            if result is not None:
                                distance, mae = result
                                pairs.append(dict(image=row["image"], holdout=old["image"],
                                                  phash_distance=distance, centered_mae=round(mae, 5)))
                                row.update(output_bucket="excluded_quality", reason="near_duplicate_of_holdout",
                                           duplicate_of=old["image"])
                    if row["output_bucket"] == "asphalt_candidate":
                        seen[row["pixel_sha256"]] = row["image"]
            name = "archive7__" + PurePosixPath(row["image"]).name
            label_name = PurePosixPath(name).with_suffix(".txt").name
            if row["output_bucket"] == "asphalt_candidate":
                dest = DATASET / "train"
                new_names.append(name)
                row["output_image"] = f"asphalt_candidate/train/images/{name}"
                row["reason"] = "added_to_train"
            else:
                dest = OUT / "archive7_separated" / row["output_bucket"]
                row["output_image"] = f"archive7_separated/{row['output_bucket']}/images/{name}"
            for folder in ("images", "labels", "labels_original"):
                (dest / folder).mkdir(parents=True, exist_ok=True)
            target = dest / "images" / name
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(archive.read(row["image"]))
            (dest / "labels_original" / label_name).write_bytes(archive.read(row["label"]))
            (dest / "labels" / label_name).write_bytes(label_bytes(row["boxes"]) if row["boxes"] else b"")
    fields = ["source_index", "image", "label", "source_group", "status", "category", "visual_reason",
              "output_bucket", "reason", "output_image", "duplicate_of", "width", "height", "pixel_sha256",
              "boxes_count", "source_class_counts", "raw_class_counts", "clipped_crack_rows", "duplicate_boxes_removed", "max_coordinate_excess"]
    write_csv(OUT / "archive7_manifest.csv", records, fields)
    write_json(WORK / "export_audit.json", records)
    write_csv(OUT / "holdout_similarity_candidates.csv", pairs,
              ["image", "holdout", "phash_distance", "centered_mae"])
    write_json(OUT / "similarity_audit.json", dict(
        incoming_images_checked=sum(r["category"] == "asphalt_candidate" for r in records),
        existing_images_pixel_hashed=len(existing), holdout_images_checked=len(holdouts),
        near_duplicate_pairs=len(pairs), exact_duplicates_excluded=sum(r["reason"] == "exact_pixel_duplicate" for r in records),
        method="Exact pixel SHA256; all 8 rotation/flip pHashes with Hamming <=4 and centered 64px MAE <=8 against every base validation/test image",
        limitation="Heuristic only; source recording/road identities are not independently verified. All archive7 images remain train-only."))
    metadata = OUT / "source_metadata"
    shutil.copytree(ROOT / "data/cracks_curated/source_metadata", metadata / "archive6")
    for filename in ("provenance.json", "validation.json", "similarity_audit.json", "visual_decisions.csv"):
        shutil.copy2(ROOT / "data/cracks_curated" / filename, metadata / "archive6" / filename)
    with zipfile.ZipFile(SOURCE) as archive:
        for filename in ("README.md", "annotations_coco.json"):
            target = metadata / "archive7" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read("data/" + filename))
    write_json(OUT / "provenance.json", dict(
        source_archive="data/archive (7).zip", source_sha256=sha256(SOURCE),
        base_dataset="data/cracks_curated/asphalt_candidate", base_provenance="source_metadata/archive6/provenance.json",
        source_class_mapping={"0": "pothole (removed)", "1": "crack -> output class 0", "2": "manhole (removed)"},
        polygons="Axis-aligned min/max envelope, clipped to image boundaries; original mixed-class labels retained verbatim",
        images="Original bytes, names prefixed archive7__, no resize/crop/augmentation",
        split_policy="Append accepted archive7 images to TRAIN ONLY. Original v1 train/valid/test files unchanged.",
        source_capture_group="Date inferred from filename for traceability; not a verified video ID"))
    write_json(OUT / "new_train_files.json", new_names)
    # Satu YAML pada root paket, tanpa path absolut komputer pembuat.
    (DATASET / "data.yaml").write_text("train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n", encoding="utf-8")
    print(json.dumps(dict(added_train=len(new_names), buckets=dict(Counter(r["output_bucket"] for r in records)),
                          near_duplicate_pairs=len(pairs))), flush=True)


def verify_and_package():
    records = json.loads((WORK / "export_audit.json").read_text(encoding="utf-8"))
    new_names = set(json.loads((OUT / "new_train_files.json").read_text(encoding="utf-8")))
    expected_rows = {"archive7__" + PurePosixPath(r["image"]).name: r for r in records if r["reason"] == "added_to_train"}
    if new_names != set(expected_rows):
        raise ValueError("Daftar tambahan train tidak cocok")
    summary, pixel_splits, base_preserved, new_verified = [], {}, 0, 0
    for split in SPLITS:
        original_names = {p.name for p in (BASE / split / "images").iterdir()}
        images = list((DATASET / split / "images").iterdir())
        actual_names = {p.name for p in images}
        if actual_names != original_names | (new_names if split == "train" else set()):
            raise ValueError(f"Keanggotaan split berubah: {split}")
        expected_labels = {p.with_suffix(".txt").name for p in images}
        for folder in ("labels", "labels_original"):
            if {p.name for p in (DATASET / split / folder).iterdir()} != expected_labels:
                raise ValueError(f"Pasangan image-label berbeda: {split}/{folder}")
        boxes = 0
        for image_path in images:
            _, pixel_hash, _, _ = features(image_path.read_bytes())
            pixel_splits.setdefault(pixel_hash, set()).add(split)
            label_path = DATASET / split / "labels" / image_path.with_suffix(".txt").name
            lines = label_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise ValueError(f"Label kosong: {label_path}")
            for line in lines:
                v = list(map(float, line.split()))
                if len(v) != 5 or not all(math.isfinite(x) for x in v) or v[0] != 0:
                    raise ValueError(f"Label bukan kelas cracks/bbox: {label_path}")
                _, x, y, w, h = v
                if not (w > 0 and h > 0 and min(x-w/2,y-h/2) >= -1e-8 and max(x+w/2,y+h/2) <= 1+1e-8):
                    raise ValueError(f"BBox di luar gambar: {label_path}")
                boxes += 1
            if image_path.name in original_names:
                for folder in ("images", "labels", "labels_original"):
                    name = image_path.name if folder == "images" else image_path.with_suffix(".txt").name
                    if (BASE / split / folder / name).read_bytes() != (DATASET / split / folder / name).read_bytes():
                        raise ValueError(f"File base berubah: {split}/{folder}/{name}")
                    base_preserved += 1
            else:
                row = expected_rows[image_path.name]
                if split != "train" or label_path.read_bytes() != label_bytes(row["boxes"]):
                    raise ValueError("Label atau split tambahan tidak sesuai audit")
                new_verified += 1
        summary.append(dict(split=split, images=len(images), boxes=boxes,
                            previous_images=len(original_names), added_images=len(actual_names-original_names)))
    cross_split = [h for h, splits in pixel_splits.items() if len(splits) > 1]
    if cross_split:
        raise ValueError("Hash piksel yang sama ditemukan antar split")
    with zipfile.ZipFile(SOURCE) as archive:
        for row in records:
            dest = OUT / row["output_image"]
            raw_dest = dest.parent.parent / "labels_original" / dest.with_suffix(".txt").name
            if dest.read_bytes() != archive.read(row["image"]):
                raise ValueError("Gambar sumber berubah")
            if raw_dest.read_bytes() != archive.read(row["label"]):
                raise ValueError("Label asli sumber berubah")
            if row["status"] == "invalid":
                try:
                    parse_cracks(archive.read(row["label"]))
                except ValueError:
                    continue
                raise ValueError("Label yang disisihkan ternyata tidak mereproduksi error audit")
            parsed, _, clipped, _, _ = parse_cracks(archive.read(row["label"]))
            if parsed != [tuple(x) for x in row["boxes"]] or clipped != row.get("clipped_crack_rows", 0):
                raise ValueError("Konversi tidak reproducible")
    validation = dict(passed=True, source_images_accounted_for=len(records),
                      base_files_byte_identical=base_preserved, new_train_images_verified=new_verified,
                      validation_test_byte_identical=True, exact_cross_split_duplicates=0,
                      all_new_images_train_only=True, all_labels_single_class_bbox=True,
                      image_and_original_label_bytes_preserved=True, split_summary=summary)
    write_json(OUT / "summary.json", summary)
    write_csv(OUT / "summary.csv", summary, ["split", "images", "boxes", "previous_images", "added_images"])
    write_json(OUT / "validation.json", validation)
    counts = Counter(r["output_bucket"] for r in records)
    clipped_new = sum(r.get("clipped_crack_rows", 0) for r in records if r["reason"] == "added_to_train")
    table = "\n".join(f"| {r['split']} | {r['previous_images']} | {r['added_images']} | {r['images']} | {r['boxes']} |" for r in summary)
    readme = f"""# Dataset crack v2: archive 6 + archive 7

Tambahan archive (7) hanya masuk **train**. Seluruh gambar dan label train/valid/test dari v1 dipertahankan identik. V1 dan kedua ZIP sumber tetap tersedia tanpa perubahan. Skrip ini tidak menjalankan training model.

| Split | Gambar v1 | Tambahan | Total gambar | Bounding box |
| --- | ---: | ---: | ---: | ---: |
{table}

## Seleksi archive 7

- Sumber: {len(records)} gambar; {sum(r.get('raw_class_counts',{}).get('1',r.get('raw_class_counts',{}).get(1,0))>0 for r in records)} memiliki anotasi crack. {sum(r['status']=='valid' for r in records)} foto memiliki label crack yang lolos geometri.
- Ditambahkan ke train: {len(new_names)} gambar. Distribusi hasil pemisahan: {dict(counts)}.
- Foto tanpa anotasi crack disisihkan, bukan dianggap background yang sudah diverifikasi.
- {sum(r['status']=='invalid' for r in records)} foto disisihkan karena geometri anotasi tidak valid (bbox crack tinggi nol); seluruh anotasinya tetap tersedia pada label asli.
- Hanya kelas sumber `1 = crack` diambil dan dipetakan ke kelas hasil `0 = cracks`. Anotasi `0 = pothole` dan `2 = manhole` dihapus dari label deteksi. Objek tersebut masih dapat terlihat di foto.
- Poligon quadrilateral diubah menjadi bbox min/max sejajar sumbu. {clipped_new} baris crack terpilih sedikit melewati batas gambar dan dibatasi ke 0-1 (toleransi sumber maksimum 2%; excess aktual dapat dilihat di manifest).
- Foto dan label asli disimpan dengan byte identik, memakai awalan nama `archive7__`. Label asli memuat semua kelas sumber dan bukan label training dataset satu kelas ini.

## Peninjauan dan batasan

Seluruh foto tambahan dengan crack diperiksa melalui contact sheet untuk menilai konteks jalan/permukaan dan kelayakan visual dasar. Penilaian material bersifat visual; kandidat aspal tidak berarti materialnya sudah terverifikasi. Bbox asli tidak dianotasi ulang dan kebenaran semantik setiap objek belum diaudit secara menyeluruh.

Archive 7 banyak berisi frame jalan berurutan. Semuanya ditempatkan dalam train; jangan membagi ulang secara acak per gambar karena frame berdekatan dapat bocor ke validasi/test. `source_group` menyimpan tanggal dari nama file sebagai petunjuk konservatif, bukan ID video terverifikasi.

Duplikat piksel diperiksa terhadap seluruh dataset v1 dan tambahan yang sudah diterima. Kemiripan terhadap seluruh 165 gambar validasi/test v1 diperiksa dengan pHash delapan orientasi (Hamming <=4) dan thumbnail mean-centered 64px (MAE <=8). Kandidat serupa disisihkan. Metode ini heuristik dan tidak menjamin bebas semua kemiripan lokasi/rekaman.

Validasi/test tetap 118/47 gambar dari archive 6; hasil evaluasi nantinya terutama mengukur domain archive 6. Performa pada gaya kamera/rekaman archive 7 memerlukan set uji terpisah berdasarkan rekaman/lokasi. Jumlah test yang kecil juga membatasi kestabilan metrik.

## Pemakaian

Ekstrak `cracks_asphalt_candidates_yolo_v2.zip`, lalu gunakan `data.yaml` pada root hasil ekstraksi untuk YOLO detect satu kelas. Path split relatif terhadap lokasi YAML. Belum ada augmentasi tambahan atau training yang dijalankan.

`images/`: foto asli. `labels/`: bbox crack satu kelas. `labels_original/`: anotasi sumber tanpa perubahan; jangan gunakan sebagai label training satu kelas.

Laporan: `archive7_manifest.csv`, `archive7_visual_decisions.csv`, `source_audit_summary.json`, `similarity_audit.json`, `validation.json`, `provenance.json`, `summary.csv/json`. `source_metadata` memuat atribusi dan catatan sumber. Contact sheet tersedia di `review_sheets`; foto yang disisihkan tersedia di `archive7_separated` pada folder kurasi penuh, di luar ZIP training.

Skrip reproduksi: `scripts/append_cracks_archive7.py` dengan tahap scan, peninjauan manual CSV, export, verify. Scan/export menolak menimpa hasil yang sudah ada.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (DATASET / "README.md").write_text(readme, encoding="utf-8")
    # Preview terpilih dari archive 7 dengan bbox hasil konversi.
    chosen = sorted(new_names)
    chosen = [chosen[round(i * (len(chosen)-1) / 11)] for i in range(12)] if len(chosen) >= 12 else chosen
    preview = Image.new("RGB", (1200, math.ceil(len(chosen)/3) * 257), "white")
    draw = ImageDraw.Draw(preview)
    for i, name in enumerate(chosen):
        row = expected_rows[name]
        with Image.open(DATASET / "train/images" / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((396, 228), Image.Resampling.LANCZOS)
        overlay = ImageDraw.Draw(image)
        for _, x, y, w, h in row["boxes"]:
            overlay.rectangle(((x-w/2)*image.width, (y-h/2)*image.height,
                               (x+w/2)*image.width, (y+h/2)*image.height), outline="cyan", width=2)
        x, y = (i % 3) * 400, (i // 3) * 257
        preview.paste(image, (x, y))
        draw.text((x+2, y+230), f"ID {row['source_index']} | {row['boxes_count']} cracks", fill="black")
    preview.save(OUT / "preview_archive7_bbox.jpg", quality=90)
    if PACKAGE.exists():
        raise FileExistsError("ZIP v2 sudah ada")
    with zipfile.ZipFile(PACKAGE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as package:
        for path in sorted(DATASET.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(DATASET).as_posix())
        reports = ["archive7_manifest.csv", "archive7_visual_decisions.csv", "source_audit_summary.json",
                   "similarity_audit.json", "holdout_similarity_candidates.csv", "validation.json",
                   "provenance.json", "summary.csv", "summary.json", "preview_archive7_bbox.jpg"]
        for name in reports:
            package.write(OUT / name, "reports/" + name)
        for path in sorted((OUT / "source_metadata").rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(PACKAGE) as package:
        bad = package.testzip()
        if bad is not None:
            raise ValueError(f"CRC ZIP gagal: {bad}")
        if len(package.namelist()) != len(set(package.namelist())) or package.namelist().count("data.yaml") != 1:
            raise ValueError("Struktur paket tidak valid")
    write_json(OUT / "package_validation.json", dict(passed=True, zip_crc_passed=True,
               filename=PACKAGE.name, bytes=PACKAGE.stat().st_size, sha256=sha256(PACKAGE)))
    print(json.dumps(validation), flush=True)
    print(json.dumps(dict(package=str(PACKAGE), bytes=PACKAGE.stat().st_size)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("scan", "export", "verify"))
    arguments = parser.parse_args()
    {"scan": scan, "export": export, "verify": verify_and_package}[arguments.stage]()
