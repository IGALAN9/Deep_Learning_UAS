"""Siapkan seluruh Roads and Bridges Cracks untuk YOLO26 detect, tanpa filter permukaan."""

from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image

from curate_cracks import read_boxes
from prepare_cracks_group_split import file_sha256, prepare_dataset, package_dataset, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/Roads and Bridges Cracks (YOLOv8 Format).zip"
WORK = ROOT / ".venv/roads_bridges_full_work"
CANONICAL = WORK / "roads_bridges_full_bbox_source.zip"
OUT = ROOT / "data/roads_bridges_cracks_yolo26"
PACKAGE = ROOT / "data/roads_bridges_cracks_yolo26.zip"


def csv_bytes(rows, fields):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def bbox_bytes(boxes):
    return ("\n".join("0 " + " ".join(f"{value:.10f}" for value in box[1:]) for box in boxes) + "\n").encode()


def normalize_source():
    WORK.mkdir(parents=True, exist_ok=True)
    if CANONICAL.exists():
        raise FileExistsError("Sumber normalisasi sudah ada; menolak menimpa")
    rows, formats, pixel_groups = [], Counter(), defaultdict(list)
    source_hash = file_sha256(SOURCE)
    with zipfile.ZipFile(SOURCE) as source:
        names = [x.filename for x in source.infolist() if not x.is_dir()]
        if len(set(names)) != len(names):
            raise ValueError("Anggota ZIP duplikat")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError("Path ZIP tidak aman")
        images = sorted(n for n in names if "/images/" in n and PurePosixPath(n).suffix.lower() in {".jpg", ".jpeg", ".png"})
        for index, name in enumerate(images):
            p = PurePosixPath(name)
            split = p.parts[0]
            if split not in {"train", "valid", "test"}:
                raise ValueError("Split sumber tidak dikenal")
            label = f"{split}/labels/{p.with_suffix('.txt').name}"
            row = dict(image=name, label=label, previous_split=split, status="keep", reason="", duplicate_of="", boxes=[])
            try:
                raw = source.read(label)
                boxes, counts, warnings = read_boxes(raw)
                formats.update(counts)
                row.update(boxes=boxes, source_formats=counts, warnings=warnings, bbox_count=len(boxes))
                if not boxes:
                    row.update(status="excluded", reason="empty_label_not_verified_background")
                else:
                    with Image.open(BytesIO(source.read(name))) as opened:
                        opened.load()
                        if opened.getexif().get(274, 1) != 1 or getattr(opened, "n_frames", 1) != 1:
                            raise ValueError("EXIF/multiframe perlu ditinjau")
                        image = opened.convert("RGB")
                        if min(image.size) < 32:
                            raise ValueError("Gambar terlalu kecil")
                    pixel_hash = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
                    row.update(pixel_sha256=pixel_hash, width=image.width, height=image.height)
                    pixel_groups[pixel_hash].append(row)
            except (ValueError, OSError, KeyError) as error:
                row.update(status="excluded", reason="invalid_image_or_label: " + str(error))
            rows.append(row)
            if (index + 1) % 500 == 0:
                print(f"Normalisasi: {index+1}/{len(images)} foto", flush=True)
        for copies in pixel_groups.values():
            if len({tuple(sorted(row["boxes"])) for row in copies}) > 1:
                for row in copies:
                    row.update(status="excluded", reason="duplicate_annotation_conflict")
            else:
                copies.sort(key=lambda r: ({"test": 0, "valid": 1, "train": 2}[r["previous_split"]], r["image"]))
                for row in copies[1:]:
                    row.update(status="excluded", reason="exact_pixel_duplicate", duplicate_of=copies[0]["image"])
        accepted = [r for r in rows if r["status"] == "keep"]
        report = dict(dataset_id="roads_bridges_cracks_full", source_images=len(rows), kept_images=len(accepted),
                      source_objects_by_format=dict(formats), kept_bbox_objects=sum(r["bbox_count"] for r in accepted),
                      exclusions=dict(Counter(r["reason"] for r in rows if r["status"] != "keep")),
                      surface_filter_applied=False, original_source=SOURCE.name, original_source_sha256=source_hash,
                      label_conversion="Class 0 cracks. Existing bbox retained geometrically; polygons converted to min/max axis-aligned bbox. Original labels preserved verbatim.")
        info = dict(dataset_id="roads_bridges_cracks_full", dataset_name="Roads and Bridges Cracks (YOLOv8 Format)",
                    original_source=SOURCE.name, original_source_sha256=source_hash, raw_images=len(rows),
                    positive_unique_images=len(accepted), surface_filter_applied=False, archive7_included=False,
                    description="Seluruh dataset **Roads and Bridges Cracks (YOLOv8 Format)** setelah validasi dan penghapusan duplikat. Semua jenis permukaan sumber dipertahankan; tidak ada filter khusus aspal. Label diseragamkan menjadi bbox deteksi YOLO, kelas 0 = cracks.")
        fields = ["image", "label", "previous_split", "status", "reason", "duplicate_of", "bbox_count", "source_formats",
                  "warnings", "width", "height", "pixel_sha256"]
        with zipfile.ZipFile(CANONICAL, "x", compression=zipfile.ZIP_STORED) as canonical:
            canonical.writestr("data.yaml", "train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n")
            for row in accepted:
                canonical.writestr(row["image"], source.read(row["image"]))
                canonical.writestr(row["label"], bbox_bytes(row["boxes"]))
                original = row["label"].replace("/labels/", "/labels_original/")
                canonical.writestr(original, source.read(row["label"]))
            for name in names:
                if PurePosixPath(name).name.lower().startswith("readme") or name == "data.yaml":
                    target_name = "data.yaml.original.txt" if name == "data.yaml" else PurePosixPath(name).name
                    canonical.writestr("source_metadata/roads_bridges/" + target_name, source.read(name))
            canonical.writestr("reports/normalization_manifest.csv", csv_bytes(rows, fields))
            canonical.writestr("reports/normalization_summary.json", json.dumps(report, indent=2))
    save_json(WORK / "dataset_info.json", info)
    print(json.dumps(report), flush=True)
    return info, rows


def main():
    if OUT.exists() or PACKAGE.exists():
        raise FileExistsError("Output penuh sudah ada; menolak menimpa")
    info, rows = normalize_source()
    config = prepare_dataset(CANONICAL, OUT, seed=42, allowed_sources={"archive6"}, dataset_info=info)
    # Verifikasi langsung terhadap ZIP asli, bukan hanya paket perantara.
    with (OUT / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    accepted = {PurePosixPath(row["image"]).name: row for row in rows if row["status"] == "keep"}
    with zipfile.ZipFile(SOURCE) as source:
        for row in manifest:
            original = accepted[row["name"]]
            if (OUT / row["image"]).read_bytes() != source.read(original["image"]):
                raise ValueError("Gambar berbeda dari sumber asli")
            raw_label = source.read(original["label"])
            label_original = OUT / row["split"] / "labels_original" / Path(row["name"]).with_suffix(".txt").name
            if label_original.read_bytes() != raw_label or (OUT / row["label"]).read_bytes() != bbox_bytes(original["boxes"]):
                raise ValueError("Label berbeda dari hasil normalisasi/sumber asli")
    save_json(OUT / "source_validation.json", dict(passed=True, original_source=SOURCE.name,
              original_source_sha256=file_sha256(SOURCE), unique_positive_images_verified=len(manifest),
              image_bytes_and_original_labels_preserved=True, surface_filter_applied=False,
              archive7_images=0, all_normalized_labels_match_conversion=True))
    print(json.dumps(package_dataset(OUT, PACKAGE)), flush=True)
    print(json.dumps(config["summary"]), flush=True)


if __name__ == "__main__":
    main()
