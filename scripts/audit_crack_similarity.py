"""Cari pasangan mirip antar split pada kandidat aspal, tanpa menghapus otomatis."""

import csv
from io import BytesIO
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".venv" / "cracks_review_work"
OUT = ROOT / "data" / "cracks_curated"


def main():
    audit = json.loads((WORK / "audit.json").read_text(encoding="utf-8"))
    with (OUT / "visual_decisions.csv").open(encoding="utf-8", newline="") as stream:
        decisions = {int(row["group_id"]): row for row in csv.DictReader(stream)}
    representatives = {}
    for row in audit:
        if row["status"] == "valid" and decisions[row["group_id"]]["category"] == "asphalt_candidate":
            representatives.setdefault(row["group_id"], row)
    items = sorted(representatives.values(), key=lambda row: row["group_id"])
    hashes, thumbnails = [], []
    with zipfile.ZipFile(ROOT / "data" / "archive (6).zip") as archive:
        for row in items:
            with Image.open(BytesIO(archive.read(row["image"]))) as image:
                gray = image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
                thumbnails.append(np.asarray(gray, dtype=np.float32))
                small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
                orientations = []
                for base in (small, np.fliplr(small)):
                    for turns in range(4):
                        coefficients = cv2.dct(np.ascontiguousarray(np.rot90(base, turns)))[:8, :8].flatten()
                        bits = coefficients > np.median(coefficients[1:])
                        bits[0] = False
                        orientations.append(sum(int(value) << index for index, value in enumerate(bits)))
                hashes.append(min(orientations))
    hashes = np.asarray(hashes, dtype=np.uint64)
    pairs = []
    for i, item in enumerate(items):
        # bit_count Python menjaga kompatibilitas dengan NumPy sebelum versi 2.
        distances = np.fromiter((int(value).bit_count() for value in (hashes[i + 1:] ^ hashes[i])), dtype=np.uint8)
        for offset in np.flatnonzero(distances <= 4):
            j = i + 1 + int(offset)
            other = items[j]
            if item["split"] == other["split"]:
                continue
            a = thumbnails[i] - thumbnails[i].mean()
            b = thumbnails[j] - thumbnails[j].mean()
            mae = min(float(np.abs(a - np.rot90(base, turns)).mean())
                      for base in (b, np.fliplr(b)) for turns in range(4))
            if mae <= 8:
                pairs.append(dict(group_a=item["group_id"], split_a=item["split"], image_a=item["image"],
                                  group_b=other["group_id"], split_b=other["split"], image_b=other["image"],
                                  phash_distance=int(distances[offset]), mean_centered_mae=round(mae, 4),
                                  action="needs_visual_review"))
    pairs.sort(key=lambda row: row["mean_centered_mae"])
    fields = ["group_a", "split_a", "image_a", "group_b", "split_b", "image_b",
              "phash_distance", "mean_centered_mae", "action"]
    with (OUT / "near_duplicate_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    report = dict(original_groups_checked=len(items), cross_split_candidates=len(pairs),
                  method="rotation/flip canonical pHash, Hamming <= 4; mean-centered 64px MAE <= 8",
                  limitation="Satu representatif per kelompok; pemeriksaan kemiripan bersifat heuristik dan tidak menjamin bebas semua kebocoran.")
    (OUT / "similarity_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(json.dumps(pairs[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
