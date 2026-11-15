"""Split ulang dataset crack YOLO berdasarkan kelompok asal dan kemiripan gambar.

Modul ini juga disertakan langsung dalam notebook Colab. Tidak menjalankan model.
"""

from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

SPLIT_NAMES = ("train", "valid", "test")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def validate_boxes(raw):
    boxes = []
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = tuple(map(float, line.split()))
        if len(values) != 5 or values[0] != 0 or not all(math.isfinite(x) for x in values):
            raise ValueError("Label harus lima kolom bbox, satu kelas: 0 = cracks")
        _, x, y, w, h = values
        if min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-8 or max(x+w/2, y+h/2) > 1+1e-8:
            raise ValueError("Bounding box tidak valid")
        boxes.append(values)
    if not boxes:
        raise ValueError("Dataset positif ini tidak menerima label kosong")
    return boxes


def image_features(raw):
    with Image.open(BytesIO(raw)) as opened:
        opened.load()
        if opened.getexif().get(274, 1) != 1:
            raise ValueError("Orientasi EXIF perlu ditinjau sebelum pemakaian bbox")
        if min(opened.size) < 32 or getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Resolusi terlalu kecil atau gambar multiframe")
        rgb = opened.convert("RGB")
    width, height = rgb.size
    pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
    gray = rgb.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    thumb = np.asarray(gray, dtype=np.float32)
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    phashes = []
    for base in (small, np.fliplr(small)):
        for turns in range(4):
            coefficients = cv2.dct(np.ascontiguousarray(np.rot90(base, turns)))[:8, :8].ravel()
            bits = coefficients > np.median(coefficients[1:])
            bits[0] = False
            phashes.append(sum(int(bit) << index for index, bit in enumerate(bits)))
    return width, height, pixel_hash, tuple(sorted(set(phashes))), thumb


def initial_group(name):
    stem = Path(name).stem
    if name.startswith("archive7__"):
        # Snapshot sering berurutan. Seluruh tanggal tetap satu kelompok,
        # kemudian kelompok tanggal dapat digabung oleh pasangan gambar mirip.
        match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", stem)
        return "archive7:date:" + "-".join(match.groups()) if match else "archive7:undated_snapshots"
    return "archive6:original:" + re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.I)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def hash_bands(value):
    # Jarak <=4 pada 64 bit menjamin minimal satu dari lima band identik.
    return [(band, (value >> shift) & ((1 << min(13, 64-shift)) - 1))
            for band, shift in enumerate(range(0, 64, 13))]


def group_images(records, hashes, thumbs, progress=print):
    union = UnionFind(len(records))
    named, exact, buckets = {}, {}, defaultdict(set)
    pairs = []
    for index, row in enumerate(records):
        for key, table in ((row["origin_group"], named), (row["pixel_sha256"], exact)):
            if key in table:
                union.union(index, table[key])
            else:
                table[key] = index
        candidates = set()
        for phash in hashes[index]:
            for band in hash_bands(phash):
                candidates.update(buckets[band])
        for other in sorted(candidates):
            distance = min((a ^ b).bit_count() for a in hashes[index] for b in hashes[other])
            if distance > 4:
                continue
            a = thumbs[index] - thumbs[index].mean()
            b = thumbs[other] - thumbs[other].mean()
            mae = min(float(np.abs(a - np.rot90(base, turns)).mean())
                      for base in (b, np.fliplr(b)) for turns in range(4))
            if mae <= 8:
                union.union(index, other)
                pairs.append(dict(image_a=records[other]["name"], image_b=row["name"],
                                  phash_distance=distance, centered_mae=round(mae, 5)))
        for phash in hashes[index]:
            for band in hash_bands(phash):
                buckets[band].add(index)
        if progress and (index+1) % 500 == 0:
            progress(f"Grouping: {index+1}/{len(records)} foto", flush=True)
    components = defaultdict(list)
    for index in range(len(records)):
        components[union.find(index)].append(index)
    groups = {}
    for indices in components.values():
        group_id = hashlib.sha256("\n".join(sorted(records[i]["name"] for i in indices)).encode()).hexdigest()[:20]
        groups[group_id] = indices
        for index in indices:
            records[index]["group_id"] = group_id
    return groups, pairs


def allocate_groups(records, groups, seed=42, ratios=(0.8, 0.1, 0.1)):
    if len(ratios) != 3 or min(ratios) <= 0 or not math.isclose(sum(ratios), 1, abs_tol=1e-8):
        raise ValueError("Isi tiga rasio positif dengan jumlah 1")
    if len(groups) < 3:
        raise ValueError("Kelompok independen terlalu sedikit untuk tiga split")
    source_names = sorted({row["source"] for row in records})
    source_totals = Counter(row["source"] for row in records)
    # Dimensi pertama jumlah seluruh foto; berikutnya jumlah per sumber.
    weights = [1.0] + [0.35] * len(source_names)
    totals = [len(records)] + [source_totals[s] for s in source_names]
    targets = [[ratio * total for total in totals] for ratio in ratios]
    sizes = {}
    for key, indices in groups.items():
        sources = Counter(records[i]["source"] for i in indices)
        sizes[key] = [len(indices)] + [sources[s] for s in source_names]
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: -sizes[key][0])
    counts = [[0] * len(totals) for _ in ratios]

    def score(split, values):
        return sum(weight * (value-target)**2 / max(1, target)
                   for weight, value, target in zip(weights, values, targets[split]))

    assignments = {}
    for key in keys:
        size = sizes[key]
        choices = list(range(3))
        rng.shuffle(choices)
        chosen = min(choices, key=lambda s: score(s, [a+b for a, b in zip(counts[s], size)]) - score(s, counts[s]))
        assignments[key] = chosen
        counts[chosen] = [a+b for a, b in zip(counts[chosen], size)]
    # Perbaiki rasio dengan memindahkan kelompok utuh; sumber ikut diseimbangkan.
    for _ in range(12):
        improved = False
        rng.shuffle(keys)
        for key in keys:
            old = assignments[key]
            size = sizes[key]
            if counts[old][0] == size[0]:
                continue
            alternatives = []
            for new in range(3):
                if new == old:
                    continue
                less = [a-b for a, b in zip(counts[old], size)]
                more = [a+b for a, b in zip(counts[new], size)]
                delta = score(old, less)+score(new, more)-score(old, counts[old])-score(new, counts[new])
                alternatives.append((delta, new, less, more))
            delta, new, less, more = min(alternatives)
            if delta < -1e-9:
                assignments[key] = new
                counts[old], counts[new] = less, more
                improved = True
        if not improved:
            break
    if min(row[0] for row in counts) <= 0:
        raise ValueError("Salah satu split kosong; butuh lebih banyak kelompok independen")
    for key, indices in groups.items():
        for index in indices:
            records[index]["split"] = SPLIT_NAMES[assignments[key]]
    return assignments


def scan_zip(source, progress=print):
    records, hashes, thumbs = [], [], []
    with zipfile.ZipFile(source) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Anggota ZIP duplikat")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError("Path ZIP tidak aman")
        yaml_text = archive.read("data.yaml").decode("utf-8-sig")
        if not re.search(r"names:\s*(?:\['cracks'\]|\[cracks\])", yaml_text):
            raise ValueError("Pakai ZIP crack v2 dengan satu kelas cracks")
        images = sorted(name for name in names if re.fullmatch(r"(?:train|valid|val|test)/images/[^/]+\.(?:jpg|jpeg|png)", name, re.I))
        if not images:
            raise ValueError("ZIP tidak memuat gambar train/valid/test")
        seen_names, seen_pixels = set(), {}
        for index, image_path in enumerate(images):
            p = PurePosixPath(image_path)
            name = p.name
            if name in seen_names:
                raise ValueError("Nama file sama di beberapa split")
            seen_names.add(name)
            split = p.parts[0]
            label_path = f"{split}/labels/{p.with_suffix('.txt').name}"
            raw_label = archive.read(label_path)
            boxes = validate_boxes(raw_label)
            raw = archive.read(image_path)
            width, height, pixel_hash, phashes, thumb = image_features(raw)
            if pixel_hash in seen_pixels:
                # Sumber v2 sudah dideduplikasi. Jangan memilih label konflik diam-diam.
                raise ValueError("Duplikat piksel masih ada: " + name + " / " + seen_pixels[pixel_hash])
            seen_pixels[pixel_hash] = name
            original = f"{split}/labels_original/{p.with_suffix('.txt').name}"
            records.append(dict(name=name, source_image=image_path, source_label=label_path,
                                source_original_label=original if original in names else "", previous_split=split,
                                source="archive7" if name.startswith("archive7__") else "archive6",
                                origin_group=initial_group(name), width=width, height=height,
                                boxes=len(boxes), pixel_sha256=pixel_hash,
                                image_sha256=hashlib.sha256(raw).hexdigest(), label_sha256=hashlib.sha256(raw_label).hexdigest()))
            hashes.append(phashes)
            thumbs.append(thumb)
            if progress and (index+1) % 500 == 0:
                progress(f"Audit: {index+1}/{len(images)} foto", flush=True)
    return records, hashes, thumbs


def dataset_fingerprint(records):
    lines = ["|".join(str(row[k]) for k in ("name", "split", "group_id", "image_sha256", "label_sha256"))
             for row in sorted(records, key=lambda r: r["name"])]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def prepare_dataset(source_zip, destination, seed=42, ratios=(0.8, 0.1, 0.1), progress=print, allowed_sources=None, dataset_info=None):
    source_zip, destination = Path(source_zip), Path(destination)
    if destination.exists():
        raise FileExistsError("Folder output sudah ada; gunakan folder baru atau cache yang sudah diverifikasi")
    source_hash = file_sha256(source_zip)
    dataset_info = dict(dataset_info or {})
    records, hashes, thumbs = scan_zip(source_zip, progress)
    sources = sorted({row["source"] for row in records})
    if allowed_sources is not None and set(sources) != set(allowed_sources):
        raise ValueError(f"Sumber dataset {sources} tidak sesuai batasan {sorted(allowed_sources)}")
    groups, pairs = group_images(records, hashes, thumbs, progress)
    allocate_groups(records, groups, seed, ratios)
    fingerprint = dataset_fingerprint(records)
    for split in SPLIT_NAMES:
        for kind in ("images", "labels", "labels_original"):
            (destination / split / kind).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        for row in records:
            new_image = f"{row['split']}/images/{row['name']}"
            label_name = PurePosixPath(row["name"]).with_suffix(".txt").name
            new_label = f"{row['split']}/labels/{label_name}"
            (destination / new_image).write_bytes(archive.read(row["source_image"]))
            (destination / new_label).write_bytes(archive.read(row["source_label"]))
            if row["source_original_label"]:
                (destination / row["split"] / "labels_original" / label_name).write_bytes(archive.read(row["source_original_label"]))
            row.update(image=new_image, label=new_label)
        # Salin atribusi serta laporan kurasi v2 tanpa menjalankan kode sumber.
        for member in archive.namelist():
            if member.startswith(("source_metadata/", "reports/")) and not member.endswith("/"):
                target = destination / "input_provenance" / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    (destination / "data.yaml").write_text("train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n", encoding="utf-8")
    fields = ["name", "source", "source_image", "source_label", "source_original_label", "previous_split",
              "origin_group", "group_id", "split", "image", "label", "width", "height", "boxes",
              "pixel_sha256", "image_sha256", "label_sha256"]
    save_csv(destination / "split_manifest.csv", records, fields)
    group_rows = [dict(group_id=key, images=len(indices), split=records[indices[0]]["split"],
                       sources=";".join(sorted({records[i]["source"] for i in indices})),
                       origin_groups=";".join(sorted({records[i]["origin_group"] for i in indices})))
                  for key, indices in sorted(groups.items())]
    save_csv(destination / "group_manifest.csv", group_rows, ["group_id", "images", "split", "sources", "origin_groups"])
    save_csv(destination / "similarity_pairs.csv", pairs, ["image_a", "image_b", "phash_distance", "centered_mae"])
    summary = []
    for split, target in zip(SPLIT_NAMES, ratios):
        selected = [r for r in records if r["split"] == split]
        summary.append(dict(split=split, images=len(selected), ratio=round(len(selected)/len(records), 6),
                            target_ratio=target, boxes=sum(r["boxes"] for r in selected),
                            groups=len({r["group_id"] for r in selected}),
                            archive6=sum(r["source"] == "archive6" for r in selected),
                            archive7=sum(r["source"] == "archive7" for r in selected)))
    save_csv(destination / "split_summary.csv", summary, list(summary[0]))
    config = dict(seed=seed, target_ratios=dict(zip(SPLIT_NAMES, ratios)), source_zip=source_zip.name,
                  source_sha256=source_hash, fingerprint=fingerprint, total_images=len(records),
                  dataset_sources=sources, source_image_counts=dict(Counter(row["source"] for row in records)),
                  total_groups=len(groups), similarity_pairs=len(pairs),
                  group_method="Roboflow original basename; archive7 whole capture-date/undated snapshots; union all near-duplicate pairs",
                  similarity_method="8 rotation/flip pHashes, Hamming <=4; mean-centered 64px MAE <=8",
                  split_method="Seeded weighted group allocation balanced by image count and source; no group divided",
                  limitation="Capture dates are inferred from filenames, not verified recording/location IDs. Similarity detection is heuristic. Existing archive6 augmentations remain together, including when a whole group is held out.",
                  summary=summary)
    if dataset_info:
        config["dataset_id"] = dataset_info["dataset_id"]
        config["original_source_sha256"] = dataset_info.get("original_source_sha256", "")
        save_json(destination / "dataset_info.json", dataset_info)
    save_json(destination / "split_config.json", config)
    archive6_only = sources == ["archive6"]
    photo_count = f"{len(records):,}".replace(",", ".")
    source_description = (f"Hanya kandidat retakan pada aspal jalan dari **archive 6** yang sudah lolos kurasi ({photo_count} foto). "
                          "Ini subset hasil kurasi, bukan seluruh foto mentah archive 6."
                          if archive6_only else "Gabungan crack archive 6 dan archive 7 yang sudah dikurasi (v2).")
    source_description = dataset_info.get("description", source_description)
    sequence_note = ("" if archive6_only else " Archive 7 dikelompokkan konservatif per tanggal dari nama file, dan snapshot tanpa tanggal tetap satu kelompok.")
    archive7_column = "" if archive6_only else " | Archive 7"
    archive7_divider = "" if archive6_only else " | ---:"
    table = "\n".join(f"| {r['split']} | {r['images']} | {r['ratio']:.1%} | {r['archive6']}" +
                      ("" if archive6_only else f" | {r['archive7']}") + " |" for r in summary)
    readme = f"""# Crack YOLO: split kelompok 80/10/10

Sumber: {source_description} Seluruh {len(records)} foto dibagi ulang bersama; split sebelumnya tidak dipertahankan. Satu kelas `0 = cracks`.

| Split | Gambar | Rasio aktual | Archive 6{archive7_column} |
| --- | ---: | ---: | ---:{archive7_divider} |
{table}

Seed: {seed}. Target train/valid/test: {ratios}. Rasio aktual mengikuti ukuran kelompok utuh. Skema A/B memakai split yang sama. Mengulangi seed dan sumber identik menghasilkan pembagian identik; urutan batch training tetap dapat diacak.

Kelompok: nama foto asal Roboflow (seluruh augmentasi tetap bersama).{sequence_note} Kelompok juga digabung jika ditemukan pasangan mirip dengan pHash delapan rotasi/flip, Hamming <=4 dan mean-centered thumbnail MAE <=8. Semua {len(pairs)} pasangan terdeteksi berada dalam split yang sama. Ini mengurangi risiko kebocoran; pemeriksaan kemiripan tidak menjamin independensi seluruh adegan/lokasi/rekaman.

Archive 6 sudah memiliki augmentasi offline; versi augmentasi dari satu foto tetap satu kelompok, termasuk bila masuk validasi/test. Metrik per gambar tidak menyatakan jumlah adegan independen; jumlah kelompok tersedia dalam manifest. Penilaian material aspal dan semantik anotasi tetap mengikuti batasan kurasi sumber. Dataset ini berisi foto positif; background tanpa retak belum diverifikasi.

Foto, bbox training, dan label asli disalin tanpa perubahan byte. `labels_original` masih memakai format/kelas sumber dan bukan label training. `input_provenance` memuat metadata/atribusi sumber serta laporan kurasi sebelumnya (laporan tersebut menjelaskan split lama, bukan split baru).

Gunakan `data.yaml` pada root untuk YOLO detect. Notebook baru memverifikasi hash dan manifest sebelum training, kemudian memakai split ini langsung. Tidak menjalankan training atau mengubah bobot model saat menyiapkan dataset ini.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    validation = verify_dataset(destination, expected_source_hash=source_hash)
    if progress:
        progress(json.dumps(dict(summary=summary, validation=validation)), flush=True)
    return config


def verify_dataset(destination, expected_source_hash=None):
    destination = Path(destination)
    config = json.loads((destination / "split_config.json").read_text(encoding="utf-8"))
    if expected_source_hash and config["source_sha256"] != expected_source_hash:
        raise ValueError("Cache memakai ZIP sumber berbeda")
    with (destination / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != config["total_images"] or dataset_fingerprint(records) != config["fingerprint"]:
        raise ValueError("Fingerprint manifest berubah")
    groups, pixels, by_name = defaultdict(set), defaultdict(set), {}
    for row in records:
        image = destination / row["image"]
        label = destination / row["label"]
        for relative in (row["image"], row["label"]):
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                raise ValueError("Path manifest tidak aman")
        if row["split"] not in SPLIT_NAMES or row["image"] != f"{row['split']}/images/{row['name']}" or row["label"] != f"{row['split']}/labels/{Path(row['name']).with_suffix('.txt').name}":
            raise ValueError("Path gambar/label tidak sesuai split")
        if file_sha256(image) != row["image_sha256"] or file_sha256(label) != row["label_sha256"]:
            raise ValueError("Isi gambar/label berubah: " + row["name"])
        if len(validate_boxes(label.read_bytes())) != int(row["boxes"]):
            raise ValueError("Jumlah bbox berubah")
        groups[row["group_id"]].add(row["split"])
        pixels[row["pixel_sha256"]].add(row["split"])
        if row["name"] in by_name:
            raise ValueError("Nama gambar duplikat")
        by_name[row["name"]] = row
    if any(len(splits) > 1 for splits in [*groups.values(), *pixels.values()]):
        raise ValueError("Kelompok/hash piksel bocor antar split")
    for split in SPLIT_NAMES:
        selected = [r for r in records if r["split"] == split]
        for kind, field in (("images", "image"), ("labels", "label")):
            actual = {p.relative_to(destination).as_posix() for p in (destination / split / kind).iterdir() if p.is_file()}
            if actual != {r[field] for r in selected}:
                raise ValueError("Daftar file tidak sesuai manifest")
    pair_count = 0
    with (destination / "similarity_pairs.csv").open(encoding="utf-8", newline="") as stream:
        for pair in csv.DictReader(stream):
            if by_name[pair["image_a"]]["split"] != by_name[pair["image_b"]]["split"]:
                raise ValueError("Pasangan mirip melintasi split")
            pair_count += 1
    if pair_count != config["similarity_pairs"]:
        raise ValueError("Jumlah pasangan kemiripan berbeda")
    report = dict(passed=True, images=len(records), groups=len(groups), known_cross_split_groups=0,
                  known_cross_split_pixel_hashes=0, known_cross_split_similar_pairs=0,
                  images_and_labels_hash_verified=True, all_labels_class_zero_bbox=True,
                  caveat="Pemeriksaan metadata dan kemiripan bersifat heuristik; bukan jaminan independensi lokasi/rekaman.")
    save_json(destination / "validation.json", report)
    return report


def package_dataset(destination, zip_path):
    destination, zip_path = Path(destination), Path(zip_path)
    if zip_path.exists():
        raise FileExistsError("Paket output sudah ada")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError("CRC ZIP gagal: " + bad)
    return dict(zip=zip_path.name, bytes=zip_path.stat().st_size, sha256=file_sha256(zip_path), crc_passed=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.source, args.output, seed=args.seed)
    if args.zip:
        print(json.dumps(package_dataset(args.output, args.zip)), flush=True)
