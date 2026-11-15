"""Buat notebook Colab crack YOLO26 dengan pipeline split yang bisa direproduksi."""

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deteksi_crack_YOLO26s_skemaAB_GROUP_SPLIT.ipynb"


def main(archive6_only=False, roads_bridges_full=False, model="yolo26s.pt"):
    cells = []
    model_names = {"yolo26s.pt": "YOLO26s", "yolo11s.pt": "YOLO11s", "yolov8s.pt": "YOLOv8s"}
    if model not in model_names:
        raise ValueError("Model tidak didukung generator notebook")
    if model != "yolo26s.pt" and not roads_bridges_full:
        raise ValueError("Varian YOLO11/YOLOv8 memakai mode --roads-bridges-full")
    model_name = model_names[model]
    output = (ROOT / f"deteksi_crack_{model_name}_ROADS_BRIDGES.ipynb" if roads_bridges_full else
              ROOT / "deteksi_crack_YOLO26s_ARCHIVE6.ipynb" if archive6_only else OUTPUT)

    def add(kind, text):
        text = textwrap.dedent(text).strip() + "\n"
        cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True), id=f"crack-cell-{len(cells):02d}")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
    # Deteksi Retakan Jalan · YOLO26s · Skema A/B · Split Kelompok

    Notebook baru khusus **satu kelas `cracks`**, memakai gabungan hasil kurasi archive 6 + 7.
    Seluruh 2.603 foto telah **dibagi ulang**, dengan target **80% train, 10% validasi, 10% test**, seed 42.
    Foto asal, versi augmentasinya, rangkaian snapshot per tanggal, dan pasangan gambar mirip tetap dalam satu split.
    Rasio aktual mengikuti ukuran kelompok utuh; tabel jumlah aktual muncul sebelum training.

    1. Aktifkan runtime **GPU** di Colab.
    2. Simpan `cracks_yolo_group_split_80_10_10.zip` di `MyDrive/deteksi-jalan/data/`.
    3. Jalankan sel secara berurutan. Default hanya **Skema A**; ubah `SKEMA_LIST` untuk menjalankan A dan B.
    4. Model dan hasil disimpan ke Drive. Menyiapkan notebook/dataset ini belum menjalankan training.

    **Split tersimpan vs acak ulang:** default memakai split dalam ZIP. Untuk mengacak ulang, set `RESPLIT=True`, ganti `SPLIT_SEED` dan `RUN_SUFFIX`. Kedua skema tetap memakai split yang sama. Seed/sumber/algoritma identik menghasilkan pembagian identik.

    Ini **fine-tuning `yolo26s.pt`**. Tanggal snapshot bukan ID rekaman/lokasi yang terverifikasi, dan pencarian kemiripan bersifat heuristik. Risiko kebocoran berkurang, tetapi independensi seluruh adegan tidak dapat dijamin. Archive 6 sudah mengandung augmentasi offline; seluruh variannya tetap satu kelompok, termasuk jika kelompok masuk validasi/test.
    """)
    add("code", """
    # 0 · Pustaka. PyTorch/CUDA mengikuti runtime Colab yang tersedia.
    import importlib.util
    from importlib.metadata import PackageNotFoundError, version
    import subprocess
    import sys

    ULTRALYTICS_VERSION = '8.4.202'
    try:
        installed_yolo = version('ultralytics')
    except PackageNotFoundError:
        installed_yolo = None
    packages = []
    if installed_yolo != ULTRALYTICS_VERSION:
        packages.append(f'ultralytics=={ULTRALYTICS_VERSION}')
    for module, package in [
        ('pandas', 'pandas'), ('matplotlib', 'matplotlib'), ('yaml', 'PyYAML'),
        ('cv2', 'opencv-python'), ('PIL', 'Pillow'), ('tqdm', 'tqdm'),
        ('albumentations', 'albumentations>=2,<3'),
    ]:
        if importlib.util.find_spec(module) is None:
            packages.append(package)
    if packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *packages])
    print('Pustaka siap. Jika runtime meminta restart, restart lalu jalankan lagi dari awal.')
    """)
    add("code", """
    # 1 · Import dan verifikasi GPU.
    from pathlib import Path, PurePosixPath
    import gc
    import hashlib
    import json
    import random
    import shutil
    import tempfile
    import zipfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from IPython.display import display
    from tqdm.auto import tqdm
    import torch
    import ultralytics
    from ultralytics import YOLO
    import yaml

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError('Ultralytics lama sudah diimpor. Restart runtime dan jalankan dari awal.')
    if not torch.cuda.is_available():
        raise RuntimeError('Pilih runtime GPU di Colab sebelum menjalankan training.')
    DEVICE = 0
    print('GPU:', torch.cuda.get_device_name(DEVICE))
    print('PyTorch:', torch.__version__, '| Ultralytics:', ultralytics.__version__)
    """)
    add("code", """
    # 2 · Google Drive, sumber data, split, dan konfigurasi training.
    from google.colab import drive
    drive.mount('/content/drive')

    DRIVE_PROJECT_DIR = Path('/content/drive/MyDrive/deteksi-jalan')
    DATA_ZIP = DRIVE_PROJECT_DIR / 'data' / 'cracks_yolo_group_split_80_10_10.zip'
    WORK_DIR = Path('/content/deteksi-jalan/cracks_group_split')
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    RESPLIT = False       # False = pakai split tersimpan. True = bagi ulang semua foto.
    SPLIT_SEED = 42       # Ganti seed dan RESPLIT=True untuk pembagian baru.
    SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train, valid, test; jumlah harus 1.
    TRAIN_SEED = 42       # Seed training terpisah dari seed pembagian data.

    MODEL = 'yolo26s.pt'
    CLASS_NAMES = ['cracks']
    SKEMA_LIST = ['A']    # Ganti menjadi ['A', 'B'] untuk menjalankan dua skema.
    RUN_SUFFIX = '01'     # Ganti untuk percobaan baru; hasil lama tidak ditimpa.
    IMG_SIZE = 640
    BATCH_SIZE = 8
    EPOCHS = 45
    PATIENCE = 10
    WORKERS = 2
    CLOSE_MOSAIC = 5
    PREDICT_CONF = 0.25
    RUNS_ROOT = DRIVE_PROJECT_DIR / 'hasil' / 'yolo_cracks_group_split'

    if not DATA_ZIP.is_file():
        raise FileNotFoundError(f'Simpan ZIP dataset di {DATA_ZIP}, atau ubah DATA_ZIP sesuai lokasi file.')
    if not SKEMA_LIST or len(set(SKEMA_LIST)) != len(SKEMA_LIST) or not set(SKEMA_LIST) <= {'A', 'B'}:
        raise ValueError("SKEMA_LIST harus ['A'], ['B'], atau ['A', 'B'].")
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', RUN_SUFFIX):
        raise ValueError('RUN_SUFFIX hanya boleh berisi huruf, angka, garis bawah, atau tanda hubung.')
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    """)
    add("markdown", """
    ## 3. Fungsi validasi dan pengacakan kelompok

    Sel berikut menyertakan seluruh fungsi persiapan data agar notebook dapat berjalan tanpa mengunggah skrip Python tambahan. Fungsi menolak label/gambar yang berubah atau tidak valid; jumlah foto tidak dikurangi diam-diam. Mode `RESPLIT=True` menggabungkan kembali semua split sebelum pembagian baru, lalu menyimpan manifest dan laporan verifikasi.
    """)
    module = (ROOT / "scripts/prepare_cracks_group_split.py").read_text(encoding="utf-8")
    module = module.split('\nif __name__ == "__main__":', 1)[0]
    add("code", module)
    add("code", """
    # 4 · Baca split tersimpan, atau acak ulang kelompok dengan seed yang dipilih.
    SOURCE_ZIP_SHA256 = file_sha256(DATA_ZIP)
    if RESPLIT:
        signature = hashlib.sha256(json.dumps(dict(source=SOURCE_ZIP_SHA256, seed=SPLIT_SEED,
                                                    ratios=SPLIT_RATIOS), sort_keys=True).encode()).hexdigest()[:16]
        YOLO_DATA_DIR = WORK_DIR / f'resplit_{signature}'
        if not YOLO_DATA_DIR.exists():
            with tempfile.TemporaryDirectory(prefix='prepare_', dir=WORK_DIR) as tmp:
                staged = Path(tmp) / 'dataset'
                prepare_dataset(DATA_ZIP, staged, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)
                staged.replace(YOLO_DATA_DIR)
        validation = verify_dataset(YOLO_DATA_DIR, expected_source_hash=SOURCE_ZIP_SHA256)
    else:
        YOLO_DATA_DIR = WORK_DIR / f'stored_{SOURCE_ZIP_SHA256[:16]}'
        if not YOLO_DATA_DIR.exists():
            with tempfile.TemporaryDirectory(prefix='extract_', dir=WORK_DIR) as tmp:
                staged = Path(tmp) / 'dataset'
                staged.mkdir()
                with zipfile.ZipFile(DATA_ZIP) as archive:
                    names = [x.filename for x in archive.infolist()]
                    if len(names) != len(set(names)):
                        raise ValueError('Anggota ZIP duplikat')
                    for name in names:
                        path = PurePosixPath(name)
                        if path.is_absolute() or '..' in path.parts or chr(92) in name or ':' in name:
                            raise ValueError('Path ZIP tidak aman')
                    required = {'data.yaml', 'split_manifest.csv', 'split_config.json', 'similarity_pairs.csv'}
                    if not required <= set(names):
                        raise ValueError('ZIP belum memiliki manifest split kelompok. Gunakan ZIP baru, atau RESPLIT=True untuk ZIP v2.')
                    archive.extractall(staged)
                verify_dataset(staged)
                staged.replace(YOLO_DATA_DIR)
        validation = verify_dataset(YOLO_DATA_DIR)

    SPLIT_CONFIG = json.loads((YOLO_DATA_DIR / 'split_config.json').read_text(encoding='utf-8'))
    stored_ratios = tuple(SPLIT_CONFIG['target_ratios'][key] for key in SPLIT_NAMES)
    if SPLIT_CONFIG['seed'] != SPLIT_SEED or stored_ratios != tuple(SPLIT_RATIOS):
        raise ValueError('Seed/rasio tidak cocok dengan split tersimpan. Aktifkan RESPLIT=True untuk membaginya ulang.')
    config = yaml.safe_load((YOLO_DATA_DIR / 'data.yaml').read_text(encoding='utf-8'))
    if config.get('names') != CLASS_NAMES or config.get('nc') != 1:
        raise ValueError('Dataset harus satu kelas cracks')

    df = pd.read_csv(YOLO_DATA_DIR / 'split_manifest.csv', keep_default_na=False)
    df['source_image_manifest'] = df['image']
    df['source_label_manifest'] = df['label']
    df['image'] = df['image'].map(lambda value: str((YOLO_DATA_DIR / value).resolve()))
    df['label'] = df['label'].map(lambda value: str((YOLO_DATA_DIR / value).resolve()))
    DATASET_FINGERPRINT = SPLIT_CONFIG['fingerprint']
    DATASET = 'cracks_archive6_archive7_curated'
    SPLIT = 'group_random_80_10_10'
    X_train, X_val, X_test = [df.loc[df['split'].eq(key), 'image'].tolist() for key in SPLIT_NAMES]

    # YAML runtime memakai path absolut lokal; file YAML dalam ZIP tetap portable.
    runtime_config = dict(path=str(YOLO_DATA_DIR.resolve()), train='train/images', val='valid/images',
                          test='test/images', nc=1, names=CLASS_NAMES)
    runtime_yaml = WORK_DIR / f'runtime_{DATASET_FINGERPRINT[:16]}.yaml'
    runtime_yaml.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding='utf-8')
    DATA = str(runtime_yaml)
    split_table = pd.DataFrame(SPLIT_CONFIG['summary'])
    class_distribution = pd.DataFrame({key: [int(df.loc[df['split'].eq(key), 'boxes'].sum()),
                                            int(df['split'].eq(key).sum())] for key in SPLIT_NAMES},
                                      index=['cracks_bbox', 'Jumlah gambar'])
    print('Validasi dataset:', validation)
    display(split_table)
    display(class_distribution)
    print('Train / valid / test:', len(X_train), len(X_val), len(X_test))
    print('Fingerprint:', DATASET_FINGERPRINT)

    # Simpan identitas split ke Drive untuk melanjutkan eksperimen dari runtime baru.
    SPLIT_REPORT_DIR = DRIVE_PROJECT_DIR / 'data' / 'split_reports' / DATASET_FINGERPRINT
    SPLIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['split_config.json', 'split_manifest.csv', 'group_manifest.csv',
                 'split_summary.csv', 'similarity_pairs.csv', 'validation.json', 'README.md']:
        shutil.copy2(YOLO_DATA_DIR / name, SPLIT_REPORT_DIR / name)
    print('Laporan split tersimpan di', SPLIT_REPORT_DIR)
    """)
    add("code", """
    # 5 · Lihat contoh bbox dari setiap split sebelum training.
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for row_index, split in enumerate(SPLIT_NAMES):
        sample = df.loc[df['split'].eq(split)].sample(n=min(3, int(df['split'].eq(split).sum())), random_state=SPLIT_SEED)
        for ax in axes[row_index]:
            ax.axis('off')
        for ax, record in zip(axes[row_index], sample.itertuples(index=False)):
            with Image.open(record.image) as opened:
                image = opened.convert('RGB')
            ax.imshow(image)
            for _, x, y, w, h in validate_boxes(Path(record.label).read_bytes()):
                ax.add_patch(Rectangle(((x-w/2)*image.width, (y-h/2)*image.height), w*image.width,
                                       h*image.height, fill=False, edgecolor='cyan', linewidth=1.4))
            ax.set_title(f'{split} | {record.source} | {record.boxes} bbox')
    plt.tight_layout()
    plt.show()
    """)
    add("markdown", """
    ## 6. Training YOLO26s

    **Skema A:** augmentasi bawaan YOLO. **Skema B:** ditambah perubahan kecerahan/kontras, silau, dan bayangan. Augmentasi online hanya untuk train. Validasi memilih checkpoint `best.pt`; metrik test dilaporkan terpisah. Pemilihan skema untuk inferensi memakai skor **validasi**, bukan skor test.

    Split/fingerprint yang sama dipakai oleh kedua skema. Run yang selesai dilewati; run terputus dilanjutkan dari `last.pt` setelah konfigurasi diperiksa. Ganti `RUN_SUFFIX` untuk eksperimen baru. Resume memerlukan dataset serta path runtime yang sama; mode sumber, nama ZIP, dan konfigurasi split sebaiknya dipertahankan saat pindah runtime.
    """)
    add("code", """
    # 6 · Lokasi run serta identitas konfigurasi.
    RUN_NAMES = {skema: f'{Path(MODEL).stem}_{skema}_cracks_s{SPLIT_SEED}_{DATASET_FINGERPRINT[:8]}_{RUN_SUFFIX}'
                 for skema in SKEMA_LIST}

    def run_paths(skema):
        folder = RUNS_ROOT / RUN_NAMES[skema]
        return folder, folder / 'weights' / 'best.pt', folder / 'weights' / 'last.pt'

    def run_done(skema):
        folder, best, last = run_paths(skema)
        return all(path.is_file() for path in [best, last, folder / 'metrics.csv', folder / 'predictions_test.csv'])

    def run_config(skema):
        return dict(model=MODEL, classes=CLASS_NAMES, skema=skema, imgsz=IMG_SIZE, batch=BATCH_SIZE,
                    epochs=EPOCHS, patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, optimizer='auto',
                    train_seed=TRAIN_SEED, split_seed=SPLIT_SEED, ratios=list(SPLIT_RATIOS),
                    split=SPLIT, dataset_fingerprint=DATASET_FINGERPRINT,
                    ultralytics=ultralytics.__version__)

    def prepare_run(skema):
        folder, best, last = run_paths(skema)
        path = folder / 'config.json'
        config = run_config(skema)
        if folder.exists():
            if not path.is_file() or json.loads(path.read_text(encoding='utf-8')) != config:
                raise RuntimeError(f'{folder} memakai konfigurasi lain. Ganti RUN_SUFFIX untuk run baru.')
            return
        folder.mkdir(parents=True, exist_ok=False)
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        for name in ['split_config.json', 'split_manifest.csv', 'group_manifest.csv',
                     'split_summary.csv', 'similarity_pairs.csv', 'validation.json']:
            shutil.copy2(YOLO_DATA_DIR / name, folder / name)

    for skema in SKEMA_LIST:
        folder, best, last = run_paths(skema)
        print(skema, RUN_NAMES[skema], ':', 'selesai' if run_done(skema) else ('akan resume' if last.is_file() else 'baru'))
    """)
    add("code", """
    # 7 · Augmentasi, training, riwayat, dan evaluasi.
    def build_augmentations(skema):
        if skema == 'A':
            return None
        import albumentations as A
        return [
            A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5)),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1)),
            ], p=0.5),
        ]

    def train_run(skema):
        folder, best, last = run_paths(skema)
        if last.is_file():
            model = YOLO(str(last))
            if model.ckpt.get('epoch', -1) >= 0:
                print('Melanjutkan training dari', last)
                model.train(resume=True, data=DATA, device=DEVICE)
            else:
                print('Checkpoint selesai; melanjutkan ke evaluasi.')
        else:
            model = YOLO(MODEL)
            model.info(detailed=False, verbose=True)
            options = dict(data=DATA, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE,
                           device=DEVICE, workers=WORKERS, seed=TRAIN_SEED, deterministic=True,
                           optimizer='auto', patience=PATIENCE, close_mosaic=CLOSE_MOSAIC,
                           save=True, val=True, plots=True, project=str(RUNS_ROOT),
                           name=RUN_NAMES[skema], exist_ok=True)
            augmentations = build_augmentations(skema)
            if augmentations is not None:
                options['augmentations'] = augmentations
            print('Training skema', skema, '| split seed', SPLIT_SEED, '| train seed', TRAIN_SEED)
            model.train(**options)
        del model
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError('Checkpoint training belum lengkap')
        print('Best model:', best)
        print('Last checkpoint:', last)

    def show_training(skema):
        folder, best, last = run_paths(skema)
        history = pd.read_csv(folder / 'results.csv')
        history.columns = history.columns.str.strip()
        display(history.tail())
        graph = folder / 'results.png'
        if graph.is_file():
            with Image.open(graph) as image:
                display(image.copy())

    def evaluate_run(skema):
        folder, best, last = run_paths(skema)
        model = YOLO(str(best))
        results = {}
        for split in ['val', 'test']:
            metric = model.val(data=DATA, split=split, imgsz=IMG_SIZE, batch=BATCH_SIZE,
                               workers=WORKERS, device=DEVICE, conf=0.001, iou=0.7,
                               plots=True, project=str(folder), name=f'eval_{split}', exist_ok=True)
            results[split] = dict(precision=float(metric.box.mp), recall=float(metric.box.mr),
                                  mAP50=float(metric.box.map50), mAP50_95=float(metric.box.map))
        row = dict(model=Path(MODEL).stem, skema=skema, checkpoint='best.pt',
                   jumlah_training=len(X_train), jumlah_validation=len(X_val), jumlah_testing=len(X_test),
                   split_seed=SPLIT_SEED, dataset_fingerprint=DATASET_FINGERPRINT,
                   **{f'{split}_{name}': value for split, values in results.items() for name, value in values.items()})
        metrics = pd.DataFrame([row])
        metrics.to_csv(folder / 'metrics.csv', index=False)
        display(metrics.round(5))

        predictions = []
        stream = model.predict(source=str(YOLO_DATA_DIR / 'test' / 'images'), imgsz=IMG_SIZE,
                               conf=PREDICT_CONF, iou=0.7, device=DEVICE, half=False, verbose=False, stream=True)
        for result in tqdm(stream, total=len(X_test), desc='Prediksi test', unit='foto'):
            if result.boxes is None or len(result.boxes) == 0:
                predictions.append(dict(image=result.path, detected=False, class_id=None, kelas=None,
                                        confidence=None, x1=None, y1=None, x2=None, y2=None))
                continue
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                       result.boxes.cls.cpu().tolist()):
                predictions.append(dict(image=result.path, detected=True, class_id=int(cls), kelas=CLASS_NAMES[int(cls)],
                                        confidence=score, **dict(zip(['x1', 'y1', 'x2', 'y2'], box))))
        pd.DataFrame(predictions).to_csv(folder / 'predictions_test.csv', index=False)
        df.loc[df['split'].eq('test')].to_csv(folder / 'ground_truth_test_manifest.csv', index=False)
        del model
        print('Metrik, prediksi bbox, dan ground truth test tersimpan di', folder)
    """)
    add("code", """
    # 8 · Jalankan skema. Sel ini benar-benar memulai training GPU.
    for skema in SKEMA_LIST:
        print('Skema', skema, '|', RUN_NAMES[skema])
        prepare_run(skema)
        if run_done(skema):
            print('Run sudah selesai, dilewati.')
            continue
        train_run(skema)
        show_training(skema)
        evaluate_run(skema)
        gc.collect()
        torch.cuda.empty_cache()
    print('Semua skema terpilih selesai.')
    """)
    add("markdown", """
    ## 9. Ringkasan dan tes foto sendiri

    mAP50 dan mAP50-95 adalah metrik deteksi bbox. Confidence inferensi default 0,25; evaluasi mAP memakai 0,001. Metrik test tidak dipakai untuk memilih skema. Semua gambar dataset ini positif crack; performa pada jalan tanpa crack perlu diuji menggunakan foto negatif terpisah yang dianotasi/diverifikasi.
    """)
    add("code", """
    summary = pd.concat([pd.read_csv(run_paths(skema)[0] / 'metrics.csv') for skema in SKEMA_LIST], ignore_index=True)
    summary.insert(0, 'run', [RUN_NAMES[skema] for skema in SKEMA_LIST])
    summary_path = RUNS_ROOT / f'perbandingan_{Path(MODEL).stem}_cracks_{DATASET_FINGERPRINT[:8]}_{RUN_SUFFIX}.csv'
    summary.to_csv(summary_path, index=False)
    display(summary[['run', 'skema', 'val_mAP50_95', 'test_precision', 'test_recall',
                     'test_mAP50', 'test_mAP50_95']].round(4))
    INFERENCE_SKEMA = summary.loc[summary['val_mAP50_95'].idxmax(), 'skema']
    print('Skema untuk inferensi berdasarkan VALIDASI:', INFERENCE_SKEMA)
    print('Ringkasan:', summary_path)
    """)
    add("code", """
    # 10 · Inference contoh gambar (opsional, setelah training).
    inference_model = YOLO(str(run_paths(INFERENCE_SKEMA)[1]))

    def predict_crack(image_path, confidence=PREDICT_CONF):
        result = inference_model.predict(source=str(image_path), imgsz=IMG_SIZE, conf=confidence,
                                         device=DEVICE, half=False, verbose=False)[0]
        print('Gambar:', image_path)
        if result.boxes is None or len(result.boxes) == 0:
            print('Tidak ada crack terdeteksi pada threshold ini.')
        else:
            for cls, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
                print('Label:', result.names[int(cls)], '| Confidence:', f'{score:.2%}',
                      '| bbox:', [round(value, 1) for value in box])
        plt.figure(figsize=(12, 6))
        plt.imshow(result.plot()[..., ::-1])
        plt.axis('off')
        plt.show()
        return result

    CONTOH_GAMBAR = X_test[0]  # Ganti dengan path foto sendiri jika diinginkan.
    contoh_hasil = predict_crack(CONTOH_GAMBAR)
    """)
    add("code", """
    # 11 · Upload foto sendiri untuk inference (opsional).
    # Aktifkan RUN_UPLOAD_TEST jika ingin dialog upload muncul.
    RUN_UPLOAD_TEST = False
    if RUN_UPLOAD_TEST:
        from google.colab import files
        uploaded = files.upload()
        for filename in uploaded:
            predict_crack(filename)
    """)
    add("markdown", """
    Hasil tersedia di `MyDrive/deteksi-jalan/hasil/yolo_cracks_group_split/<RUN_NAME>/`:

    | File | Isi |
    | --- | --- |
    | `weights/best.pt`, `weights/last.pt` | Model terbaik berdasarkan validasi dan checkpoint resume |
    | `results.csv`, `results.png` | Riwayat serta grafik training |
    | `eval_val/`, `eval_test/` | Grafik evaluasi, confusion matrix, dan kurva PR |
    | `metrics.csv` | Precision, recall, mAP50, dan mAP50-95 validasi/test |
    | `predictions_test.csv` | Prediksi bbox pada test |
    | `split_manifest.csv`, `group_manifest.csv` | Pembagian foto dan kelompok untuk audit |
    | `split_config.json`, `validation.json`, `config.json` | Seed, rasio, fingerprint, dan konfigurasi run |

    Sumber/atribusi dataset berada dalam ZIP di `input_provenance/source_metadata/`. Dataset pothole dan notebook lama tidak diubah. Jangan membandingkan metrik split baru langsung dengan metrik split lama sebagai perbandingan model yang setara.
    """)
    if archive6_only or roads_bridges_full:
        def replace_cell(index, value):
            cells[index]["source"] = (textwrap.dedent(value).strip() + "\n").splitlines(keepends=True)

        replace_cell(0, """
        # Deteksi Retakan Jalan · YOLO26s · Archive 6 · Split Kelompok

        Notebook Google Colab khusus **satu kelas `cracks`**, hanya memakai **1.232 foto kandidat aspal hasil kurasi archive 6**. Ini subset yang telah dipilah sebelumnya, bukan seluruh foto mentah archive 6. Seluruh gambar dan label training berasal dari archive 6.

        Target pembagian **80% train, 10% validasi, 10% test**, seed 42. Foto asal, versi augmentasinya, dan pasangan yang terdeteksi mirip tetap dalam satu split. Tabel jumlah aktual ditampilkan sebelum training. Kedua skema memakai pembagian yang sama.

        1. Aktifkan runtime **GPU** di Colab.
        2. Simpan `cracks_archive6_group_split_80_10_10.zip` di `MyDrive/deteksi-jalan/data/`.
        3. Jalankan sel berurutan. Default **Skema A**, sesuai eksperimen sebelumnya; B tetap tersedia.
        4. Bobot dan hasil disimpan dalam folder Drive khusus archive 6. Tidak memakai checkpoint eksperimen gabungan.

        Default memakai split dalam ZIP. Untuk pembagian baru, set `RESPLIT=True`, ganti `SPLIT_SEED` dan `RUN_SUFFIX`. Sumber dan seed identik menghasilkan pembagian identik. Pemeriksaan jumlah/sumber menolak paket yang masih memuat archive 7.

        Ini **fine-tuning `yolo26s.pt`**. Pencarian kemiripan bersifat heuristik; independensi seluruh lokasi/adegan tidak dapat dijamin. Archive 6 telah memuat augmentasi offline; variannya tetap satu kelompok, termasuk bila kelompok masuk validasi/test. Material aspal adalah hasil pemilahan visual, bukan verifikasi material. Bbox mengikuti anotasi sumber.

        Nilai evaluasi dataset gabungan dan archive 6 saja tidak dapat dibandingkan langsung sebagai pengaruh dataset training karena domain dan keanggotaan set uji ikut berubah. Nilai sebelumnya sekitar 0,50 belum membuktikan penyebabnya, dan notebook ini tidak menjamin skor tertentu. Training baru belum dijalankan saat notebook disiapkan.
        """)
        config = "".join(cells[3]["source"])
        config = config.replace("cracks_yolo_group_split_80_10_10.zip", "cracks_archive6_group_split_80_10_10.zip")
        config = config.replace("/content/deteksi-jalan/cracks_group_split", "/content/deteksi-jalan/cracks_archive6")
        config = config.replace("'yolo_cracks_group_split'", "'yolo_cracks_archive6'")
        config = config.replace("RESPLIT = False", "EXPECTED_SOURCES = {'archive6'}\nEXPECTED_IMAGES = 1232\n\nRESPLIT = False")
        replace_cell(3, config)
        loading = "".join(cells[6]["source"])
        loading = loading.replace("prepare_dataset(DATA_ZIP, staged, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)",
                                  "prepare_dataset(DATA_ZIP, staged, seed=SPLIT_SEED, ratios=SPLIT_RATIOS, allowed_sources=EXPECTED_SOURCES)")
        marker = "df = pd.read_csv(YOLO_DATA_DIR / 'split_manifest.csv', keep_default_na=False)"
        loading = loading.replace(marker, marker + "\n" + textwrap.dedent("""
        if set(df['source']) != EXPECTED_SOURCES or df['name'].str.startswith('archive7__').any():
            raise ValueError('Notebook ini hanya menerima hasil kurasi archive 6. ZIP masih memuat sumber lain.')
        if len(df) != EXPECTED_IMAGES:
            raise ValueError(f'Diharapkan {EXPECTED_IMAGES} foto hasil kurasi archive 6, ditemukan {len(df)}.')
        """).strip())
        loading = loading.replace("cracks_archive6_archive7_curated", "cracks_archive6_curated")
        loading = loading.replace("group_random_80_10_10", "archive6_group_random_80_10_10")
        loading = loading.replace("display(split_table)", "display(split_table.drop(columns=['archive7'], errors='ignore'))")
        replace_cell(6, loading)
        runs = "".join(cells[9]["source"]).replace("_cracks_s{SPLIT_SEED}", "_archive6_cracks_s{SPLIT_SEED}")
        replace_cell(9, runs)
        inference_summary = "".join(cells[13]["source"]).replace("_cracks_{DATASET_FINGERPRINT", "_archive6_cracks_{DATASET_FINGERPRINT")
        replace_cell(13, inference_summary)
        final_notes = "".join(cells[16]["source"]).replace("yolo_cracks_group_split", "yolo_cracks_archive6")
        replace_cell(16, final_notes)

        if roads_bridges_full:
            info_path = ROOT / ".venv/roads_bridges_full_work/dataset_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            count = int(info["positive_unique_images"])
            replace_cell(0, f"""
            # Roads and Bridges Cracks · YOLO26s · Google Colab

            Dataset: **Roads and Bridges Cracks (YOLOv8 Format)** yang lengkap, satu kelas **`cracks`**. Semua jenis permukaan dari sumber dipakai, termasuk aspal, beton, dan struktur lain. Tidak ada filter khusus aspal dan tidak ada foto archive 7.

            **Format YOLOv8 tidak membatasi versi model.** Notebook ini memakai `yolo26s.pt` untuk deteksi bbox. Sumber mencampur 1.141 anotasi bbox dan 2.811 poligon; seluruh label dalam ZIP siap training telah diseragamkan menjadi bbox YOLO lima kolom dengan ID kelas `0`. Anotasi poligon diubah menjadi bbox min/max, sedangkan gambar dan label asli disimpan tanpa perubahan byte.

            Dari **{info['raw_images']:,} foto sumber**, terdapat **{count:,} foto positif unik** untuk training/evaluasi setelah 18 label kosong yang belum terverifikasi sebagai background dan satu duplikat disisihkan. Semua gambar berlabel yang valid dipakai, tanpa pemilahan material permukaan.

            Target split **80% train, 10% validasi, 10% test**, seed 42. Seluruh augmentasi dari satu foto asal dan pasangan terdeteksi mirip tetap satu split. Tabel jumlah aktual ditampilkan sebelum training. Skema A dan B memakai pembagian identik.

            1. Aktifkan runtime **GPU** di Colab.
            2. Upload **`roads_bridges_cracks_yolo26.zip`** ke `MyDrive/deteksi-jalan/data/`. Pakai ZIP siap training ini agar label bbox-nya sudah seragam.
            3. Jalankan sel berurutan. Default **Skema A**; B opsional.
            4. Model dan metrik tersimpan di folder Drive khusus `yolo_roads_bridges_cracks`.

            Default memakai split dalam ZIP. Untuk acak ulang, set `RESPLIT=True`, ganti `SPLIT_SEED` dan `RUN_SUFFIX`. Sumber dan seed yang identik menghasilkan split identik. Notebook menolak ZIP subset aspal maupun dataset gabungan.

            Ini **fine-tuning YOLO26s**. Augmentasi offline sumber tetap satu kelompok, termasuk jika kelompok masuk validasi/test. Pemeriksaan kemiripan mengurangi risiko kebocoran tetapi tidak menjamin independensi seluruh lokasi/adegan. Jumlah kelompok asal tersedia di manifest. Semantik anotasi mengikuti sumber dan belum dianotasi ulang. Nilai 0,50 pada dataset sebelumnya belum cukup untuk memastikan penyebabnya; skor pada dataset ini belum diketahui sebelum training.
            """)
            config = "".join(cells[3]["source"])
            config = config.replace("cracks_archive6_group_split_80_10_10.zip", "roads_bridges_cracks_yolo26.zip")
            config = config.replace("cracks_archive6", "roads_bridges_cracks")
            config = config.replace("EXPECTED_IMAGES = 1232", f"EXPECTED_IMAGES = {count}\nEXPECTED_DATASET_ID = 'roads_bridges_cracks_full'")
            replace_cell(3, config)
            loading = "".join(cells[6]["source"])
            input_info_check = textwrap.dedent("""
            with zipfile.ZipFile(DATA_ZIP) as archive:
                if 'dataset_info.json' not in archive.namelist():
                    raise ValueError('Pakai roads_bridges_cracks_yolo26.zip yang label bbox-nya sudah diseragamkan.')
                SOURCE_DATASET_INFO = json.loads(archive.read('dataset_info.json'))
                if SOURCE_DATASET_INFO.get('dataset_id') != EXPECTED_DATASET_ID:
                    raise ValueError('Notebook ini untuk seluruh Roads and Bridges Cracks, bukan subset/gabungan lain.')
            """).strip() + "\n"
            loading = input_info_check + loading
            loading = loading.replace("allowed_sources=EXPECTED_SOURCES)", "allowed_sources=EXPECTED_SOURCES, dataset_info=SOURCE_DATASET_INFO)")
            loading = loading.replace("Notebook ini hanya menerima hasil kurasi archive 6. ZIP masih memuat sumber lain.",
                                      "Notebook ini hanya menerima Roads and Bridges Cracks. ZIP masih memuat sumber lain.")
            loading = loading.replace("foto hasil kurasi archive 6", "foto positif unik Roads and Bridges Cracks")
            loading = loading.replace("cracks_archive6_curated", "roads_bridges_cracks_full")
            loading = loading.replace("archive6_group_random_80_10_10", "roads_bridges_group_random_80_10_10")
            loading = loading.replace("SPLIT_CONFIG = json.loads((YOLO_DATA_DIR / 'split_config.json').read_text(encoding='utf-8'))",
                                      "SPLIT_CONFIG = json.loads((YOLO_DATA_DIR / 'split_config.json').read_text(encoding='utf-8'))\n"
                                      "if SPLIT_CONFIG.get('dataset_id') != EXPECTED_DATASET_ID:\n"
                                      "    raise ValueError('Identitas dataset dalam cache tidak sesuai Roads and Bridges Cracks lengkap.')")
            replace_cell(6, loading)
            replace_cell(9, "".join(cells[9]["source"]).replace("archive6_cracks_s", "roads_bridges_cracks_s"))
            replace_cell(13, "".join(cells[13]["source"]).replace("archive6_cracks_", "roads_bridges_cracks_"))
            replace_cell(16, "".join(cells[16]["source"]).replace("yolo_cracks_archive6", "yolo_roads_bridges_cracks"))

    if model != "yolo26s.pt":
        for cell in cells:
            text = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                text = text.replace("YOLO26s", model_name).replace("`yolo26s.pt`", f"`{model}`")
            else:
                text = text.replace("MODEL = 'yolo26s.pt'", f"MODEL = '{model}'")
            cell["source"] = text.splitlines(keepends=True)

    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.12"),
        colab=dict(name=output.name, provenance=[]), accelerator="GPU"), nbformat=4, nbformat_minor=5)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(str(output))
    print(f"{len(cells)} cells")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--archive6-only", action="store_true")
    group.add_argument("--roads-bridges-full", action="store_true")
    parser.add_argument("--model", choices=("yolo26s.pt", "yolo11s.pt", "yolov8s.pt"), default="yolo26s.pt")
    args = parser.parse_args()
    main(archive6_only=args.archive6_only, roads_bridges_full=args.roads_bridges_full, model=args.model)
