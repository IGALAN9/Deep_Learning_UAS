# Crack & Pothole Detection API - YOLO26s Skema A

API FastAPI untuk mengecek **crack dan pothole pada satu foto**. `POST /predict/combined` menjalankan dua checkpoint `best.pt` YOLO26s Skema A secara berurutan, lalu menggabungkan deteksinya. Kedua bobot sudah tersedia di `src/models`; tidak perlu training ulang atau mengunduh dataset untuk menjalankan API.

Buka [UI upload](http://127.0.0.1:8000/ui) setelah server berjalan. Kotak **merah = pothole**, **biru = crack**. Endpoint lama `/predict`, `/predict/batch`, dan `/predict/image` tetap untuk pothole saja.

Ini dua detektor terpisah dalam satu alur pengecekan. Kedua checkpoint memakai ID lokal 0; response gabungan memetakannya menjadi **0=pothole, 1=cracks**, sambil menyimpan `source_model` dan `source_class_id`. Kotak berbeda kelas yang bertumpang tindih tetap dipertahankan.

Susunannya mengikuti repository [IGALAN9/Model](https://github.com/IGALAN9/Model): satu `api.py` yang berisi lifespan, app state, schemas, helpers, dan routes; notebook, bobot, dan laporan berada di `src`; tersedia collection Bruno, Dockerfile, dan konfigurasi Render.

## Struktur

| Path | Isi |
| --- | --- |
| `api.py` | Startup model, validasi gambar, inference, schemas, dan endpoint FastAPI. |
| `test_ui.html` | UI sementara untuk upload foto dan mencoba API; dapat dihapus setelah selesai. |
| `src/models/best_yolo26s_A.pt` | Salinan bobot terbaik dari `Hasil Run/yolo26s_A_kaggle_pothole_01/weights/best.pt`. |
| `src/models/model_info.json` | Identitas model, SHA-256 bobot, konfigurasi, dan metrik test tersimpan. |
| `src/models/best_yolo26s_cracks_A.pt` | Salinan bobot `best.pt` dari `yolo26s_A_roads_bridges_cracks_01`. |
| `src/models/crack_model_info.json` | Identitas, SHA-256, dan metrik validasi/test model crack. |
| `src/Notebooks/deteksi_pothole_YOLO26s_skemaAB.ipynb` | Salinan notebook sumber training; masih memuat eksperimen A dan B. API hanya menggunakan A. |
| `src/logs/yolo26s_A_kaggle_pothole_01/` | Salinan konfigurasi dan metrik run A untuk penelusuran hasil. |
| `src/logs/yolo26s_A_roads_bridges_cracks_01/` | Konfigurasi dan metrik sumber model crack. |
| `bruno/Pothole/` | Collection uji request; pilih environment Local dan isi path gambar pada request upload. |
| `.devcontainer/devcontainer.json` | Lingkungan pengembangan Python 3.11. |
| `requirements.txt` | Dependensi runtime; Ultralytics dipin ke versi training `8.4.202`. |
| `requirements-dev.txt` | Dependensi tambahan untuk pengujian. |
| `Dockerfile`, `render.yaml` | Konfigurasi menjalankan API pada container dan Render. |
| `tests/test_api.py` | Pengujian endpoint dengan checkpoint asli. |
| `tests/test_inference.py` | Pengujian fungsi prediksi tanpa socket HTTP. |
| `tests/test_combined_api.py` | Dua checkpoint asli, pemetaan ID, output gabungan, warna bbox, dan kesiapan kedua model. |

YOLO menerima gambar, sehingga runtime ini menggunakan PyTorch/Ultralytics dan tidak membutuhkan tokenizer, TensorFlow, atau SHAP dari API klasifikasi teks.

## Model yang digunakan

Kedua detektor memakai **Skema A**. Untuk crack, validation mAP50-95 A=0,61754 dan B=0,60893. Pothole tetap memakai A yang sudah ada; nilai validation terbaik pada `results.csv` A=0,64267 dan B=0,63227.

| Detektor | Checkpoint | Kelas lokal | ID gabungan | Test mAP50 | Test mAP50-95 |
| --- | --- | --- | --- | ---: | ---: |
| Pothole | `best_yolo26s_A.pt` | `0=pothole` | `0=pothole` | 0,93147 | 0,66155 |
| Crack | `best_yolo26s_cracks_A.pt` | `0=cracks` | `1=cracks` | 0,96480 | 0,74062 |

Metrik ini berasal dari set uji masing-masing model yang berbeda; belum ada pengukuran akurasi gabungan pada foto beranotasi kedua kelas. Dataset crack mencakup berbagai permukaan jalan/jembatan. Bagian metrik pothole berikut tetap mendokumentasikan endpoint lama.

| Pengaturan | Nilai |
| --- | --- |
| Model / skema | YOLO26s / A (augmentasi standar saat training) |
| Kelas | `0: pothole` |
| Ukuran input | 640 |
| Confidence awal untuk prediksi | 0,25 |
| IoU awal | 0,70 |
| Maksimal bounding box per gambar | 300 |
| Device awal | CPU; bisa diganti dengan `YOLO_DEVICE` |

Nilai berikut berasal dari `metrics.csv` hasil run yang sudah ada, bukan hasil evaluasi ulang ketika API dijalankan:

| Metrik test | Nilai |
| --- | --- |
| Precision | 0,91528 |
| Recall | 0,87332 |
| mAP@0.5 | 0,93147 |
| mAP@0.5:0.95 | 0,66155 |
| Jumlah gambar train / val / test | 3.345 / 397 / 198 |

Confidence 0,25 mengikuti ekspor prediksi notebook. Evaluasi mAP notebook memakai confidence 0,001 untuk menghitung kurva; keduanya memiliki tujuan berbeda. Skema A adalah resep augmentasi training, sehingga API tidak menjalankan augmentasi cuaca atau training tambahan pada gambar pengguna.

## Menjalankan di Windows

Gunakan Python 3.11 atau lebih baru yang didukung dependensi; container memakai Python 3.11. Dari folder root repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Buka [UI upload crack dan pothole](http://127.0.0.1:8000/ui), pilih foto, lalu klik **Prediksi foto**. Untuk JSON, buka [Swagger UI](http://127.0.0.1:8000/docs) dan pilih `POST /predict/combined`.

Untuk pengembangan, tambahkan `--reload`. Alternatif setelah dependensi terpasang: jalankan `.\.venv\Scripts\python.exe api.py`.

Kedua model dimuat satu kali saat startup. Startup memeriksa checksum, kelas, dan inference pemanasan untuk masing-masing detektor. Startup gagal jika salah satu bobot tidak dapat dimuat; API tidak mengunduh bobot pengganti secara otomatis.

## UI sementara untuk mencoba upload

Setelah API berjalan, buka [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui). Alternatifnya, buka `test_ui.html` langsung di browser dan isi alamat API. UI memeriksa `/health/combined` lalu mengirim foto sekali ke `/predict/combined`. Halaman menampilkan bbox berwarna berbeda, jumlah per kelas, confidence, dan JSON.

UI ini hanya terdiri dari satu file tanpa dependensi tambahan. Untuk menghapusnya nanti, hapus `test_ui.html`. Jika `CORS_ORIGINS` diubah dari default `*`, origin halaman juga perlu diizinkan; saat dibuka langsung sebagai file lokal, origin browser umumnya `null`.

## Endpoint

| Method | Endpoint | Input / hasil |
| --- | --- | --- |
| GET | `/` | Identitas layanan dan lokasi dokumentasi. |
| GET | `/health` | Status kesiapan model; HTTP 503 jika belum siap. |
| GET | `/health/combined` | Kesiapan kedua model; HTTP 503 jika salah satu belum siap. |
| GET | `/ui` | UI upload dan hasil crack/pothole bersama. |
| GET | `/model` | Kelas, device, threshold, identitas bobot, versi runtime, dan metrik test tersimpan. |
| GET | `/models` | Metadata kedua checkpoint dan pemetaan kelas gabungan. |
| POST | `/predict/combined` | Satu file `file`; JSON berisi deteksi crack dan pothole. |
| POST | `/predict/combined/image` | JPEG dengan pothole merah dan crack biru. |
| POST | `/predict` | Satu file pada field `file`; hasil JSON berisi deteksi dan bounding box. |
| POST | `/predict/batch` | 1-10 file dengan nama field `files`; hasil JSON per gambar. |
| POST | `/predict/image` | Satu file pada field `file`; hasil JPEG dengan bounding box dan confidence. |

Semua endpoint prediksi menerima query `confidence` (0,01-1), `iou` (0,01-1), dan `max_det` (1-300). Pada prediksi gabungan, confidence yang sama diterapkan ke kedua detektor dan `max_det` berlaku **per model**, sehingga total bisa dua kali nilai tersebut. Postprocessing mengikuti masing-masing model; tidak ada NMS lintas kelas atau perataan confidence antar model.

Format yang diterima adalah JPEG, PNG, dan WebP statis, maksimal 10 MB dan 16 megapiksel per gambar. Orientasi EXIF diperbaiki sebelum prediksi. Gambar grayscale dan RGBA dikonversi ke RGB. Bounding box mengacu pada dimensi gambar setelah perbaikan orientasi.

```powershell
# Crack dan pothole dalam satu response
curl.exe -X POST "http://127.0.0.1:8000/predict/combined?confidence=0.25" -F "file=@C:\foto\jalan.jpg"

# JPEG dengan bbox kedua kelas
curl.exe --fail-with-body -X POST "http://127.0.0.1:8000/predict/combined/image" -F "file=@C:\foto\jalan.jpg" --output hasil_gabungan.jpg

# JSON bounding box
curl.exe -X POST "http://127.0.0.1:8000/predict?confidence=0.25" -F "file=@C:\foto\jalan.jpg"

# Simpan gambar hasil deteksi di komputer pemanggil
curl.exe --fail-with-body -X POST "http://127.0.0.1:8000/predict/image?confidence=0.25" -F "file=@C:\foto\jalan.jpg" --output hasil_deteksi.jpg

# Batch: nama field harus files untuk setiap gambar
curl.exe -X POST "http://127.0.0.1:8000/predict/batch" -F "files=@C:\foto\jalan1.jpg" -F "files=@C:\foto\jalan2.jpg"
```

Contoh bentuk response `/predict` (angka ilustrasi):

```json
{
  "filename": "jalan.jpg",
  "width": 1280,
  "height": 720,
  "detected": true,
  "count": 1,
  "confidence_threshold": 0.25,
  "iou_threshold": 0.7,
  "processing_time_ms": 125.4,
  "detections": [
    {
      "class_id": 0,
      "class_name": "pothole",
      "confidence": 0.9123,
      "bbox": {"x1": 128.0, "y1": 144.0, "x2": 384.0, "y2": 360.0},
      "bbox_normalized": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.5}
    }
  ]
}
```

`bbox` memakai `xyxy` dalam piksel. `bbox_normalized` membagi x dengan lebar dan y dengan tinggi. `count` adalah jumlah objek yang lolos threshold. Jika kosong, response berisi `detected: false`, `count: 0`, dan `detections: []`; ini tidak menyatakan kondisi jalan aman. `/predict` hanya mendeteksi pothole. `/predict/combined` menambahkan `counts: {"pothole": ..., "cracks": ...}`, `models_used`, `max_detections_per_model`, serta sumber model/ID lokal pada setiap bbox. Kegagalan salah satu model menghasilkan error, bukan hasil parsial yang dianggap lengkap.

`processing_time_ms` mencakup decoding, antrean model, inference, dan penyusunan hasil. Model dipakai bergiliran melalui lock supaya request bersamaan tidak saling mengubah konfigurasi predictor. Batch diproses berurutan dengan konfigurasi yang sama dan seluruh request gagal jika satu file tidak valid. Upload dan hasil prediksi tidak disimpan oleh aplikasi ke disk; framework dapat memakai file sementara untuk menampung upload.

## Environment variable

| Variabel | Default | Kegunaan |
| --- | --- | --- |
| `MODEL_PATH` | `src/models/best_yolo26s_A.pt` | Lokasi salinan checkpoint A yang sama; path relatif dihitung dari folder `api.py`. SHA-256 tetap harus sesuai metadata. |
| `CRACK_MODEL_PATH` | `src/models/best_yolo26s_cracks_A.pt` | Lokasi bobot crack; SHA-256 harus sesuai `crack_model_info.json`. |
| `YOLO_DEVICE` | `cpu` | Isi `0` untuk GPU pertama setelah memasang PyTorch dengan CUDA yang sesuai melalui [petunjuk resmi PyTorch](https://pytorch.org/get-started/locally/). |
| `CORS_ORIGINS` | `*` | Daftar origin frontend, dipisah koma. |
| `PORT` | `8000` | Port saat menggunakan `python api.py` atau Docker. Untuk perintah Uvicorn langsung, gunakan `--port`. |
| `YOLO_CONFIG_DIR` | Default Ultralytics | Lokasi konfigurasi Ultralytics; container mengarahkannya ke `/tmp/ultralytics`. |

Contoh GPU pada PowerShell:

```powershell
$env:YOLO_DEVICE = "0"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## Docker dan Render

```powershell
docker build -t pothole-yolo26-api .
docker run --rm -p 8000:8000 pothole-yolo26-api
```

Docker memasang PyTorch CPU dan library OpenCV. `api.py`, `test_ui.html`, dan `src/models` disalin ke runtime. Notebook dan folder eksperimen tidak diperlukan pada image deployment.

Untuk Render, push kode beserta `src/models` ke repository, lalu buat Blueprint dari `render.yaml`. Health check memakai `/health/combined`; port mengikuti `PORT`. Konfigurasi sudah disiapkan; proyek belum dipublikasikan oleh perubahan ini.

## Pengujian

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Pengujian memuat checkpoint asli dan melakukan inference CPU. Salah satu pengujian menggunakan montage `val_batch0_labels.jpg` dari hasil run sebagai pemeriksaan alur deteksi dan serialisasi bounding box. Montage tersebut sudah mengandung anotasi, sehingga pengujian ini hanya memeriksa jalannya API, bukan mengukur akurasi baru. Pengujian itu dilewati jika folder hasil eksperimen tidak tersedia. Pengujian lain tetap memeriksa health, metadata, upload, batch, output JPEG, EXIF, dan penolakan input tidak valid.

Jika lingkungan Windows membatasi socket lokal yang diperlukan TestClient, jalankan pengujian fungsi inference tanpa server HTTP:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_inference.py -v
```

Pengujian tersebut tetap memuat checkpoint asli dan memanggil fungsi prediksi yang sama, tetapi tidak memverifikasi transport HTTP atau validasi query FastAPI.

## Referensi implementasi

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics predict dan format Results](https://docs.ultralytics.com/modes/predict/)
- [FastAPI upload files](https://fastapi.tiangolo.com/tutorial/request-files/)

