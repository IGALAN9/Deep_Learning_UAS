"""API deteksi pothole dan crack dengan dua checkpoint YOLO26s Skema A."""

from contextlib import asynccontextmanager
from io import BytesIO
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from PIL.Image import open as pillow_open
from pydantic import BaseModel
from ultralytics import YOLO


# Paths dan konfigurasi mengikuti susunan repository Model: api.py + src/models.
BASE_DIR = Path(__file__).resolve().parent
_model_path = Path(os.getenv("MODEL_PATH", "src/models/best_yolo26s_A.pt"))
MODEL_PATH = _model_path if _model_path.is_absolute() else BASE_DIR / _model_path
METADATA_PATH = BASE_DIR / "src" / "models" / "model_info.json"
_crack_path = Path(os.getenv("CRACK_MODEL_PATH", "src/models/best_yolo26s_cracks_A.pt"))
CRACK_MODEL_PATH = _crack_path if _crack_path.is_absolute() else BASE_DIR / _crack_path
CRACK_METADATA_PATH = BASE_DIR / "src" / "models" / "crack_model_info.json"
DEVICE = os.getenv("YOLO_DEVICE", "cpu")
IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.7
MAX_DETECTIONS = 300
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_BATCH_SIZE = 10
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
app_state: dict = {}
logger = logging.getLogger("pothole_api")
COMBINED_CLASSES = {0: "pothole", 1: "cracks"}
DETECTION_COLORS = {"pothole": "#c62828", "cracks": "#1565c0"}


# Schemas
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    bbox_normalized: BoundingBox


class PredictResponse(BaseModel):
    filename: str
    width: int
    height: int
    detected: bool
    count: int
    confidence_threshold: float
    iou_threshold: float
    processing_time_ms: float
    detections: list[Detection]


class BatchPredictResponse(BaseModel):
    total: int
    results: list[PredictResponse]


class CombinedDetection(Detection):
    source_model: str
    source_class_id: int


class CombinedPredictResponse(PredictResponse):
    detections: list[CombinedDetection]
    counts: dict[str, int]
    models_used: list[str]
    max_detections_per_model: int


class CombinedHealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    combined_ready: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model: str
    scheme: str
    task: str
    classes: dict[int, str]
    device: str
    image_size: int
    default_confidence: float
    default_iou: float
    max_detections: int
    max_batch_size: int
    max_upload_mb: int
    checkpoint: str
    checkpoint_sha256: str
    ultralytics_version: str
    training_ultralytics_version: str | None
    test_metrics: dict[str, float]


def _load_model() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Bobot tidak ditemukan: {MODEL_PATH}. Letakkan best_yolo26s_A.pt "
            "di src/models atau atur MODEL_PATH ke checkpoint YOLO26s Skema A."
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with MODEL_PATH.open("rb") as checkpoint:
        checksum = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if checksum != metadata["sha256"]:
        raise RuntimeError("Checksum bobot berbeda dari checkpoint YOLO26s Skema A yang dipilih.")

    model = YOLO(str(MODEL_PATH), task="detect")
    if model.task != "detect" or model.names != {0: "pothole"}:
        raise RuntimeError("Checkpoint harus berupa model deteksi dengan kelas 0 = pothole.")
    model.to(DEVICE)
    # Warm-up memverifikasi kompatibilitas bobot/runtime sebelum API menerima request.
    warmup = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    try:
        model.predict(
            source=warmup, imgsz=IMAGE_SIZE, conf=DEFAULT_CONFIDENCE,
            iou=DEFAULT_IOU, max_det=MAX_DETECTIONS, device=DEVICE,
            half=False, verbose=False, save=False,
        )
    finally:
        warmup.close()

    app_state.update(model=model, lock=Lock(), metadata=metadata, checksum=checksum)
    logger.info("YOLO26s Skema A siap pada device %s", DEVICE)


def _load_crack_model() -> None:
    if not CRACK_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Bobot crack tidak ditemukan: {CRACK_MODEL_PATH}")
    metadata = json.loads(CRACK_METADATA_PATH.read_text(encoding="utf-8"))
    with CRACK_MODEL_PATH.open("rb") as checkpoint:
        checksum = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if checksum != metadata["sha256"]:
        raise RuntimeError("Checksum bobot crack berbeda dari checkpoint yang dipilih.")
    model = YOLO(str(CRACK_MODEL_PATH), task="detect")
    if model.task != "detect" or model.names != {0: "cracks"}:
        raise RuntimeError("Checkpoint crack harus berupa model deteksi dengan kelas 0 = cracks.")
    model.to(DEVICE)
    warmup = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    try:
        model.predict(source=warmup, imgsz=IMAGE_SIZE, conf=DEFAULT_CONFIDENCE,
                      iou=DEFAULT_IOU, max_det=MAX_DETECTIONS, device=DEVICE,
                      half=False, verbose=False, save=False)
    finally:
        warmup.close()
    app_state.update(crack_model=model, crack_lock=Lock(), crack_metadata=metadata, crack_checksum=checksum)
    logger.info("YOLO26s crack Skema %s siap pada device %s", metadata["scheme"], DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
        _load_crack_model()
        yield
    finally:
        app_state.clear()


app = FastAPI(
    title="Road Damage Detection API",
    description=(
        "Deteksi crack dan pothole memakai dua model YOLO26s Skema A. "
        "Gunakan /predict/combined untuk satu foto dengan kedua detektor. "
        "Endpoint /predict tetap memakai detektor pothole saja. "
        "Bounding box memakai xyxy dalam piksel dan koordinat ternormalisasi 0-1. "
        "Tidak terdeteksi berarti tidak ada prediksi yang melewati threshold."
    ),
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[value.strip() for value in os.getenv("CORS_ORIGINS", "*").split(",") if value.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

Confidence = Annotated[float, Query(ge=0.01, le=1.0, description="Ambang confidence setiap bounding box.")]
IoU = Annotated[float, Query(ge=0.01, le=1.0, description="Ambang IoU untuk postprocessing model.")]
DetectionLimit = Annotated[int, Query(ge=1, le=MAX_DETECTIONS)]


# Helpers
def _require_model():
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    return app_state["model"]


def _read_image(file: UploadFile) -> Image.Image:
    """Periksa isi file, batasi ukuran, perbaiki orientasi EXIF, lalu ubah ke RGB."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="File gambar kosong.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran gambar maksimal 10 MB per file.")
    try:
        # Alias diimpor sebelum Ultralytics mengubah Image.open menjadi decoder
        # dengan auto-install HEIF. Input rusak cukup ditolak sebagai HTTP 400.
        with pillow_open(BytesIO(content)) as opened:
            if opened.format not in SUPPORTED_FORMATS:
                raise HTTPException(status_code=415, detail="Format gambar harus JPEG, PNG, atau WebP.")
            if getattr(opened, "is_animated", False):
                raise HTTPException(status_code=415, detail="Gunakan gambar statis, bukan animasi.")
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Resolusi gambar maksimal 16 megapiksel.")
            opened.load()
            oriented = ImageOps.exif_transpose(opened)
            try:
                return oriented.convert("RGB")
            finally:
                oriented.close()
    except Image.DecompressionBombError as error:
        raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(status_code=400, detail="File bukan gambar valid atau rusak.") from error


def _predict_single(
    file: UploadFile, confidence: float, iou: float, max_det: int, *, annotated: bool = False,
) -> tuple[PredictResponse, bytes | None]:
    model = _require_model()
    started = perf_counter()
    image = _read_image(file)
    try:
        # Predictor Ultralytics menyimpan argumen/result; satu model dipakai bergiliran
        # agar request bersamaan tidak saling menimpa konfigurasi dan hasil plot.
        with app_state["lock"]:
            result = model.predict(
                source=image, imgsz=IMAGE_SIZE, conf=confidence, iou=iou,
                max_det=max_det, device=DEVICE, half=False, verbose=False, save=False,
            )[0]
            width, height = image.size
            detections = []
            if result.boxes is not None:
                for coordinates, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                ):
                    x1, y1, x2, y2 = coordinates
                    detections.append(Detection(
                        class_id=int(class_id), class_name=result.names[int(class_id)],
                        confidence=round(float(score), 6),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        bbox_normalized=BoundingBox(
                            x1=x1 / width, y1=y1 / height, x2=x2 / width, y2=y2 / height,
                        ),
                    ))
            jpeg = None
            if annotated:
                # plot() menghasilkan BGR; Pillow menyimpan RGB.
                plotted = Image.fromarray(result.plot()[..., ::-1])
                try:
                    buffer = BytesIO()
                    plotted.save(buffer, format="JPEG", quality=90)
                    jpeg = buffer.getvalue()
                finally:
                    plotted.close()
        response = PredictResponse(
            filename=Path((file.filename or "image").replace("\\", "/")).name,
            width=width, height=height, detected=bool(detections), count=len(detections),
            confidence_threshold=confidence, iou_threshold=iou,
            processing_time_ms=round((perf_counter() - started) * 1000, 2),
            detections=detections,
        )
        return response, jpeg
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Prediksi gambar gagal")
        raise HTTPException(status_code=500, detail="Prediksi gagal. Periksa log server.") from error
    finally:
        image.close()


def _combined_detections(result, source_model: str, width: int, height: int) -> list[CombinedDetection]:
    if result.boxes is None:
        return []
    global_class_id = 0 if source_model == "pothole" else 1
    detections = []
    for coordinates, score, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                             result.boxes.conf.cpu().tolist(),
                                             result.boxes.cls.cpu().tolist()):
        x1, y1, x2, y2 = coordinates
        detections.append(CombinedDetection(
            class_id=global_class_id, class_name=COMBINED_CLASSES[global_class_id],
            source_model=source_model, source_class_id=int(class_id), confidence=round(float(score), 6),
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            bbox_normalized=BoundingBox(x1=x1/width, y1=y1/height, x2=x2/width, y2=y2/height),
        ))
    return detections


def _combined_jpeg(image: Image.Image, detections: list[CombinedDetection]) -> bytes:
    plotted = image.copy()
    try:
        draw = ImageDraw.Draw(plotted)
        font = ImageFont.load_default(size=max(12, min(48, plotted.width // 60)))
        line_width = max(2, plotted.width // 400)
        for item in detections:
            box = item.bbox
            color = DETECTION_COLORS[item.class_name]
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=line_width)
            label = f"{item.class_name} {item.confidence:.1%}"
            text_box = draw.textbbox((0, 0), label, font=font)
            label_width, label_height = text_box[2]-text_box[0]+8, text_box[3]-text_box[1]+8
            x = max(0, min(box.x1, plotted.width-label_width))
            y = max(0, min(box.y1-label_height, plotted.height-label_height))
            draw.rectangle((x, y, x+label_width, y+label_height), fill=color)
            draw.text((x+4, y+4-text_box[1]), label, fill="white", font=font)
        buffer = BytesIO()
        plotted.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    finally:
        plotted.close()


def _predict_combined(
    file: UploadFile, confidence: float, iou: float, max_det: int, *, annotated: bool = False,
) -> tuple[CombinedPredictResponse, bytes | None]:
    pothole_model = _require_model()
    if "crack_model" not in app_state:
        raise HTTPException(status_code=503, detail="Model crack belum siap; prediksi gabungan tidak dijalankan.")
    started = perf_counter()
    image = _read_image(file)
    try:
        width, height = image.size
        options = dict(source=image, imgsz=IMAGE_SIZE, conf=confidence, iou=iou, max_det=max_det,
                       device=DEVICE, half=False, verbose=False, save=False)
        # Lock selalu dalam urutan yang sama. Inferensi berurutan pada gambar RGB yang sama.
        with app_state["lock"], app_state["crack_lock"]:
            pothole_result = pothole_model.predict(**options)[0]
            detections = _combined_detections(pothole_result, "pothole", width, height)
            crack_result = app_state["crack_model"].predict(**options)[0]
            detections.extend(_combined_detections(crack_result, "cracks", width, height))
        # Tidak ada NMS lintas kelas: crack dan pothole yang bertumpang tindih tetap tampil.
        detections.sort(key=lambda item: item.confidence, reverse=True)
        counts = {name: sum(item.class_name == name for item in detections) for name in COMBINED_CLASSES.values()}
        jpeg = _combined_jpeg(image, detections) if annotated else None
        result = CombinedPredictResponse(
            filename=Path((file.filename or "image").replace("\\", "/")).name,
            width=width, height=height, detected=bool(detections), count=len(detections),
            confidence_threshold=confidence, iou_threshold=iou,
            processing_time_ms=round((perf_counter()-started)*1000, 2), detections=detections,
            counts=counts, models_used=["pothole", "cracks"], max_detections_per_model=max_det,
        )
        return result, jpeg
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Prediksi gabungan gagal")
        raise HTTPException(status_code=500, detail="Prediksi gabungan gagal. Periksa log server.") from error
    finally:
        image.close()


# Routes
@app.get("/", tags=["Health"])
def root():
    return {"message": "Crack & Pothole Detection API", "model": "YOLO26s Skema A",
            "docs": "/docs", "ui": "/ui", "combined_prediction": "/predict/combined"}


@app.get("/ui", include_in_schema=False)
def test_ui():
    path = BASE_DIR / "test_ui.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="UI sementara tidak tersedia.")
    return FileResponse(path, media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(response: Response):
    loaded = "model" in app_state
    if not loaded:
        response.status_code = 503
    return HealthResponse(status="ok" if loaded else "error", model_loaded=loaded)


@app.get("/health/combined", response_model=CombinedHealthResponse, tags=["Health"])
def combined_health(response: Response):
    loaded = {"pothole": "model" in app_state, "cracks": "crack_model" in app_state}
    ready = all(loaded.values())
    if not ready:
        response.status_code = 503
    return CombinedHealthResponse(status="ok" if ready else "error", models_loaded=loaded, combined_ready=ready)


@app.get("/model", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    from ultralytics import __version__

    model = _require_model()
    metadata = app_state["metadata"]
    return ModelInfoResponse(
        model=metadata["model"], scheme=metadata["scheme"], task=model.task,
        classes=model.names, device=DEVICE, image_size=IMAGE_SIZE,
        default_confidence=DEFAULT_CONFIDENCE, default_iou=DEFAULT_IOU,
        max_detections=MAX_DETECTIONS, max_batch_size=MAX_BATCH_SIZE,
        max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024), checkpoint=MODEL_PATH.name,
        checkpoint_sha256=app_state["checksum"], ultralytics_version=__version__,
        training_ultralytics_version=metadata["training_ultralytics_version"],
        test_metrics=metadata["test_metrics"],
    )


@app.get("/models", tags=["Model"])
def combined_model_info():
    from ultralytics import __version__

    pothole = model_info()
    if "crack_model" not in app_state:
        raise HTTPException(status_code=503, detail="Model crack belum siap.")
    metadata = app_state["crack_metadata"]
    crack = ModelInfoResponse(
        model=metadata["model"], scheme=metadata["scheme"], task=app_state["crack_model"].task,
        classes=app_state["crack_model"].names, device=DEVICE, image_size=IMAGE_SIZE,
        default_confidence=DEFAULT_CONFIDENCE, default_iou=DEFAULT_IOU,
        max_detections=MAX_DETECTIONS, max_batch_size=MAX_BATCH_SIZE,
        max_upload_mb=MAX_UPLOAD_BYTES // (1024*1024), checkpoint=CRACK_MODEL_PATH.name,
        checkpoint_sha256=app_state["crack_checksum"], ultralytics_version=__version__,
        training_ultralytics_version=metadata["training_ultralytics_version"], test_metrics=metadata["test_metrics"],
    )
    return {"mode": "two_detectors", "classes": COMBINED_CLASSES, "models": {"pothole": pothole, "cracks": crack}}


@app.post("/predict/combined", response_model=CombinedPredictResponse, tags=["Combined prediction"])
def predict_combined(
    file: Annotated[UploadFile, File(description="Satu foto untuk deteksi crack dan pothole.")],
    confidence: Confidence = DEFAULT_CONFIDENCE,
    iou: IoU = DEFAULT_IOU,
    max_det: DetectionLimit = MAX_DETECTIONS,
):
    """Jalankan dua detektor. ID gabungan: 0=pothole, 1=cracks. max_det berlaku per model."""
    return _predict_combined(file, confidence, iou, max_det)[0]


@app.post(
    "/predict/combined/image", response_class=Response, tags=["Combined prediction"],
    responses={200: {"content": {"image/jpeg": {}}, "description": "Pothole merah dan crack biru dalam satu gambar."}},
)
def predict_combined_image(
    file: Annotated[UploadFile, File(description="Foto yang akan diberi bbox crack dan pothole.")],
    confidence: Confidence = DEFAULT_CONFIDENCE,
    iou: IoU = DEFAULT_IOU,
    max_det: DetectionLimit = MAX_DETECTIONS,
):
    result, jpeg = _predict_combined(file, confidence, iou, max_det, annotated=True)
    return Response(content=jpeg, media_type="image/jpeg", headers={
        "Content-Disposition": 'inline; filename="road_damage_detection.jpg"',
        "X-Detection-Count": str(result.count), "X-Pothole-Count": str(result.counts["pothole"]),
        "X-Crack-Count": str(result.counts["cracks"]),
    })


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(
    file: Annotated[UploadFile, File(description="Gambar JPEG, PNG, atau WebP.")],
    confidence: Confidence = DEFAULT_CONFIDENCE,
    iou: IoU = DEFAULT_IOU,
    max_det: DetectionLimit = MAX_DETECTIONS,
):
    return _predict_single(file, confidence, iou, max_det)[0]


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(
    files: Annotated[list[UploadFile], File(description="Maksimal 10 gambar per request.")],
    confidence: Confidence = DEFAULT_CONFIDENCE,
    iou: IoU = DEFAULT_IOU,
    max_det: DetectionLimit = MAX_DETECTIONS,
):
    if not 1 <= len(files) <= MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail="Kirim 1 sampai 10 gambar per batch.")
    results = []
    for file in files:
        try:
            results.append(_predict_single(file, confidence, iou, max_det)[0])
        except HTTPException as error:
            raise HTTPException(
                status_code=error.status_code,
                detail={"filename": Path((file.filename or "image").replace("\\", "/")).name,
                        "message": error.detail},
            ) from error
    return BatchPredictResponse(total=len(results), results=results)


@app.post(
    "/predict/image", tags=["Prediction"], response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}, "description": "Gambar dengan bounding box dan confidence."}},
)
def predict_image(
    file: Annotated[UploadFile, File(description="Gambar yang akan diberi bounding box.")],
    confidence: Confidence = DEFAULT_CONFIDENCE,
    iou: IoU = DEFAULT_IOU,
    max_det: DetectionLimit = MAX_DETECTIONS,
):
    result, jpeg = _predict_single(file, confidence, iou, max_det, annotated=True)
    return Response(
        content=jpeg, media_type="image/jpeg",
        headers={"Content-Disposition": 'inline; filename="pothole_detection.jpg"',
                 "X-Detection-Count": str(result.count)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

