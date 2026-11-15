FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Wheel CPU menjaga image deployment tidak membawa pustaka CUDA yang besar.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY test_ui.html .
COPY src/models ./src/models

ENV YOLO_DEVICE=cpu YOLO_CONFIG_DIR=/tmp/ultralytics PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --start-period=120s --interval=30s --timeout=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health/combined', timeout=5)"

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

