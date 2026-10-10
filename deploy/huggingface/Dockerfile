# FlowSight full backend (FastAPI + YOLO11 + analytics) for a Hugging Face Docker Space (free CPU tier).
# Built by scripts/deploy_hf.py, which copies the repository next to this file.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 nodejs npm ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    FLOWSIGHT_DEMO=1 \
    FLOWSIGHT_CORS_ORIGINS=*
WORKDIR /home/user/app

# Python deps (CPU PyTorch wheels keep the image small)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Console, served by the API from web/dist on the same origin
RUN cd web && npm ci --no-audit --no-fund && npm run build && rm -rf node_modules && chown -R user:user /home/user/app

USER user
# Real CCTV test clips + floor plans + YOLO weights, fetched at build time so the Space works on cold start
RUN python scripts/download_data.py --caviar-only \
    && python -c "from flowsight.config import load_config; from flowsight.perception.detector_tracker import ensure_weights; c=load_config().runtime; ensure_weights(c.model_cpu, c.models_dir)"

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
