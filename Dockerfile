# ISHI FastAPI backend - production container
# - Small, fast, CPU-only Torch
# - Works on Cloud Run (binds $PORT) and locally (8080)

FROM python:3.11-slim

# Keep Python output unbuffered and avoid .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Minimal system libs helpful for PyTorch/OpenCV
# (keep this lean; add only if you hit missing libs at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
  && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# --- Install runtime dependencies ---
# Use the lean requirements.txt we discussed (server-only)
# --- Install runtime dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2 && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application code & model ---
# Copy only what's needed to run the server
COPY api ./api
COPY scripts ./scripts
# If your checkpoint is tracked in the repo under api/models/, this brings it in.
# Adjust path if your model lives elsewhere.
COPY api/models ./api/models

# --- Environment defaults (override in Cloud Run if needed) ---
ENV PYTHONPATH=/app \
    PORT=8080 \
    ANEMIA_CKPT=/app/api/models/anemia_cnn.pth \
    ANEMIA_ARCH=simple_cnn \
    ANEMIA_THRESHOLD=0.50

# Cloud Run maps traffic to $PORT (default 8080)
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
