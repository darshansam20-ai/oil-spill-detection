# ==============================================================================
# AEGIS-SAR Production ML Inference Service Container
# Configured for Render Web Service (Docker Runtime) & Container Hosts
# ==============================================================================

FROM python:3.11-slim

# Prevent interactive prompts and optimize Linux memory footprint
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    DEVICE=cpu

# Install required system libraries (GDAL, OpenCV headless, libspatialindex, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, model checkpoints (383MB best_model.pt), AIS reference data, and assets
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY data/sample_ais/ /app/data/sample_ais/
COPY data/sample_metadata.json /app/data/sample_metadata.json
COPY public/ /app/public/

# Ensure writable directories exist for runtime outputs
RUN mkdir -p /app/output /app/data/outputs /app/data/processed /app/data/extracted && \
    chmod -R 777 /app/output /app/data

# Expose default HTTP port
EXPOSE 8000

# Container healthcheck using Render PORT fallback
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start FastAPI production server binding to 0.0.0.0 and dynamic Render $PORT
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
