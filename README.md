# 🛰️ AEGIS-SAR: Sentinel-1 SAR Oil Spill Detection & AIS Maritime Vessel Tracking System

An end-to-end automated deep learning and geospatial intelligence platform that detects oil spills from Sentinel-1 SAR satellite imagery and correlates the detected spill zones with AIS (Automatic Identification System) maritime vessel trajectories to identify suspect polluter vessels.

---

## 📌 Architecture & Production System Overview

AEGIS-SAR couples two specialized machine learning / geospatial systems into a unified production pipeline:

```mermaid
flowchart TD
    User["End User / Browser"] -->|HTTPS| Frontend["Vercel Deployed Web Application\n(Interactive Dashboard & Visualizer)"]
    Frontend -->|POST /predict (Multipart Form)| MLAPI["Dedicated ML Inference API (FastAPI)"]

    subgraph PipelineExecution ["Sequential Dual-Model Pipeline"]
        MLAPI -->|Input SAR Scene| MyModel["1. My Model: Sentinel-1 SAR Deep Learning\n(ConvNeXt-Tiny + U-Net)"]
        MyModel -->|DetectionResult: Spills, Centroids, BBoxes| Adapter["2. Intelligent Pipeline Adapter\n(Epicenter Extraction, ISO-8601 Timestamp, AOI)"]
        Adapter -->|Spill Lat/Lon, Time, Radius| FriendsModel["3. Friend's Model: AIS Vessel Correlator\n(Global Fishing Watch + Haversine Geodesics)"]
        FriendsModel -->|Vessel Rankings & Maps| Response["4. Incident Dossier & Visual Artifacts"]
    end

    Response -->|JSON + Base64 PNG + HTML Map| Frontend
    Frontend -->|Render Split-Screen Intelligence View| User
```

### Production Flow:
1. **My Model (SAR Oil Spill Detection)**:
   - Deep learning semantic segmentation (`ConvNeXt-Tiny + U-Net`) trained on SAR satellite imagery.
   - Radiometric calibration, speckle filtering (Refined Lee), sliding-window 256x256 patch inference, connected component analysis, and geospatial polygon conversion.
2. **Adapter Layer (`PipelineAdapter`)**:
   - Bridges the semantic output of the detector to the AIS engine.
   - Selects primary spill epicenter, converts dates/times to ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`), and configures geodesic search radius.
3. **Friend's Model (`AISCorrelator`)**:
   - Queries Global Fishing Watch (GFW) API and calculates geodesic Haversine trajectories.
   - Ranks suspect vessels by minimum distance (point of closest approach) and generates interactive Folium HTML maps and CSV/JSON rankings.
4. **Final Deliverables**:
   - High-resolution annotated SAR image with bounding boxes & contours.
   - Interactive AIS maritime vessel radar map with trajectories and suspect highlights.
   - Ranked suspect vessels table (MMSI, ship name, type, distance km).
   - Exportable full incident dossier (`.json` and `.csv`).

---

## 🚀 Quickstart: Running Locally

### 1. Unified Python CLI
```bash
# Run pipeline on sample Sentinel-1 image with metadata
python run.py --image data/extracted/test/images/2018_12_19_d.tif --metadata data/sample_metadata.json

# Pure image input (auto-infers and correlates)
python run.py --image data/extracted/test/images/2018_12_19_d.tif
```

### 2. Launch Local FastAPI Web Server & Dashboard
```bash
# Start local development server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Open your browser at:
# http://localhost:8000/
# API Documentation: http://localhost:8000/docs
```

### 3. Run with Docker Compose
```bash
docker compose up --build
```

---

## 🌐 Production Deployment

AEGIS-SAR uses a **split hybrid architecture**:
- **Frontend**: Deployed on **Vercel** (`vercel.json`, `package.json`, `public/`).
- **ML Service**: Deployed on **Hugging Face Spaces** (Free 16GB RAM container), **Render**, or **Docker VPS** (`Dockerfile`, `docker-compose.yml`).

For step-by-step instructions, see the complete [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 📡 Production API Reference

### `GET /health`
Liveness and readiness probe.
```json
{
  "status": "online",
  "service": "Automated SAR Oil-Spill Detection & AIS Tracking System",
  "version": "1.0.0",
  "model_version": "v1.0.0-convnext-unet",
  "device": "cpu",
  "cuda_available": false,
  "checkpoint_loaded": true
}
```

### `POST /predict`
Runs the complete 3-stage pipeline on an uploaded SAR image.
- **Form Data**:
  - `image`: SAR image file (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`) — *required*
  - `metadata`: Optional JSON string
  - `date`: Optional date string (e.g. `2018-12-19`)
  - `time`: Optional time string (e.g. `06:15:22 UTC`)
  - `aoi`: Optional bounding box string (`min_lon,min_lat,max_lon,max_lat`)
  - `lat` / `lon`: Optional override coordinates
  - `threshold`: Detection threshold (default `0.50`)
  - `search_radius_km`: AIS search radius in km (default `20.0`)

---

## 🧪 Automated Testing

Run the end-to-end production verification suite:
```bash
python scripts/test_production_api.py
```

---

## 📄 License
MIT License.
