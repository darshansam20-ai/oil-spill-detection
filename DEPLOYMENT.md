# AEGIS-SAR Production Deployment Guide

This guide explains how to deploy the **Integrated Sentinel-1 SAR Oil Spill Detection & AIS Maritime Vessel Tracking System** to production using a hybrid split architecture:
- **Frontend & Web Application**: Deployed on **Vercel**
- **Dedicated ML Inference Service**: Deployed on **Render** (Free Docker Web Service), **GCP Cloud Run**, or a **Docker VPS**

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    User["Teammates / End Users (Web Browser)"] -->|HTTPS| Vercel["Vercel Deployed Frontend (Global Edge CDN)"]
    Vercel -->|Multipart POST /predict| RenderHost["Render ML Service Container (FastAPI + Docker)"]
    
    subgraph RenderMLContainer ["Render Dedicated ML Container (0.0.0.0:$PORT)"]
        RenderHost -->|1. SAR Image (.tif/.png)| Model1["My Model: Sentinel-1 SAR Deep Learning\n(ConvNeXt-Tiny + U-Net)"]
        Model1 -->|DetectionResult: Epicenter & BBoxes| Adapter["Pipeline Adapter Layer\n(Epicenter & ISO Normalization)"]
        Adapter -->|Spill Lat/Lon, ISO Time, Radius| Model2["Friend's Model: AIS Vessel Correlator\n(Global Fishing Watch + Haversine)"]
        Model2 -->|Ranked Suspects & Interactive HTML Map| ResponseBuilder["Structured JSON + Base64 Visuals"]
    end
    
    ResponseBuilder -->|Instant Result Payload| Vercel
    Vercel -->|Dual-Pane Visual Intelligence View| User
```

### Why this architecture was chosen:
1. **Model Weight Size**: The PyTorch model (`best_model.pt`) is **383.28 MB**, and deep learning dependencies exceed **2.2 GB**. Vercel Serverless Functions have a maximum unzipped bundle limit of 250 MB.
2. **Satellite Image Payload Size**: Sentinel-1 SAR scenes are **10 MB to 55 MB**, exceeding Vercel's 4.5 MB request body limit.
3. **In-Memory Model Loading**: The Render container loads weights once on startup, providing fast ~4–7 second inference without cold-start model downloads.

---

## 2. Deploying the ML Inference Service to Render (Step 1)

Render hosts custom Docker containers completely for free.

### Step-by-Step Render Deployment:
1. **Push your repository to GitHub**:
   - Create a GitHub repository (e.g. `https://github.com/<your-user>/aegis-sar`) and push your code.
2. **Log in to Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
3. **Create a New Web Service**:
   - Click the **New +** button in the top right and select **Web Service**.
   - Choose **Build and deploy from a Git repository** and connect your GitHub repository.
4. **Configure the Service Settings**:
   - **Name**: `aegis-sar-ml` (or your preferred name)
   - **Region**: Select closest to you (e.g., `Oregon (US West)` or `Frankfurt (EU Central)`)
   - **Branch**: `main` (or `master`)
   - **Root Directory**: Leave blank (uses repository root)
   - **Runtime**: Select **Docker**
   - **Dockerfile Path**: `Dockerfile`
   - **Instance Type**: Select **Free** (or Starter)
5. **Configure Environment Variables**:
   In the **Environment Variables** section, add:
   - `PORT` = `8000`
   - `DEVICE` = `cpu`
   - `DETECTION_THRESHOLD` = `0.50`
   - `MIN_SPILL_PIXELS` = `50`
   - `AIS_SEARCH_RADIUS_KM` = `20.0`
   - `CORS_ORIGINS` = `*`
   - `GFW_API_TOKEN` = `your_token_here` *(optional, built-in fallback provided)*
6. **Health Check Path**:
   - Expand **Advanced** and set **Health Check Path** to `/health`.
7. **Deploy**:
   - Click **Create Web Service**.
   - Render will build the Docker container and deploy it.
   - When the logs show `Application startup complete` and the status badge turns green (**Live**), copy your Render service URL from the top of the dashboard:
     `https://<your-service-name>.onrender.com`

---

## 3. Verifying the Live Render Service (Step 2)

Test the live Render service directly using the test script:
```bash
python scripts/test_remote_service.py --url https://<your-service-name>.onrender.com
```
This tests `GET /health` and runs `POST /predict` on the sample Sentinel-1 image `2018_12_19_d.tif`, confirming that both models and the adapter execute properly on Render.

---

### Option C: Docker Compose / VPS / Cloud Run
Run directly on any Ubuntu/Debian server or VM:
```bash
# Clone repository
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# Start ML Service via Docker Compose
docker compose up -d --build

# Verify Service Health
curl http://localhost:8000/health
```

---

## 3. Deploying the Web Frontend to Vercel (Step 2)

### Option A: Vercel Web Dashboard (Simplest)
1. Push your code to **GitHub**.
2. Log in to [vercel.com](https://vercel.com) and click **Add New > Project**.
3. Import your GitHub repository.
4. **Project Settings**:
   - **Framework Preset**: `Other`
   - **Root Directory**: `./` (or select root)
   - **Output Directory**: `public` (or leave default based on `vercel.json`)
5. Click **Deploy**.
6. Once deployed, open your live Vercel URL (e.g., `https://aegis-sar.vercel.app`).

### Option B: Deploy via Vercel CLI
```bash
npm install -g vercel
vercel login
vercel --prod
```

---

## 4. Connecting Vercel Frontend to the ML Service

Once both are deployed:
1. Open your deployed Vercel website.
2. Click **⚙️ Endpoint Settings** in the top navigation bar.
3. Enter your live ML Service URL (e.g. `https://<your-username>-<your-space-name>.hf.space`).
4. Click **⚡ Test Connection**. When verified (`✓ Connected!`), click **Save & Apply**.
5. The setting is saved in your browser's `localStorage` and will persist for all team members.

---

## 5. Required Environment Variables

| Variable | Scope | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | ML Service | `8000` | HTTP port for FastAPI server |
| `HOST` | ML Service | `0.0.0.0` | Host IP binding |
| `DEVICE` | ML Service | `cpu` | PyTorch inference device (`cpu` or `cuda`) |
| `DETECTION_THRESHOLD` | ML Service | `0.50` | Default detection probability cutoff |
| `MIN_SPILL_PIXELS` | ML Service | `50` | Minimum connected pixel area to filter SAR noise |
| `AIS_SEARCH_RADIUS_KM`| ML Service | `20.0` | Radius in kilometers around spill epicenter for AIS queries |
| `GFW_API_TOKEN` | ML Service | Built-in | Global Fishing Watch API JWT token |
| `CORS_ORIGINS` | ML Service | `*` | Allowed CORS origins for browser access |

---

## 6. How to Test the Live Production System

1. Open the live Vercel web URL.
2. In the left panel, click **Load Gulf of Mexico 2018 Scene Metadata** or select a SAR image (`.tif` or `.png`).
3. Click **⚡ Run Dual-Model Detection Pipeline**.
4. Observe the live 4-stage stepper:
   - *Stage 1: Upload & Preprocessing*
   - *Stage 2: Neural Segmentation (ConvNeXt-Tiny + U-Net)*
   - *Stage 3: Adapter Coordinate Extraction*
   - *Stage 4: AIS Vessel Radar & Trajectory Correlation*
5. Inspect the dual-view results:
   - **Left**: Annotated SAR image with oil spill contours and bounding boxes.
   - **Right**: Embedded interactive AIS maritime map with vessel tracks.
   - **Table**: Ranked suspect vessels sorted by point of closest approach.
6. Click **📄 Full Incident Report (.json)** or **📊 Vessel Rankings (.csv)** to download export dossiers.

---

## 7. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"ML Service: OFFLINE" badge** | Backend container is asleep or unreachable | Check ML Service status on Hugging Face / Render. Verify the URL in **⚙️ Endpoint Settings**. |
| **File upload fails (413)** | Image exceeds 100 MB | Compress or crop GeoTIFF or increase `MAX_FILE_SIZE_BYTES` in `src/api/routes_predict.py`. |
| **CUDA error on modern GPUs** | Local PyTorch build missing sm_120 | System automatically falls back to CPU without crashing. To use GPU, install PyTorch with CUDA 12.4+. |
| **No AIS vessels displayed** | Spill location is in open ocean with no traffic | Try expanding the search radius slider (e.g. 50 km) in metadata settings. |
