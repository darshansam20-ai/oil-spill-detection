"""
FastAPI Application Entrypoint for Automated SAR Oil-Spill Detection System.
Provides RESTful APIs, OpenAPI documentation, and serves the interactive geospatial dashboard.
"""
import os
from pathlib import Path
import torch
try:
    torch.set_num_threads(1)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config.settings import settings
from src.api.routes_predict import router as predict_router
from src.api.routes_scenes import router as scenes_router
from src.api.routes_pipeline import router as pipeline_router
from src.api.routes_events import router as events_router
from src.api.routes_alerts import router as alerts_router
from src.api.routes_reports import router as reports_router
from src.api.routes_system import router as system_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Integrated Sentinel-1 SAR Deep Learning Oil Spill Detection & AIS Maritime Vessel Tracking Production System.",
)

# Enable unrestricted CORS for web clients (including Vercel deployed frontend)
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_origins_env.split(",")] if cors_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(predict_router)
app.include_router(scenes_router)
app.include_router(pipeline_router)
app.include_router(events_router)
app.include_router(alerts_router)
app.include_router(reports_router)
app.include_router(system_router)

# Mount Dashboard Static Files
dashboard_static_dir = Path(__file__).parent.parent / "dashboard" / "static"
dashboard_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(dashboard_static_dir)), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_dashboard():
    """Serve the interactive geospatial dashboard index page."""
    index_path = dashboard_static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "SAR Oil-Spill Detection API is running.", "docs": "/docs"}
