"""Start with: python -m uvicorn api:app --host 0.0.0.0 --port 8000"""

import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from inference import predict

PROJECT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("PATHSNAP_DATA_ROOT", PROJECT / "Student folder")).resolve()
PARTITION = os.getenv("PATHSNAP_PARTITION", "student_test")
MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL_PATH", PROJECT / "Student folder" / "pathsnap_outputs" /
                            "tuned_random_forest_map_matching_model.joblib"))

app = FastAPI(title="PathSnap evaluation API")
ui_origins = [origin.strip() for origin in os.getenv("PATHSNAP_UI_ORIGINS", "").split(",") if origin.strip()]
if ui_origins:
    app.add_middleware(CORSMiddleware, allow_origins=ui_origins, allow_methods=["GET", "POST"])


def available_trips():
    if not re.fullmatch(r"[A-Za-z0-9_-]+", PARTITION):
        raise HTTPException(503, "Invalid partition configuration")
    root = DATA_ROOT / PARTITION
    if not root.is_dir():
        raise HTTPException(503, f"Trip partition is unavailable: {root}")
    return {p.name: p for p in root.iterdir() if p.is_dir() and
            all((p / f"{p.name}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))}


@lru_cache(maxsize=1)
def loaded_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model has not been exported: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support candidate probabilities")
    if hasattr(model, "n_features_in_") and model.n_features_in_ != 8:
        raise ValueError("Model does not match the notebook's eight features")
    return model


@lru_cache(maxsize=10)
def predict_cached(trip_id: str):
    folder = available_trips().get(trip_id)
    if folder is None:
        raise HTTPException(404, "Unknown trip ID")
    return predict(folder, loaded_model())


@app.get("/", include_in_schema=False)
async def home():
    return HTMLResponse((PROJECT / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/static/styles.css", include_in_schema=False)
@app.get("/styles.css", include_in_schema=False)
async def styles():
    return Response((PROJECT / "static" / "styles.css").read_text(encoding="utf-8"), media_type="text/css")


@app.get("/static/app.js", include_in_schema=False)
@app.get("/app.js", include_in_schema=False)
async def script():
    return Response((PROJECT / "static" / "app.js").read_text(encoding="utf-8"), media_type="text/javascript")


@app.get("/api/health")
async def health():
    try:
        loaded_model()
        count = len(available_trips())
    except (FileNotFoundError, OSError, ValueError, HTTPException) as error:
        raise HTTPException(503, str(error)) from error
    return {"status": "ready", "model": MODEL_PATH.name,
            "partition": PARTITION, "available_trips": count}


@app.get("/api/trips")
async def trips():
    return {"trips": sorted(available_trips())}


@app.get("/api/predict/{trip_id}")
@app.post("/api/predict/{trip_id}")
async def predict_endpoint(trip_id: str):
    try:
        result = predict_cached(trip_id)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(503, str(error)) from error
    print(f"Inference request {trip_id}: {result['metrics']['observations']} observations", flush=True)
    return result


@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
async def map_endpoint(trip_id: str):
    try:
        from folium_map import render_trip_map
        result = predict_cached(trip_id)
        return HTMLResponse(render_trip_map(result))
    except ImportError as error:
        raise HTTPException(503, "Folium is missing. Install requirements-api.txt on Laptop 1.") from error
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(503, str(error)) from error
