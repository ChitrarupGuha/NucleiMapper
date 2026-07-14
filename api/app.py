"""
app.py - FastAPI backend for NucleiMapper AI.

This is the "Python Backend" step in:
    Website -> Upload Image -> Python Backend -> Model -> Prediction -> Result

Run from the backend/ directory (so `common` and `api` resolve as packages):

    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Env vars (all optional, defaults assume you trained with the defaults in
train.py and left checkpoints in ./checkpoints):

    CHECKPOINT_PATH   default ./checkpoints/best_model.pt
    CLASS_NAMES_PATH  default ./checkpoints/class_names.json
    MODEL_ARCH        default resnet18
"""
import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.inference import Predictor

CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "./checkpoints/best_model.pt")
CLASS_NAMES = os.environ.get("CLASS_NAMES_PATH", "./checkpoints/class_names.json")
ARCH = os.environ.get("MODEL_ARCH", "resnet18")

app = FastAPI(title="NucleiMapper AI Backend")

# Dev-friendly CORS so the standalone HTML frontend (often opened from
# file:// or a different port) can call this API. Tighten allow_origins
# before exposing this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[Predictor] = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = Predictor(CHECKPOINT, CLASS_NAMES, arch=ARCH)
        print(f"Model loaded: arch={ARCH} classes={predictor.class_names}")
    except FileNotFoundError as e:
        print(f"WARNING: could not load model ({e}). /predict will fail until "
              f"a checkpoint exists at {CHECKPOINT}. Train one with train.py first.")


@app.get("/health")
def health():
    return {
        "status": "ok" if predictor is not None else "model_not_loaded",
        "classes": predictor.class_names if predictor else [],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Train it first (see train.py).")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Could not read image file")
    return predictor.predict(image)
