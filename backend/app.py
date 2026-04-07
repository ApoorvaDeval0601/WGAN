"""
FastAPI Backend — WGAN Lab 5
Serves: generated images, training loss log, on-demand generation
"""

import os
import json
import io
import base64
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Paths (all relative to this file's location) ────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "model_output")
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
LOG_FILE     = os.path.join(SAVE_DIR, "training_log.json")
WEIGHTS      = os.path.join(SAVE_DIR, "generator_final.pth")
NZ           = 100
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model_output dir if it doesn't exist yet
os.makedirs(SAVE_DIR, exist_ok=True)

# ── App ──────────────────────────────────────────────
app = FastAPI(title="WGAN Lab 5 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve epoch images
app.mount("/images", StaticFiles(directory=SAVE_DIR), name="images")

# Serve frontend at /ui
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ── Generator ────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ,  512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,   3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


_generator = None

def get_generator():
    global _generator
    if _generator is None:
        if not os.path.exists(WEIGHTS):
            return None
        model = Generator().to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
        model.eval()
        _generator = model
    return _generator


# ── Routes ───────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "WGAN Lab 5 API running. Open /ui for the dashboard."}


@app.get("/api/log")
def get_log():
    if not os.path.exists(LOG_FILE):
        return {"log": [], "message": "Training not started yet."}
    with open(LOG_FILE) as f:
        log = json.load(f)
    return {"log": log}


@app.get("/api/epochs")
def list_epochs():
    files = sorted([
        f for f in os.listdir(SAVE_DIR)
        if f.startswith("epoch_") and f.endswith(".png")
    ])
    return {"epochs": files}


@app.get("/api/epoch/{epoch_num}")
def get_epoch_image(epoch_num: int):
    filename = f"epoch_{epoch_num:03d}.png"
    filepath = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Epoch {epoch_num} image not found.")
    return FileResponse(filepath, media_type="image/png")


@app.get("/api/generate")
def generate_images(n: int = 16):
    if n > 64:
        raise HTTPException(status_code=400, detail="Max 64 images per request.")
    gen = get_generator()
    if gen is None:
        raise HTTPException(status_code=503, detail="Generator weights not found. Train the model first.")

    from PIL import Image
    import numpy as np

    with torch.no_grad():
        noise  = torch.randn(n, NZ, 1, 1, device=DEVICE)
        images = gen(noise)
        images = (images * 0.5 + 0.5).clamp(0, 1)
        images = (images * 255).byte().cpu().numpy()

    encoded = []
    for img in images:
        arr = np.transpose(img, (1, 2, 0))
        pil = Image.fromarray(arr, "RGB").resize((128, 128), Image.NEAREST)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode())

    return {"images": encoded, "count": n, "device": str(DEVICE)}


@app.get("/api/status")
def status():
    trained     = os.path.exists(WEIGHTS)
    epochs_done = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            epochs_done = len(json.load(f))
    return {
        "trained":      trained,
        "epochs_done":  epochs_done,
        "device":       str(DEVICE),
        "gpu":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }