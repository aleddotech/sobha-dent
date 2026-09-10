from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MODELS = ROOT / "models"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOCUS = {"dent", "scratch"}
COLORS = {
    "dent": (232, 93, 62),
    "scratch": (47, 128, 237),
    "crack": (242, 201, 76),
    "glass_shatter": (111, 207, 151),
    "lamp_broken": (155, 81, 224),
    "tire_flat": (130, 130, 130),
}

_model: YOLO | None = None
_weight_path: Path | None = None


def cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def weight_file() -> Path:
    choice = os.environ.get("SOBHA_DENT_MODEL", "yolov8s").lower()
    if choice == "yolo11":
        p = MODELS / "cardd-yolo11x-seg.pt"
    else:
        p = MODELS / "cardd-yolov8s-seg.pt"
    if not p.exists():
        from download_weights import download

        p = download(choice)
    return p


def get_model() -> YOLO:
    global _model, _weight_path
    path = weight_file()
    if _model is None or _weight_path != path:
        _model = YOLO(str(path))
        _weight_path = path
    return _model


app = FastAPI(title="Sobha Dent")
origins = cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    path = weight_file()
    return {
        "ok": True,
        "weights": path.name,
        "model": os.environ.get("SOBHA_DENT_MODEL", "yolov8s"),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.25,
    focus_only: bool = False,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc

    model = get_model()
    results = model.predict(
        source=np.array(image),
        conf=conf,
        verbose=False,
        imgsz=640,
    )
    r = results[0]
    names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}

    detections = []
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, score, cid in zip(xyxy, confs, clss):
            label = str(names.get(int(cid), cid)).replace(" ", "_")
            if focus_only and label not in FOCUS:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                {
                    "label": label,
                    "confidence": float(score),
                    "box": [x1, y1, x2, y2],
                    "color": COLORS.get(label, (200, 200, 200)),
                }
            )

    plotted = r.plot()
    annotated = Image.fromarray(plotted[..., ::-1])
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    w, h = image.size
    dents = sum(1 for d in detections if d["label"] == "dent")
    scratches = sum(1 for d in detections if d["label"] == "scratch")

    return JSONResponse(
        {
            "width": w,
            "height": h,
            "weights": weight_file().name,
            "count": len(detections),
            "dents": dents,
            "scratches": scratches,
            "detections": detections,
            "annotated_jpeg_b64": b64,
        }
    )
