from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.gemini_detect import annotate, detect_image

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WEB = ROOT / "web"
STATIC = ROOT / "app" / "static"
UI = WEB if (WEB / "index.html").exists() else STATIC


def cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Sobha Dent")
origins = cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if UI.exists():
    app.mount("/static", StaticFiles(directory=UI), name="static")


@app.get("/")
def index():
    return FileResponse(UI / "index.html")


@app.get("/config.js")
def config_js():
    path = UI / "config.js"
    if path.exists():
        return FileResponse(path, media_type="application/javascript")
    return Response("window.SOBHA_API = '';\n", media_type="application/javascript")


@app.get("/health")
def health():
    return {
        "ok": True,
        "backend": "gemini",
        "model": os.environ.get("GEMINI_MODEL", "gemini-3.7-flash"),
        "has_key": bool(os.environ.get("GOOGLE_API_KEY", "").strip()),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    focus_only: bool = True,
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

    try:
        detections, raw_text = detect_image(image, focus_only=focus_only)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Gemini detect failed: {exc}") from exc

    annotated = annotate(image, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)

    payload = []
    for d in detections:
        payload.append(
            {
                "label": d.label,
                "confidence": d.confidence,
                "box": d.box,
                "polygon": [[float(x), float(y)] for x, y in d.polygon],
                "n_points": d.n_points,
                "area_frac": d.area_frac,
                "color": d.color,
            }
        )

    w, h = image.size
    return JSONResponse(
        {
            "width": w,
            "height": h,
            "backend": "gemini",
            "model": os.environ.get("GEMINI_MODEL", "gemini-3.7-flash"),
            "count": len(payload),
            "dents": sum(1 for d in payload if d["label"] == "dent"),
            "scratches": sum(1 for d in payload if d["label"] == "scratch"),
            "detections": payload,
            "annotated_jpeg_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "raw_model_text": raw_text[:4000],
        }
    )
