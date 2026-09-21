#!/usr/bin/env python3
"""Rank val_realworld photos by smallest YOLO dent box, then confirm with Gemini."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "val_realworld" / "images"
OUT = ROOT / "data" / "examples" / "bare_minimum_dent"
WEIGHTS = ROOT / "models" / "cardd-yolov8s-seg.pt"


def yolo_rank() -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    rows = []
    for img in sorted(SRC.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        r = model.predict(source=str(img), conf=0.15, verbose=False, imgsz=640)[0]
        names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
        w, h = Image.open(img).size
        img_area = float(w * h) or 1.0
        if r.boxes is None or not len(r.boxes):
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        dents = []
        for box, score, cid in zip(xyxy, confs, clss):
            label = str(names.get(int(cid), cid)).replace(" ", "_")
            if label != "dent":
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            dents.append({"box": [x1, y1, x2, y2], "conf": float(score), "area_frac": area / img_area})
        if not dents:
            continue
        smallest = min(dents, key=lambda d: d["area_frac"])
        rows.append(
            {
                "file": img.name,
                "path": str(img),
                "width": w,
                "height": h,
                "n_dents": len(dents),
                **smallest,
            }
        )
    rows.sort(key=lambda r: r["area_frac"])
    return rows


def gemini_confirm(paths: list[Path]) -> list[dict]:
    from app.gemini_detect import detect_image

    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        dets, raw = detect_image(im, focus_only=True)
        dents = [d for d in dets if d.label == "dent"]
        smallest = min(dents, key=lambda d: d.area_frac) if dents else None
        out.append(
            {
                "file": p.name,
                "gemini_n": len(dets),
                "gemini_dents": len(dents),
                "gemini_min_area_frac": smallest.area_frac if smallest else None,
                "gemini_n_points": smallest.n_points if smallest else None,
            }
        )
        print(p.name, "gemini", out[-1])
    return out


def main() -> None:
    if not SRC.exists() or not any(SRC.iterdir()):
        sys.exit(f"no images in {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = yolo_rank()
    (OUT / "yolo_rank.json").write_text(json.dumps(rows, indent=2))
    print(f"YOLO dent hits: {len(rows)}")
    top = rows[:8]
    for i, r in enumerate(top, 1):
        src = Path(r["path"])
        dest = OUT / f"{i:02d}_area{r['area_frac']*100:.3f}_{src.name}"
        shutil.copy2(src, dest)
        print(f"{i:02d} {r['area_frac']*100:.3f}% conf={r['conf']:.2f} {src.name}")
    confirm = os.environ.get("GEMINI_CONFIRM", "1") != "0"
    if confirm and top:
        gemini_confirm([Path(r["path"]) for r in top[:5]])


if __name__ == "__main__":
    main()
