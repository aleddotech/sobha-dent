#!/usr/bin/env python3
"""Download CarDD YOLO weights into models/."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

CHOICE = os.environ.get("SOBHA_DENT_MODEL", "yolov8s").lower()

SPECS = {
    "yolov8s": {
        "repo_id": "abdullahg7/cardd-yolov8s",
        "filename": "v2.0/best.pt",
        "dest": MODELS / "cardd-yolov8s-seg.pt",
    },
    "yolo11": {
        "repo_id": "harpreetsahota/car-dd-segmentation-yolov11",
        "filename": "best.pt",
        "dest": MODELS / "cardd-yolo11x-seg.pt",
    },
}


def download(choice: str) -> Path:
    spec = SPECS.get(choice) or SPECS["yolov8s"]
    dest = spec["dest"]
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"already present: {dest}")
        return dest
    print(f"downloading {spec['repo_id']} {spec['filename']} ...")
    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        local_dir=MODELS / "_hf",
    )
    dest.write_bytes(Path(path).read_bytes())
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    download(CHOICE)
