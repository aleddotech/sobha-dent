#!/usr/bin/env python3
"""Download public Roboflow dent/scratch datasets as YOLOv8 zips."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PROJECTS = [
    ("damgesmarceedes", "only-dents-scratches"),
    ("damgesmarceedes", "final-final-dents-scratches"),
    ("ankit-tiwari", "car-dents-xpp3w"),
    ("satish-satpal-ibrpv", "dents-8vbvd"),
    ("dar-vir4t", "car-5nip6"),
]


def main() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit(
            "Set ROBOFLOW_API_KEY (https://app.roboflow.com/settings/api) then rerun:\n"
            "  conda activate sobha-dent\n"
            "  pip install roboflow\n"
            "  export ROBOFLOW_API_KEY=...\n"
            "  python scripts/download_roboflow.py"
        )

    from roboflow import Roboflow

    RAW.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    for workspace, slug in PROJECTS:
        dest = RAW / slug
        if dest.exists() and any(dest.rglob("*.txt")):
            print(f"skip existing {slug}")
            continue
        print(f"download {workspace}/{slug} ...")
        project = rf.workspace(workspace).project(slug)
        versions = project.versions()
        if not versions:
            print(f"  no versions: {slug}")
            continue
        ver = versions[0]
        vid = int(getattr(ver, "version", None) or str(ver).split()[-1])
        location = str(dest)
        try:
            project.version(vid).download("yolov8", location=location)
        except Exception:
            project.version(vid).download("yolov5", location=location)
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
