#!/usr/bin/env bash
# Fine-tune YOLOv8n on this Mac with a small memory/CPU footprint.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

YAML="$ROOT/data/yolo/dataset.yaml"
test -f "$YAML" || { echo "run scripts/prepare_dataset.py first"; exit 1; }

# nano + 416px + batch 2 is the realistic 8GB-class recipe.
# v8s/640 will swap on 8GB. MPS is used if available.
exec yolo detect train \
  model=yolov8n.pt \
  data="$YAML" \
  imgsz=416 \
  batch=2 \
  epochs=40 \
  workers=0 \
  device=mps \
  cache=false \
  amp=true \
  patience=12 \
  project="$ROOT/runs" \
  name=dent-n-lowmem \
  exist_ok=true
