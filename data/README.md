# Datasets (gitignored)

| Path | Role |
| --- | --- |
| `raw/` | Roboflow YOLO exports (train + their val/test) |
| `yolo/` | Merged 2-class `dent` / `scratch` set used for training |
| `val_realworld/` | Unlabeled photos from `Downloads/car damage` — qualitative eval only |

Train downloads need `ROBOFLOW_API_KEY`. See `scripts/download_roboflow.py`.
