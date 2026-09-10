# Sobha Dent

POC dashboard for vehicle **dent / scratch** detection. Drag-and-drop a photo; a CarDD-trained YOLO model draws bounding boxes (and masks when available).

## Model

Public SOTA-style weights on the [CarDD](https://cardd-ustc.github.io/) dataset (dents, scratches, cracks, glass, lamps, tires):

| Model | Source | Why |
| --- | --- | --- |
| **YOLOv8s-seg (default)** | [`abdullahg7/cardd-yolov8s`](https://huggingface.co/abdullahg7/cardd-yolov8s) `v2.0/best.pt` | Strong published CarDD mAP (~75.8% mAP50), dent 60.6% / scratch 63.2% box mAP50, easy Ultralytics load |
| YOLOv11x-seg | [`harpreetsahota/car-dd-segmentation-yolov11`](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11) | Larger YOLO11-seg baseline; slower, similar classes |

Classes: `dent`, `scratch`, `crack`, `glass_shatter`, `lamp_broken`, `tire_flat`.

## Local

```bash
conda env create -f environment.yml
conda activate sobha-dent
python download_weights.py
uvicorn app.main:app --host 127.0.0.1 --port 8766
```

Open http://127.0.0.1:8766 (do not use 8765 if Sobha Voice is bound there).

## Deploy

YOLO does not run on Vercel. **API → Render**, **UI → Vercel**. See [DEPLOY.md](DEPLOY.md).
