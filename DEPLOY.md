# Deploy Sobha Dent

The detector is **YOLOv8 + PyTorch**. That cannot run on Vercel (serverless size/time limits). Split it:

| Piece | Host | Why |
| --- | --- | --- |
| API (`/detect`, `/health`) | **Render** (Docker) | Persistent process, CPU, disk for weights |
| Dashboard | **Vercel** (static) | Cheap CDN UI that posts images to Render |

Local still works as one process: `uvicorn app.main:app --port 8766`.

## 1. Render (backend)

1. New → **Blueprint** (uses `render.yaml`) **or** Web Service from this repo.
2. If not using Blueprint: **Docker**, branch `main`, Dockerfile at repo root.
3. Instance: **Standard** (2 GB) minimum. Free/Starter will often OOM on Torch.
4. Health check: `/health`
5. Env:
   - `SOBHA_DENT_MODEL=yolov8s`
   - `CORS_ORIGINS=https://YOUR-APP.vercel.app` (comma-separated if multiple)
6. First boot downloads `cardd-yolov8s-seg.pt` from Hugging Face (~24 MB) then loads YOLO (1–2 min).
7. Note the URL, e.g. `https://sobha-dent-api.onrender.com`.

Render spin-down on idle: first request after idle can take a minute. Keep the service live or hit `/health` on a cron if you need it always warm.

## 2. Vercel (frontend)

1. Import `aleddotech/sobha-dent`.
2. **Root Directory:** `web`
3. Framework: Other / static.
4. Env:
   - `SOBHA_API_URL=https://sobha-dent-api.onrender.com` (no trailing slash)
5. Deploy. `web/vercel.json` writes `config.js` at build time.

After the Vercel URL exists, put it in Render `CORS_ORIGINS` and **redeploy the API**.

## 3. Optional: UI only on Render

Skip Vercel and open the Render URL directly. FastAPI still serves the dashboard from `/`. Then `CORS_ORIGINS` can stay `*`.
