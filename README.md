# Sobha Dent

Gemini polygon demo for vehicle **dent / scratch** detection. Drag-and-drop a photo; Gemini returns an n-point outline (not just a rectangle).

## Local

```bash
conda activate sobha-dent
pip install -r requirements.txt
# put GOOGLE_API_KEY in .env
uvicorn app.main:app --host 127.0.0.1 --port 8766
```

Open http://127.0.0.1:8766

## Render

Dashboard → **sobha-dent-api** → **Environment** → add:

| Key | Value |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini API key |
| `GEMINI_MODEL` | `gemini-3.7-flash` (optional) |
| `CORS_ORIGINS` | `https://sobha-dent.vercel.app` |

Redeploy after saving. The Docker image no longer needs Torch.

## Vercel

Unchanged: root `web`, `SOBHA_API_URL` = Render URL.
