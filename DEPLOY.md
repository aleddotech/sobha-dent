# Deploy Sobha Dent (Gemini)

YOLO is no longer required on Render. The API calls Gemini and returns n-point polygons.

## Render env

**Dashboard → service `sobha-dent-api` → Environment → Add Environment Variable**

| Key | Value |
| --- | --- |
| `GOOGLE_API_KEY` | from Google AI Studio / Gemini |
| `GEMINI_MODEL` | `gemini-3.7-flash` |
| `CORS_ORIGINS` | `https://sobha-dent.vercel.app` |

Save, then **Manual Deploy**. Do not put the key in the repo.

If the service still uses the old Torch Docker image, trigger a deploy from latest `main` so it picks up the slim Dockerfile.
