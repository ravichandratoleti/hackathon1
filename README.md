# VERITAS AI

VERITAS AI is a local-first verification workbench for claims and media. It is explicit about unavailable models and never substitutes fabricated scores or citations. The prototype includes live web retrieval for claims when configured, optional Hugging Face image classification, SQLite history, report downloads, and clearly unavailable video/audio analysis until a suitable detector is configured.

## Requirements

- Node.js 20+
- Python 3.10+
- Optional: Google Custom Search credentials for live fact-check source retrieval; a Hugging Face Inference API token and image classifier model ID for image analysis.

## Run on Windows

1. Copy `backend/.env.example` to `backend/.env` and fill in optional credentials.
2. In a PowerShell terminal:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

3. In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Frontend API requests use `FRONTEND_ORIGIN` as their base URL when set, and default to the current origin otherwise. Next.js proxies `/api` to `http://localhost:8000` by default; set `BACKEND_URL` before starting Next.js to change the backend URL. If you set `FRONTEND_ORIGIN`, use the frontend's origin (for example, `http://localhost:3000`) and restart Next.js.

## Configuration

See `backend/.env.example`. Fact-check retrieval uses Google Custom Search when both credentials are set. Without them, the result says live verification is unavailable. Even when sources are retrieved, verdict adjudication is only enabled when `OPENAI_API_KEY` is set; otherwise the sources are shown with an insufficient-evidence verdict. Image analysis requires a configured Hugging Face image classification model and token. It reports unavailable when either is missing. Video and audio endpoints are explicit unavailable states in this prototype.

The SQLite database is created at `backend/data/veritas.db`. Uploads are handled as temporary request data and are not retained after processing; the multipart framework may spool larger uploads to its temporary directory. Only analysis metadata, findings, and retrieved source snippets are persisted. History deletion removes the stored record. Reports are generated from stored analysis data.

## API

- `GET /api/health`
- `POST /api/fact-check` (`{ "claim": "..." }`)
- `POST /api/analyze/image` (multipart `file`)
- `POST /api/analyze/video` and `/api/analyze/audio` (multipart `file`)
- `GET /api/history`, `GET /api/history/{analysis_id}`, `DELETE /api/history/{analysis_id}`
- `GET /api/report/{analysis_id}` (PDF)

## Limitations

Detector output is probabilistic and can be wrong. Source retrieval does not establish truth by itself. Results should be reviewed against the linked sources, especially for consequential decisions. Video and audio detection require compatible, validated models and are not represented as operational here.
