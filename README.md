# VERITAS AI

VERITAS AI is a local-first verification workbench for claims and media. It is explicit about unavailable models and never substitutes fabricated scores or citations. The prototype includes live web retrieval for claims when configured, Gemini image understanding, SQLite history, report downloads, and Gemini-based video/audio observations.

## Requirements

- Node.js 20+
- Python 3.10+
- Optional: a Tavily API key for web evidence retrieval and a Gemini API key for fact-check and media analysis.

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

See `backend/.env.example`. Fact checks retrieve up to five web results from Tavily using `TAVILY_API_KEY` (basic search only). Gemini, configured with `GEMINI_API_KEY` and `GEMINI_MODEL`, reviews the retrieved snippets and returns a structured verdict. Image, video, and audio observations use Gemini; image analysis uses `GEMINI_IMAGE_MODEL`. Media findings remain inconclusive and include no model-generated confidence score. Retrieval errors or empty results return insufficient evidence; Gemini review failures return `review_unavailable` with no factual verdict. Neither case substitutes generated citations. Provider quotas and billing settings are controlled by those providers.

The SQLite database is created at `backend/data/veritas.db`. Uploads are handled as temporary request data and are not retained after processing; the multipart framework may spool larger uploads to its temporary directory. Only analysis metadata, findings, and retrieved source snippets are persisted. History deletion removes the stored record. Reports are generated from stored analysis data.

## API

- `GET /api/health`
- `POST /api/fact-check` (`{ "claim": "..." }`)
- `POST /api/analyze/image` (multipart `file`)
- `POST /api/analyze/video` and `/api/analyze/audio` (multipart `file`)
- `GET /api/history`, `GET /api/history/{analysis_id}`, `DELETE /api/history/{analysis_id}`
- `GET /api/report/{analysis_id}` (PDF)

## Limitations

Model observations are probabilistic and can be wrong. Source retrieval does not establish truth by itself. Results should be reviewed against the linked sources, especially for consequential decisions. Gemini media observations do not prove authenticity or manipulation.
