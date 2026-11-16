import json, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from PIL import Image, UnidentifiedImageError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Analysis
from .services import assess_claim, classify_image, search_sources
from .schemas import DISCLAIMER

try:
    # create_all creates missing tables without dropping or rewriting existing data.
    Base.metadata.create_all(bind=engine)
except SQLAlchemyError as exc:
    raise RuntimeError(
        "Database initialization failed. Check DATABASE_URL connectivity and database permissions."
    ) from exc
app = FastAPI(title="VERITAS AI API", version="1.0.0")
allowed_origins = {
    "http://localhost:3000",
    "https://hackathon1-9s6d-ih8wk8vod-idkme.vercel.app",
}
allowed_origins.update(
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    return JSONResponse(
        status_code=503,
        content={"detail": "The database is temporarily unavailable. Please retry shortly."},
    )

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


def packed(row):
    return {"id": row.id, "analysis_type": row.analysis_type, "original_filename": row.original_filename,
            "verdict": row.verdict, "manipulation_likelihood": row.manipulation_likelihood,
            "model_confidence": row.model_confidence, "evidence_coverage": row.evidence_coverage,
            "summary": row.summary, "findings": json.loads(row.findings_json), "sources": json.loads(row.sources_json),
            "limitations": json.loads(row.limitations_json), "created_at": row.created_at}

def save(db, kind, filename, verdict, summary, findings=None, sources=None, limitations=None, likelihood=None, confidence=None, coverage="not_available"):
    row = Analysis(id=str(uuid.uuid4()), analysis_type=kind, original_filename=filename, verdict=verdict,
        manipulation_likelihood=likelihood, model_confidence=confidence, evidence_coverage=coverage, summary=summary,
        findings_json=json.dumps(findings or []), sources_json=json.dumps(sources or []),
        limitations_json=json.dumps((limitations or []) + [DISCLAIMER]), created_at=datetime.now(timezone.utc))
    db.add(row); db.commit(); db.refresh(row); return row

@app.get("/api/health")
def health():
    return {"status":"ok", "models":{"image":bool(os.getenv("HF_API_TOKEN") and os.getenv("HF_IMAGE_MODEL")),"video":False,"audio":False},
            "fact_search":bool(os.getenv("GOOGLE_CSE_API_KEY") and os.getenv("GOOGLE_CSE_CX")),"fact_review":bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/api/fact-check")
async def fact_check(payload: dict, db: Session=Depends(get_db)):
    claim=str(payload.get("claim", "")).strip()
    if len(claim)<8 or len(claim)>5000: raise HTTPException(400,"Enter a claim between 8 and 5,000 characters.")
    try:
        sources=await search_sources(claim); assessment=await assess_claim(claim,sources)
    except Exception:
        sources=[]; assessment={"verdict":"insufficient_evidence","explanation":"Live verification could not be completed. Check provider configuration and try again.","findings":[],"limitations":["The configured provider returned an error."]}
    findings=[{"type":"claim","text":claim}, *[{"type":"evidence","text":x} for x in assessment.get("findings",[])] ]
    row=save(db,"fact_check",None,assessment["verdict"],assessment.get("explanation",""),findings,sources,assessment.get("limitations",[]),coverage="search_snippets" if sources else "none")
    return {**packed(row),"claim":claim}

async def read_upload(file, allowed):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in allowed: raise HTTPException(415,f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}")
    data=await file.read(MAX_BYTES+1)
    if not data: raise HTTPException(400,"The uploaded file is empty.")
    if len(data)>MAX_BYTES: raise HTTPException(413,f"File exceeds the {MAX_BYTES//(1024*1024)} MB upload limit.")
    return data

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile=File(...), db: Session=Depends(get_db)):
    data=await read_upload(file,{".jpg",".jpeg",".png",".webp"})
    try:
        image=Image.open(BytesIO(data)); image.verify()
        with Image.open(BytesIO(data)) as img: width,height,fmt=img.width,img.height,img.format
    except (UnidentifiedImageError,OSError,ValueError): raise HTTPException(400,"The uploaded file is not a readable image.")
    findings=[{"type":"metadata","text":f"Readable {fmt} image, {width} × {height} pixels. Metadata alone does not establish authenticity."}]
    configured=bool(os.getenv("HF_API_TOKEN") and os.getenv("HF_IMAGE_MODEL"))
    model_failed=False
    try: result=await classify_image(data)
    except Exception: result=None; model_failed=True
    if result:
        scores=result["scores"]
        findings.append({"type":"model","text":f"Classifier {result['model']} returned labels: "+", ".join(f"{x.get('label','unknown')} ({x.get('score',0):.3f})" for x in scores[:5])})
        suspect=next((x for x in scores if any(w in x.get("label","").lower() for w in ("fake","synthetic","ai-generated","artificial"))),None)
        likelihood=suspect.get("score") if suspect else None
        confidence=max((x.get("score",0) for x in scores),default=None)
        verdict="potentially_manipulated" if likelihood is not None and likelihood>=0.65 else "inconclusive"
        summary="Classifier output is only an indicator; labels and calibration depend on the configured model."
        limits=["This classifier may not detect edits or generated images reliably. Review independently."]
        coverage="whole_image_classifier"
    else:
        likelihood=confidence=None; verdict="inconclusive"
        summary="Image received and validated, but no authenticity assessment was performed. " + ("The configured model service did not return a usable result." if configured else "No image detection model is configured.")
        limits=["Check the configured model ID, token, and inference service." if model_failed else "Configure HF_API_TOKEN and HF_IMAGE_MODEL to enable model inference."]
        coverage="metadata_only"
    row=save(db,"image",Path(file.filename or "image").name,verdict,summary,findings,limitations=limits,likelihood=likelihood,confidence=confidence,coverage=coverage)
    return packed(row)

@app.post("/api/analyze/{kind}")
async def unavailable_media(kind: str,file: UploadFile=File(...),db: Session=Depends(get_db)):
    types={"video":{".mp4",".mov",".avi",".webm"},"audio":{".wav",".mp3",".m4a",".flac"}}
    if kind not in types: raise HTTPException(404,"Unknown analysis type.")
    await read_upload(file,types[kind]); label=kind.title()
    row=save(db,kind,Path(file.filename or kind).name,"inconclusive",f"{label} upload validated; a compatible analysis model is not configured, so no detection was performed.",limitations=[f"{label} detection is unavailable in this prototype."],coverage="none")
    return packed(row)

@app.get("/api/history")
def history(db: Session=Depends(get_db)):
    return [packed(x) for x in db.query(Analysis).order_by(Analysis.created_at.desc()).limit(100).all()]

@app.get("/api/history/{analysis_id}")
def get_analysis(analysis_id: str,db: Session=Depends(get_db)):
    row=db.get(Analysis,analysis_id)
    if not row: raise HTTPException(404,"Analysis not found.")
    return packed(row)

@app.delete("/api/history/{analysis_id}")
def delete_analysis(analysis_id: str,db: Session=Depends(get_db)):
    row=db.get(Analysis,analysis_id)
    if not row: raise HTTPException(404,"Analysis not found.")
    db.delete(row); db.commit(); return {"deleted":True}

@app.get("/api/report/{analysis_id}")
def report(analysis_id: str,db: Session=Depends(get_db)):
    row=db.get(Analysis,analysis_id)
    if not row: raise HTTPException(404,"Analysis not found.")
    buf=BytesIO(); pdf=canvas.Canvas(buf,pagesize=letter); y=750
    def line(text):
        nonlocal y
        if y<60: pdf.showPage(); y=750
        pdf.drawString(50,y,str(text).encode("latin-1","replace").decode("latin-1")[:110]); y-=18
    line("VERITAS AI - Verification report"); line(f"ID: {row.id}"); line(f"Type: {row.analysis_type}"); line(f"Verdict: {row.verdict}")
    line(f"Created: {row.created_at.isoformat()}"); line(f"Summary: {row.summary}")
    for item in json.loads(row.findings_json): line("Finding: "+str(item.get("text",item)))
    for item in json.loads(row.sources_json): line(f"Source: {item.get('title')} - {item.get('url')}")
    for item in json.loads(row.limitations_json): line("Limitation: "+str(item))
    pdf.save(); return Response(buf.getvalue(),media_type="application/pdf",headers={"Content-Disposition":f'attachment; filename="veritas-{row.id[:8]}.pdf"'})



