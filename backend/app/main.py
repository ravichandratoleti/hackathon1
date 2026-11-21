import json, os, uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from PIL import ExifTags, Image, UnidentifiedImageError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Analysis
from .services import EvidenceRetrievalError, analyze_image_gemini, analyze_media_gemini, assess_claim, gemini_image_configured, gemini_media_configured, search_provider_configured, search_sources
from .schemas import DISCLAIMER

logger = logging.getLogger(__name__)

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
    gemini_ready=gemini_media_configured()
    return {"status":"ok", "models":{"image":gemini_image_configured(),"video":gemini_ready,"audio":gemini_ready},
            "fact_search":search_provider_configured(),"fact_review":gemini_media_configured()}

@app.post("/api/fact-check")
async def fact_check(payload: dict, db: Session=Depends(get_db)):
    claim=str(payload.get("claim", "")).strip()
    if len(claim)<8 or len(claim)>5000: raise HTTPException(400,"Enter a claim between 8 and 5,000 characters.")
    request_id=uuid.uuid4().hex[:12]
    logger.info("Fact-check request started request_id=%s claim_length=%d", request_id, len(claim))
    try:
        sources=await search_sources(claim)
    except EvidenceRetrievalError as exc:
        logger.warning("Fact-check evidence retrieval unavailable request_id=%s error_type=%s", request_id, type(exc).__name__)
        sources=[]
        assessment={"verdict":"insufficient_evidence","explanation":str(exc),"findings":[],"limitations":["No source evidence was retrieved, so no factual verdict was generated."]}
    except Exception as exc:
        logger.error("Unexpected evidence retrieval failure request_id=%s error_type=%s", request_id, type(exc).__name__)
        sources=[]
        assessment={"verdict":"insufficient_evidence","explanation":"Evidence retrieval is unavailable because the search provider returned an unexpected error.","findings":[],"limitations":["No source evidence was retrieved, so no factual verdict was generated."]}
    else:
        logger.info("Fact-check evidence retrieval completed request_id=%s source_count=%d", request_id, len(sources))
        if not sources:
            assessment=await assess_claim(claim,[])
        else:
            assessment=await assess_claim(claim,sources)
    findings=[{"type":"claim","text":claim}, *[{"type":"evidence","text":x} for x in assessment.get("findings",[])] ]
    row=save(db,"fact_check",None,assessment["verdict"],assessment.get("explanation",""),findings,sources,assessment.get("limitations",[]),confidence=assessment.get("model_confidence"),coverage="search_snippets" if sources else "none")
    logger.info("Fact-check request completed request_id=%s report_id=%s verdict=%s confidence_available=%s source_count=%d", request_id, row.id, row.verdict, row.model_confidence is not None, len(sources))
    return {**packed(row),"claim":claim}

async def read_upload(file, allowed):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in allowed: raise HTTPException(415,f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}")
    data=await file.read(MAX_BYTES+1)
    if not data: raise HTTPException(400,"The uploaded file is empty.")
    if len(data)>MAX_BYTES: raise HTTPException(413,f"File exceeds the {MAX_BYTES//(1024*1024)} MB upload limit.")
    return data


def selected_image_metadata(image: Image.Image) -> dict[str, str]:
    """Return a small technical EXIF allowlist; intentionally omit GPS and comments."""
    tags = {
        "DateTime", "DateTimeOriginal", "DateTimeDigitized", "Make", "Model", "Software",
        "Orientation", "ImageWidth", "ImageLength", "ExifImageWidth", "ExifImageHeight",
    }
    try:
        exif = image.getexif()
    except (OSError, TypeError, ValueError):
        return {}
    metadata: dict[str, str] = {}
    for tag_id, value in exif.items():
        name = ExifTags.TAGS.get(tag_id)
        if name not in tags or isinstance(value, (bytes, bytearray)):
            continue
        value_text = str(value).strip()
        if value_text:
            metadata[name] = value_text[:160]
    return metadata

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile=File(...), db: Session=Depends(get_db)):
    data=await read_upload(file,{".jpg",".jpeg",".png",".webp"})
    try:
        image=Image.open(BytesIO(data)); image.verify()
        with Image.open(BytesIO(data)) as img:
            width,height,fmt=img.width,img.height,img.format
            embedded_metadata=selected_image_metadata(img)
    except (UnidentifiedImageError,OSError,ValueError): raise HTTPException(400,"The uploaded file is not a readable image.")
    findings=[{"type":"metadata","text":f"Readable {fmt} image, {width} × {height} pixels. Metadata alone does not establish authenticity."}]
    if embedded_metadata:
        findings.append({"type":"metadata_evidence","text":"Selected embedded EXIF fields: "+", ".join(f"{key}={value}" for key,value in sorted(embedded_metadata.items()))})
    else:
        findings.append({"type":"metadata_evidence","text":"No selected embedded EXIF fields were available for comparison."})
    configured=gemini_image_configured()
    model_failed=False
    result=None
    if configured:
        try:
            mime_type=Image.MIME.get(fmt or "", "application/octet-stream")
            metadata_context={
                "container_format":str(fmt or "unknown"),
                "width_px":str(width),
                "height_px":str(height),
                **embedded_metadata,
            }
            result=await analyze_image_gemini(data,mime_type,metadata_context)
        except Exception as exc:
            logger.error("Gemini image route failed error_type=%s", type(exc).__name__)
            model_failed=True
    if result:
        for observation in result["visual_observations"]:
            findings.append({"type":"visual_observation","text":observation})
        for observation in result["metadata_observations"]:
            findings.append({"type":"metadata_observation","text":observation})
        for reason in result["inconclusive_reasons"]:
            findings.append({"type":"inconclusive","text":reason})
        likelihood=confidence=None
        verdict="inconclusive"
        summary="Gemini returned visual observations and a review of available metadata. No authenticity verdict or confidence score was assigned."
        limits=["Visual observations are model-generated indicators and do not prove authenticity or manipulation.","Embedded metadata may be missing, altered, or inconsistent with the image content."]
        coverage="gemini_image_review"
    else:
        likelihood=confidence=None; verdict="inconclusive"
        summary="Image received and validated, but no authenticity assessment was performed. " + ("The configured model service did not return a usable result." if configured else "No image detection model is configured.")
        limits=["Check GEMINI_API_KEY, GEMINI_IMAGE_MODEL, SDK availability, and the provider error logs." if model_failed else "Configure GEMINI_API_KEY and GEMINI_IMAGE_MODEL to enable image analysis."]
        coverage="metadata_only"
    row=save(db,"image",Path(file.filename or "image").name,verdict,summary,findings,limitations=limits,likelihood=likelihood,confidence=confidence,coverage=coverage)
    return packed(row)

@app.post("/api/analyze/{kind}")
async def unavailable_media(kind: str,file: UploadFile=File(...),db: Session=Depends(get_db)):
    extensions={"video":{".mp4":"video/mp4", ".mov":"video/quicktime", ".avi":"video/x-msvideo", ".webm":"video/webm"},
                "audio":{".wav":"audio/wav", ".mp3":"audio/mpeg", ".m4a":"audio/mp4", ".flac":"audio/flac"}}
    if kind not in extensions: raise HTTPException(404,"Unknown analysis type.")
    data=await read_upload(file,set(extensions[kind])); label=kind.title()
    findings=[]
    limitations=["Model generated observations are not an authenticity determination; review the original media independently."]
    configured=gemini_media_configured()
    if configured:
        try:
            observations=await analyze_media_gemini(data,kind,extensions[kind][Path(file.filename or "").suffix.lower()])
            findings.append({"type":"model", "text":observations})
            summary=f"Gemini analyzed the {kind} and returned observations. No authenticity verdict was generated."
            coverage="gemini_multimodal_analysis"
        except Exception as exc:
            logger.error("Gemini %s route failed error_type=%s", kind, type(exc).__name__)
            summary=f"{label} was received, but Gemini analysis failed. No model assessment was performed."
            limitations.append("Gemini media analysis failed; check the configured key, model access, and provider status.")
            coverage="analysis_failed"
    else:
        summary=f"{label} upload validated, but Gemini media analysis is not configured; no model assessment was performed."
        limitations.append("Configure GEMINI_API_KEY and ensure the google-genai dependency is installed.")
        coverage="none"
    row=save(db,kind,Path(file.filename or kind).name,"inconclusive",summary,findings,limitations=limitations,coverage=coverage)
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
    for item in json.loads(row.sources_json): line(f"Source: {item.get('title')} - {item.get('publisher')} [{item.get('provider', 'unknown provider')}] - {item.get('url')}")
    for item in json.loads(row.limitations_json): line("Limitation: "+str(item))
    pdf.save(); return Response(buf.getvalue(),media_type="application/pdf",headers={"Content-Disposition":f'attachment; filename="veritas-{row.id[:8]}.pdf"'})



