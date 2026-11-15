from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AnalysisOut(BaseModel):
    id: str
    analysis_type: str
    original_filename: str | None
    verdict: str
    manipulation_likelihood: float | None
    model_confidence: float | None
    evidence_coverage: str
    summary: str
    findings: list[Any]
    sources: list[Any]
    limitations: list[Any]
    created_at: datetime

    class Config:
        from_attributes = True


DISCLAIMER = (
    "AI-generated verification results are probabilistic and may contain errors. "
    "This assessment is not definitive proof of authenticity or manipulation. "
    "Verify important findings using independent sources."
)
