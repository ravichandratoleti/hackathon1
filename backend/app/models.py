from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_type: Mapped[str] = mapped_column(String(24), index=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    verdict: Mapped[str] = mapped_column(String(40))
    manipulation_likelihood: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_coverage: Mapped[str] = mapped_column(String(40), default="not_available")
    summary: Mapped[str] = mapped_column(Text, default="")
    findings_json: Mapped[str] = mapped_column(Text, default="[]")
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    limitations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
