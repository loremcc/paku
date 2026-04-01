from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OcrBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox | None = None
    type: Literal["line", "word", "paragraph"] = "line"


class OcrResult(BaseModel):
    engine: str
    raw_text: str
    blocks: list[OcrBlock] = Field(default_factory=list)
    language: str | None = None
    meta: dict = Field(default_factory=dict)


# --- Extraction models (v0.2+) ---


class ExtractionResult(BaseModel):
    """Shared base for all extractor outputs."""

    extractor: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False
    source_screenshot: str
    extracted_at: str  # ISO 8601 UTC


class URLExtractionResult(ExtractionResult):
    """URL extractor output."""

    extractor: str = "url"
    resolved_url: str | None = None
    raw_text_snippet: str = ""
    raw_keywords: str | None = None
    extraction_mode: str = "direct"
    extraction_tier: int  # 1-4
