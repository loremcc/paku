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


class AnimeExtractionResult(ExtractionResult):
    """Anime extractor output."""

    extractor: str = "anime"
    raw_title: str  # exactly what was extracted from OCR
    canonical_title: str | None = None  # AniList english title (or romaji fallback)
    native_title: str | None = None  # AniList native (Japanese) title
    romaji: str | None = None  # AniList romaji title
    media_type: str | None = None  # "ANIME" | "MANGA" | None
    media_source: str = "unknown"  # "anime" | "manga" | "donghua" | "western" | "anilist_app" | "unknown"
    episodes: int | None = None
    status: str | None = None
    genres: list[str] = Field(default_factory=list)
    score: float | None = None
    anilist_id: int | None = None
    anilist_url: str | None = None
    cover_image: str | None = None
    extraction_mode: str = "fast"  # "fast" | "smart" | "anilist_app"
    title_pattern: str | None = None  # "label" | "quoted" | "numbered" | "year_tagged" | "romaji" | "hashtag" | "italian_signal" | "fallback"
    extraction_context: str = "recommendation"  # "recommendation" | "discussion"
    multi_title_detected: bool = False
    dedup_key: str | None = None  # anilist_id as str, else canonical_title.lower() or raw_title.lower()
    levenshtein_ratio: float | None = None  # always logged, even for high-confidence matches
