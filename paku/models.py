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
