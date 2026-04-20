from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from ..models import OcrResult


class OCREngine(ABC):
    """Base interface for all OCR engines used by paku."""

    @abstractmethod
    def name(self) -> str:
        """Unique engine name (e.g. 'stub', 'google_vision', 'ollama_vlm')."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, image: Image.Image | Path) -> OcrResult:
        """Run OCR on a preprocessed PIL image (or a path) and return a unified OcrResult."""
        raise NotImplementedError

    def kind(self) -> str:
        """
        Engine kind for routing: 'light' or 'heavy'.

        Default is 'light'. Heavy engines are more accurate but slower or
        require an external service.
        """
        return "light"

    def is_healthy(self) -> bool:
        """Best-effort health indicator for routing."""
        return True
