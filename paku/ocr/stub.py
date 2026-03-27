from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .base import OCREngine
from ..models import OcrResult


class StubOCREngine(OCREngine):
    """Fake OCR engine for development and pipeline validation.

    Returns predictable stub text without reading the image. Replace with a
    real engine (Chandra) once the pipeline plumbing is verified.
    """

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def name(self) -> str:
        return "stub"

    def extract(self, image: Image.Image | Path) -> OcrResult:
        path_info = image if isinstance(image, Path) else None
        name = path_info.name if path_info else "<image>"
        self._logger.debug(f"[stub] OCR on {name}")
        return OcrResult(
            engine=self.name(),
            raw_text=(
                f"[stub] This is fake OCR output for {name}.\n"
                "Send message... story chrome detected.\n"
                "Sample caption text with enough characters to pass quality guard."
            ),
            blocks=[],
            language=None,
            meta={"note": "stub engine — replace with Chandra OCR", "path": str(path_info or "")},
        )
