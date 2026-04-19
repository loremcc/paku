from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from PIL import Image

from .base import OCREngine
from ..models import BoundingBox, OcrBlock, OcrResult


class GoogleVisionOCREngine(OCREngine):
    """OCR engine backed by Google Cloud Vision API (document_text_detection).

    Auth priority (checked in is_healthy and applied in _build_client):
    1. GOOGLE_APPLICATION_CREDENTIALS env var → ImageAnnotatorClient()
    2. google_vision.api_key in config.yaml  → ImageAnnotatorClient(client_options={"api_key": ...})

    is_healthy() never instantiates ImageAnnotatorClient — only checks SDK importability
    and credential presence, avoiding any network call during AppContext initialization.
    """

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def name(self) -> str:
        return "google_vision"

    def kind(self) -> str:
        return "heavy"

    def is_healthy(self) -> bool:
        """Check SDK importability and credential presence. No network call, no client init."""
        try:
            import google.cloud.vision  # noqa: F401
        except ImportError:
            self._logger.debug("[google_vision] SDK not installed — engine unavailable")
            return False

        has_env = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip())
        has_key = bool(
            self._config.get("google_vision", {}).get("api_key", "").strip()
        )
        has_file = bool(
            self._config.get("google_vision", {}).get("credentials_file", "").strip()
        )
        if not (has_env or has_key or has_file):
            self._logger.debug(
                "[google_vision] No credentials found — set GOOGLE_APPLICATION_CREDENTIALS, "
                "google_vision.credentials_file, or google_vision.api_key in config.yaml"
            )
            return False

        return True

    def extract(self, image: Image.Image | Path) -> OcrResult:
        from google.cloud import vision

        if isinstance(image, Path):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image

        # Convert PIL Image to PNG bytes — never touch the filesystem
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        client = self._build_client(vision)
        vision_image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=vision_image)

        if response.error.message:
            raise RuntimeError(
                f"[google_vision] Vision API error: {response.error.message}"
            )

        raw_text = (
            response.text_annotations[0].description
            if response.text_annotations
            else ""
        )
        blocks = self._map_blocks(response)
        language = self._detect_language(response)

        self._logger.debug(
            f"[google_vision] extracted {len(raw_text)} chars, "
            f"{len(blocks)} blocks, lang={language!r}"
        )

        return OcrResult(
            engine=self.name(),
            raw_text=raw_text,
            blocks=blocks,
            language=language,
            meta={
                "block_count": len(blocks),
                "detected_language": language,
                "annotation_count": len(response.text_annotations),
            },
        )

    def _build_client(self, vision_module):
        # Env var path: SDK reads GOOGLE_APPLICATION_CREDENTIALS automatically
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
            return vision_module.ImageAnnotatorClient()
        # Service account JSON file path from config
        credentials_file = (
            self._config.get("google_vision", {}).get("credentials_file", "").strip()
        )
        if credentials_file:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                credentials_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return vision_module.ImageAnnotatorClient(credentials=creds)
        # API key path: explicit client_options
        api_key = self._config.get("google_vision", {}).get("api_key", "").strip()
        return vision_module.ImageAnnotatorClient(client_options={"api_key": api_key})

    def _map_blocks(self, response) -> list[OcrBlock]:
        blocks: list[OcrBlock] = []
        full = response.full_text_annotation
        if not full or not full.pages:
            return blocks

        for page in full.pages:
            for block in page.blocks:
                text = self._block_text(block)
                if not text.strip():
                    continue
                confidence = float(getattr(block, "confidence", 1.0))
                confidence = min(max(confidence, 0.0), 1.0)
                bbox = self._map_bbox(block.bounding_box)
                blocks.append(
                    OcrBlock(text=text, confidence=confidence, bbox=bbox, type="paragraph")
                )

        return blocks

    @staticmethod
    def _block_text(block) -> str:
        words = []
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                symbols = [s.text for s in word.symbols]
                words.append("".join(symbols))
        return " ".join(words)

    @staticmethod
    def _map_bbox(bounding_box) -> BoundingBox | None:
        verts = list(bounding_box.vertices) if bounding_box else []
        if len(verts) < 4:
            return None
        x = int(getattr(verts[0], "x", 0))
        y = int(getattr(verts[0], "y", 0))
        width = abs(int(getattr(verts[1], "x", x)) - x)
        height = abs(int(getattr(verts[3], "y", y)) - y)
        return BoundingBox(x=x, y=y, width=width, height=height)

    @staticmethod
    def _detect_language(response) -> str | None:
        full = response.full_text_annotation
        if not full or not full.pages:
            return None
        page = full.pages[0]
        langs = list(page.property.detected_languages) if page.property else []
        if langs:
            return langs[0].language_code or None
        return None
