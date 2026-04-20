from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import requests
from PIL import Image

from .base import OCREngine
from ..models import OcrResult

_MAX_IMAGE_PX = 1024
_HEALTH_TIMEOUT_S = 2
_GENERATE_TIMEOUT_S = 30

_PROMPT = (
    "<thought off> Look at this image. List every anime or manga title you can see, "
    "one per line. If you find nothing, reply with the single word: none"
)


class OllamaVLMEngine(OCREngine):
    """OCR engine backed by a local Ollama VLM (e.g. gemma4-paku:latest).

    Streams NDJSON from /api/generate, stops at first done:true line.
    Health is checked lazily on first is_healthy() call (GET /api/tags with 2s timeout),
    cached for the session lifetime.
    """

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        ollama_cfg = config.get("ollama", {})
        self._base_url = ollama_cfg.get("base_url", "http://192.168.1.114:11434").rstrip("/")
        self._model = ollama_cfg.get("model", "gemma4-paku:latest")
        self._logger = logger
        self._health_cached: bool | None = None

    def name(self) -> str:
        return "ollama_vlm"

    def kind(self) -> str:
        return "smart"

    def is_healthy(self) -> bool:
        """Lazy health check: GET /api/tags with short timeout, cached for session."""
        if self._health_cached is not None:
            return self._health_cached

        try:
            resp = requests.get(
                f"{self._base_url}/api/tags",
                timeout=_HEALTH_TIMEOUT_S,
            )
            self._health_cached = resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout) as e:
            self._logger.debug(f"[ollama_vlm] Health check failed: {e}")
            self._health_cached = False
        except requests.RequestException as e:
            self._logger.debug(f"[ollama_vlm] Health check error: {e}")
            self._health_cached = False

        return self._health_cached

    def extract(self, image: Image.Image | Path) -> OcrResult:
        if isinstance(image, Path):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image

        # Resize longest side to <= 1024 px, encode as JPEG quality=85
        pil_image.thumbnail((_MAX_IMAGE_PX, _MAX_IMAGE_PX), Image.LANCZOS)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": self._model,
            "prompt": _PROMPT,
            "images": [image_b64],
            "stream": True,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=_GENERATE_TIMEOUT_S,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            self._logger.error(f"[ollama_vlm] Request failed: {e}")
            return OcrResult(
                engine=self.name(),
                raw_text="",
                blocks=[],
                language=None,
                meta={"error": str(e)},
            )

        # Stream parse NDJSON — stop at first done:true
        chunks: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done") is True:
                break
            token = obj.get("response", "")
            if token:
                chunks.append(token)

        raw_text = "".join(chunks).strip()

        self._logger.debug(
            f"[ollama_vlm] Streamed {len(chunks)} chunks, "
            f"{len(raw_text)} chars from model={self._model!r}"
        )

        return OcrResult(
            engine=self.name(),
            raw_text=raw_text,
            blocks=[],
            language=None,
            meta={"model": self._model, "streamed_chunks": len(chunks)},
        )
