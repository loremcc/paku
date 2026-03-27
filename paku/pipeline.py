from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .models import OcrResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

_STORY_SIGNALS = ("Send message", "See translation >")
_FEED_CARD_SIGNALS = ("Website", "HuggingFace", "Demo", "GitHub", "Paper")

DOMAIN_PATTERNS = [
    r"github\.com/[\w\-]+/[\w\-\.]+",
    r"arxiv\.org/(?:abs|pdf)/[\d\.]+",
    r"huggingface\.co/[\w\-]+/[\w\-]+",
    r"npmjs\.com/package/[\w\-@/]+",
    r"pypi\.org/project/[\w\-]+",
]

INGREDIENT_ANCHORS = [
    r"ingredients?:?",
    r"ingredienti:?",
    r"what you(?:\'ll)? need:?",
]

ITALIAN_SIGNALS = [
    "rilasciato",
    "episodio di",
    "stagione",
    "guardato",
    "anime",
    "manga",
    "uscito",
    "primo episodio",
]

_MIN_MEANINGFUL_CHARS = 15
_MAX_IMAGE_WIDTH = 2160
_STATUS_PENDING = "pending_extraction"
_CONFIDENCE_ERROR = "error"


class PoorOCRQuality(Exception):
    """Raised when OCR output is below the minimum quality threshold."""


def load_image(path: Path) -> Image.Image:
    """Load image from disk. Raises OSError if file cannot be opened."""
    return Image.open(path)


def preprocess(image: Image.Image) -> Image.Image:
    """Normalize image for OCR: convert to RGB, resize if wider than 2160px."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width > _MAX_IMAGE_WIDTH:
        ratio = _MAX_IMAGE_WIDTH / image.width
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image


def guard_ocr_quality(text: str) -> None:
    """Raise PoorOCRQuality if text has fewer than 15 meaningful characters."""
    meaningful = [c for c in text if c.isalpha() or c.isdigit()]
    if len(meaningful) < _MIN_MEANINGFUL_CHARS:
        raise PoorOCRQuality(
            f"OCR output too short: {len(meaningful)} meaningful chars "
            f"(minimum {_MIN_MEANINGFUL_CHARS})"
        )


def classify_screen_type(ocr_text: str) -> str:
    """Classify Instagram screen type from OCR text.

    Returns: 'post' | 'story' | 'feed_card'
    """
    if any(signal in ocr_text for signal in _STORY_SIGNALS):
        return "story"

    if re.search(r"\n[A-Z][^\n]{10,}\n", ocr_text) and any(
        x in ocr_text for x in _FEED_CARD_SIGNALS
    ):
        return "feed_card"

    return "post"


def classify_content(ocr_text: str, screen_type: str, mode: str = "auto") -> str:
    """Classify content type from OCR text.

    Returns: 'anime' | 'url' | 'recipe' | 'unknown'
    If mode is not 'auto', returns mode directly (explicit override).
    """
    if mode != "auto":
        return mode

    text_lower = ocr_text.lower()

    if any(re.search(p, ocr_text) for p in DOMAIN_PATTERNS):
        return "url"

    if any(re.search(a, text_lower) for a in INGREDIENT_ANCHORS):
        return "recipe"

    if any(signal in text_lower for signal in ITALIAN_SIGNALS):
        return "anime"

    if screen_type == "story":
        return "anime"

    return "unknown"


def discover_images(root: Path) -> list[Path]:
    """Recursively find all image files under root, sorted by name."""
    if root.is_file():
        if root.suffix.lower() in IMAGE_EXTENSIONS:
            return [root]
        return []
    return sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def append_review_queue(entry: dict, queue_path: str | Path) -> None:
    """Atomically append an entry to review_queue.json.

    Read → append → write to .tmp → os.replace (atomic on POSIX and Windows).
    """
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    try:
        with open(queue_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        pass
    existing.append(entry)
    tmp_path = queue_path.parent / (queue_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, queue_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_error_entry(path: Path, reason: str) -> dict:
    return {
        "screenshot": str(path),
        "extractor": None,
        "reason": reason,
        "confidence": _CONFIDENCE_ERROR,
        "timestamp": _now_iso(),
    }


def process_image(
    image_path: str | Path,
    mode: str = "auto",
    smart: bool = False,
    outputs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Run the full pipeline for a single image (steps 1–6 for v0.1).

    Steps 7–10 (extract, normalize, fan_out, route_confidence) are stubs
    pending extractor implementation in v0.2.

    Returns result dict, or None if image is unprocessable.
    Low-confidence results are written to review_queue AND returned.
    """
    from .context import AppContext

    ctx = AppContext.instance()
    config = ctx.config
    logger = ctx.logger

    if outputs is None:
        outputs = []

    queue_path: str = config.get("outputs", {}).get(
        "review_queue", "./output/review_queue.json"
    )

    path = Path(image_path)

    try:
        image = load_image(path)
    except Exception as e:
        logger.warning(f"[pipeline] Cannot load {path.name}: {e}")
        append_review_queue(_queue_error_entry(path, f"load_error: {e}"), queue_path)
        return None

    image = preprocess(image)

    engine = ctx.resolve_engine("auto")
    ocr_result: OcrResult = engine.extract(image)
    text = ocr_result.raw_text

    try:
        guard_ocr_quality(text)
    except PoorOCRQuality as e:
        logger.warning(f"[pipeline] Poor OCR quality for {path.name}: {e}")
        append_review_queue(_queue_error_entry(path, f"poor_ocr: {e}"), queue_path)
        return None

    screen_type = classify_screen_type(text)
    logger.debug(f"[pipeline] {path.name}: screen_type={screen_type}")

    content_type = classify_content(text, screen_type, mode)
    logger.debug(f"[pipeline] {path.name}: content_type={content_type}")

    return {
        "screenshot": str(path),
        "screen_type": screen_type,
        "content_type": content_type,
        "ocr_text": text,
        "engine": ocr_result.engine,
        "outputs": list(outputs),
        "smart": smart,
        "extracted_at": _now_iso(),
        "status": _STATUS_PENDING,
    }
