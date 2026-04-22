from __future__ import annotations

import dataclasses
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .models import OcrResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

_STORY_SIGNALS = ("Send message", "See translation >")
_FEED_CARD_SIGNALS = ("Website", "HuggingFace", "Demo", "GitHub", "Paper")

DOMAIN_PATTERNS = [
    r"github\.com/[\w\-]+/[\w\-\.]+",
    r"gitlab\.com/[\w\-]+/[\w\-\.]+",
    r"arxiv\.org/(?:abs|pdf)/[\d\.]+",
    r"huggingface\.co/[\w\-]+/[\w\-]+",
    r"npmjs\.com/package/[\w\-@/]+",
    r"pypi\.org/project/[\w\-]+",
    r"docs\.google\.com/[\w\-/]+",
]

INGREDIENT_ANCHORS = [
    r"ingredients?:?",
    r"ingredienti:?",
    r"what you(?:\'ll)? need:?",
    r"ingredientes:?",
    r"zutaten:?",
    r"ingr[eé]dients?:?",
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


@dataclasses.dataclass
class BatchReport:
    total: int
    processed: int
    skipped: int
    failed: int
    review_queued: int
    by_content_type: dict[str, int]


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


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    return set(checkpoint_path.read_text(encoding="utf-8").splitlines())


def _append_checkpoint(checkpoint_path: Path, image_path: str) -> None:
    existing = _load_checkpoint(checkpoint_path)
    existing.add(image_path)
    tmp = checkpoint_path.parent / (checkpoint_path.name + ".tmp")
    tmp.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
    os.replace(tmp, checkpoint_path)


def process_batch(
    root: Path,
    mode: str = "auto",
    smart: bool = False,
    outputs: list[str] | None = None,
    resume: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[BatchReport, list[dict[str, Any]]]:
    """Process all images under root, with checkpoint/resume and error isolation.

    Returns BatchReport (counts) and list of successful result dicts.
    "txt" output is stripped from per-image fan-out — callers handle batch consolidation.
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
    output_dir = config.get("outputs", {}).get("base_dir", "./output")
    checkpoint_path = Path(output_dir) / ".paku_checkpoint"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images = discover_images(root)
    total = len(images)

    checkpoint: set[str] = set()
    if resume:
        checkpoint = _load_checkpoint(checkpoint_path)

    # txt consolidation is done after batch — skip per-image txt writes
    batch_outputs = [o for o in outputs if o != "txt"]

    results: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0
    review_queued = 0
    by_content_type: dict[str, int] = {}

    for i, img in enumerate(images):
        if progress_callback is not None:
            progress_callback(i, total, img.name)

        img_str = str(img)
        if resume and img_str in checkpoint:
            skipped += 1
            continue

        try:
            result = process_image(
                image_path=img,
                mode=mode,
                smart=smart,
                outputs=batch_outputs,
            )
        except Exception as e:
            logger.warning(f"[batch] Unhandled error processing {img.name}: {e}")
            append_review_queue(
                _queue_error_entry(img, f"batch_error: {e}"), queue_path
            )
            failed += 1
            review_queued += 1
            continue

        if result is None:
            failed += 1
            review_queued += 1
            continue

        processed += 1
        content_type = result.get("content_type", "unknown")
        by_content_type[content_type] = by_content_type.get(content_type, 0) + 1

        extraction = result.get("extraction") or {}
        if extraction.get("needs_review", False):
            review_queued += 1

        results.append(result)
        _append_checkpoint(checkpoint_path, img_str)

    if progress_callback is not None:
        progress_callback(total, total, "")

    return (
        BatchReport(
            total=total,
            processed=processed,
            skipped=skipped,
            failed=failed,
            review_queued=review_queued,
            by_content_type=by_content_type,
        ),
        results,
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


def _anime_review_reason(res: Any) -> str:
    """Derive review reason from an AnimeExtractionResult."""
    if getattr(res, "levenshtein_ratio", None) is None:
        if not getattr(res, "raw_title", ""):
            return "no_title_extracted"
        if getattr(res, "media_source", None) in ("donghua", "western"):
            return "non_anime_media"
        return "network_error"
    ratio = res.levenshtein_ratio
    if getattr(res, "multi_title_detected", False):
        return "multi_title_screenshot"
    if getattr(res, "extraction_context", None) == "discussion":
        return "discussion_context"
    if ratio < 0.4:
        return "no_anilist_match"
    return f"low_ratio_{ratio:.2f}"


def _anime_review_entry(res: Any, path: Path) -> dict:
    return {
        "screenshot": str(path),
        "extractor": "anime",
        "raw_title": getattr(res, "raw_title", None),
        "canonical_title": getattr(res, "canonical_title", None),
        "anilist_id": getattr(res, "anilist_id", None),
        "levenshtein_ratio": getattr(res, "levenshtein_ratio", None),
        "title_pattern": getattr(res, "title_pattern", None),
        "confidence": res.confidence,
        "reason": _anime_review_reason(res),
        "timestamp": _now_iso(),
    }


def _queue_error_entry(path: Path, reason: str) -> dict:
    return {
        "screenshot": str(path),
        "extractor": None,
        "reason": reason,
        "confidence": _CONFIDENCE_ERROR,
        "timestamp": _now_iso(),
    }


def _review_reason(tier: int) -> str:
    """Return a human-readable review reason for a given extraction tier."""
    reasons = {
        2: "domain-only match — verify target URL",
        3: "reconstructed from author/repo — verify correctness",
        4: "name-only extraction — manual URL lookup needed",
    }
    return reasons.get(tier, f"tier {tier} — needs review")


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

    # --- Extraction dispatch ---
    extraction_result = None

    if content_type == "url":
        from .extractors.url import extract as url_extract

        extraction_result = url_extract(
            ocr_text=text,
            screenshot_path=str(path),
            config=config,
            logger=logger,
        )

    elif content_type == "anime":
        from .extractors.anime import extract as anime_extract

        anime_result = anime_extract(
            ocr_text=text,
            screenshot_path=str(path),
            config=config,
            logger=logger,
        )
        if isinstance(anime_result, list):
            anime_results = anime_result
        else:
            anime_results = [anime_result]

        # Smart re-run: if confidence < 0.4 and --smart is set, re-OCR with Ollama VLM
        if smart and anime_results and anime_results[0].confidence < 0.4:
            try:
                smart_engine = ctx.router.select("smart")
                # Only re-run if we got the Ollama engine (not a fallback)
                if smart_engine.name() == "ollama_vlm":
                    logger.debug(
                        f"[pipeline] Smart re-run triggered for {path.name} "
                        f"(confidence={anime_results[0].confidence:.2f})"
                    )
                    smart_ocr = smart_engine.extract(image)
                    smart_text = smart_ocr.raw_text
                    if smart_text:
                        smart_screen = classify_screen_type(smart_text)
                        smart_result = anime_extract(
                            ocr_text=smart_text,
                            screenshot_path=str(path),
                            config=config,
                            logger=logger,
                        )
                        if isinstance(smart_result, list):
                            anime_results = smart_result
                        else:
                            anime_results = [smart_result]
                        # Stamp extraction_mode on all results from smart path
                        for res in anime_results:
                            res.extraction_mode = "smart"
                        logger.debug(
                            f"[pipeline] Smart re-run complete: "
                            f"confidence={anime_results[0].confidence:.2f}"
                        )
            except RuntimeError as e:
                logger.warning(
                    f"[pipeline] Smart engine unavailable, using fast-path result: {e}"
                )

        for res in anime_results:
            if res.needs_review:
                append_review_queue(_anime_review_entry(res, path), queue_path)
        extraction_result = anime_results[0] if anime_results else None

    elif content_type == "recipe":
        from .extractors.recipe import extract as recipe_extract

        extraction_result = recipe_extract(
            ocr_text=text,
            screenshot_path=str(path),
            config=config,
            logger=logger,
        )
        if extraction_result.needs_review:
            append_review_queue(
                {
                    "screenshot": str(path),
                    "extractor": "recipe",
                    "title": extraction_result.title,
                    "ingredient_count": len(extraction_result.ingredients),
                    "confidence": extraction_result.confidence,
                    "reason": "low_confidence_recipe",
                    "timestamp": _now_iso(),
                },
                queue_path,
            )

    # --- Build result dict ---
    result: dict[str, Any] = {
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

    if extraction_result is not None:
        result["extraction"] = extraction_result.model_dump()
        result["status"] = "extracted"
        if content_type == "anime" and isinstance(anime_results, list) and len(anime_results) > 1:
            result["extractions"] = [r.model_dump() for r in anime_results]

        # Route to review queue for URL extraction (anime handled its own entries above).
        if content_type == "url" and extraction_result.needs_review:
            review_entry = {
                "screenshot": str(path),
                "extractor": extraction_result.extractor,
                "raw_text_snippet": extraction_result.raw_text_snippet,
                "attempted_extraction": {
                    "resolved_url": extraction_result.resolved_url,
                    "raw_keywords": extraction_result.raw_keywords,
                },
                "confidence": extraction_result.confidence,
                "reason": _review_reason(extraction_result.extraction_tier),
                "timestamp": _now_iso(),
            }
            append_review_queue(review_entry, queue_path)

        # Output fan-out.
        output_dir = config.get("outputs", {}).get("base_dir", "./output")
        stem = path.stem

        if "json" in outputs:
            from .outputs.json_out import write_json

            write_json(result["extraction"], stem, output_dir)

        if "txt" in outputs:
            from .outputs.txt_out import write_txt

            if content_type == "anime":
                txt_value = getattr(extraction_result, "canonical_title", None) or getattr(extraction_result, "raw_title", None)
            elif content_type == "recipe":
                txt_value = getattr(extraction_result, "title", None)
            else:
                txt_value = getattr(extraction_result, "resolved_url", None)
            write_txt(txt_value, stem, output_dir)

        if "csv" in outputs and content_type == "recipe":
            from .outputs.csv_out import write_csv

            write_csv(result["extraction"].get("ingredients", []), stem, output_dir)

    return result
