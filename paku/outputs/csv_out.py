from __future__ import annotations

# Writes recipe ingredient rows and anime Notion import CSV.
# Atomic write (tmp → os.replace) matches review_queue.json pattern.
# Column headers must exactly match Notion property names (case-sensitive).
import csv
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

if TYPE_CHECKING:
    from ..models import AnimeExtractionResult

AnimeRecord = Union["AnimeExtractionResult", dict[str, Any]]


def _field(record: AnimeRecord, key: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic model or a plain dict."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)

# Exact Notion "Full Catalog" DB property names — single source of truth.
# Only includes columns paku can populate from AnimeExtractionResult + AniList data.
ANIME_CSV_HEADERS: list[str] = [
    "English Title",
    "Romaji Title",
    "Cover",
    "Format",
    "Source",
    "Debut Year",
    "Status",
    "Country",
    "Studios",
]

_FORMAT_MAP: dict[str, str] = {
    "TV": "TV",
    "MOVIE": "Movie",
    "OVA": "OVA",
    "ONA": "ONA",
    "SPECIAL": "Special",
    "TV_SHORT": "TV Short",
}

_SOURCE_MAP: dict[str, str] = {
    "MANGA": "Manga",
    "LIGHT_NOVEL": "Light Novel",
    "ORIGINAL": "Original",
    "VIDEO_GAME": "Video Game",
    "NOVEL": "Novel",
    "DOUJINSHI": "Doujinshi",
    "ANIME": "Anime",
    "VISUAL_NOVEL": "Visual Novel",
    "COMIC": "Comic",
    "OTHER": "Other",
}


def write_csv(ingredients: list[dict], screenshot_stem: str, output_dir: str) -> Path:
    """Write ingredient rows to <output_dir>/<screenshot_stem>.csv.

    Columns: ingredient, quantity, unit, notes
    Uses atomic write: writes to .tmp then os.replace to avoid partial files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{screenshot_stem}.csv"
    tmp_path = out_dir / f"{screenshot_stem}.csv.tmp"

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["ingredient", "quantity", "unit", "notes"],
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in ingredients:
        qty = row.get("quantity")
        writer.writerow(
            {
                "ingredient": row.get("name", ""),
                "quantity": str(qty) if qty is not None else "",
                "unit": row.get("unit") or "",
                "notes": row.get("notes") or "",
            }
        )

    tmp_path.write_text(buf.getvalue(), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path


def write_anime_csv(
    results: Iterable[AnimeRecord],
    output_path: Path,
) -> Path:
    """Write anime results to a Notion-importable CSV at output_path.

    Accepts either AnimeExtractionResult objects or plain dicts (per-image JSON
    rehydrated from disk). Deduplicates by dedup_key (higher confidence wins).
    Headers are from ANIME_CSV_HEADERS — exact Notion DB property names.
    Uses atomic write: writes to .tmp then os.replace.
    """
    # Dedup: keep highest-confidence record per dedup_key
    seen: dict[str, AnimeRecord] = {}
    for res in results:
        key = _field(res, "dedup_key") or _field(res, "raw_title") or ""
        existing = seen.get(key)
        confidence = _field(res, "confidence", 0.0) or 0.0
        existing_conf = (_field(existing, "confidence", 0.0) or 0.0) if existing else -1.0
        if existing is None or confidence > existing_conf:
            seen[key] = res

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / (output_path.name + ".tmp")

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=ANIME_CSV_HEADERS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()

    for res in seen.values():
        canonical = _field(res, "canonical_title")
        raw = _field(res, "raw_title") or ""
        # Skip entries with no resolvable title.
        if not canonical and not raw:
            continue
        # Skip unresolved (no AniList match) entries flagged for review — these
        # are multi-title blobs or poor-OCR rows that add no signal to Notion.
        if not canonical and _field(res, "needs_review", False):
            continue
        media_format = _field(res, "media_format")
        source = _field(res, "source")
        debut_year = _field(res, "debut_year")
        country = _field(res, "country_of_origin")
        studios = _field(res, "studios") or []

        writer.writerow(
            {
                "English Title": canonical or raw,
                "Romaji Title": _field(res, "romaji") or "",
                "Cover": _field(res, "cover_image") or "",
                "Format": _FORMAT_MAP.get(media_format or "", media_format or ""),
                "Source": _SOURCE_MAP.get(source or "", source or ""),
                "Debut Year": str(debut_year) if debut_year is not None else "",
                "Status": "Not Started",
                "Country": country or "",
                "Studios": ", ".join(studios) if studios else "",
            }
        )

    tmp_path.write_text(buf.getvalue(), encoding="utf-8")
    os.replace(tmp_path, output_path)
    return output_path
