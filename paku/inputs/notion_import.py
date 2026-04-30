from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

# Matches Notion page URLs embedded in CSV values: "Genre (https://www.notion.so/...)"
_NOTION_URL_RE = re.compile(r"\s*\(https://www\.notion\.so/[^)]+\)")

_COLUMN_MAP: dict[str, str] = {
    "english title": "english_title",
    "romaji title": "romaji_title",
    "cover": "cover_image",
    "format": "media_format",
    "source": "source",
    "debut year": "debut_year",
    "status": "user_status",
    "country": "country_of_origin",
    "studios": "studios",
    "score": "user_score",
    "episodes": "episodes",
    "notes": "notes",
    "anilist url": "anilist_url",
    "anilist id": "anilist_id",
    "native title": "native_title",
    "genres": "genres",
}

# Notion status values as they might appear in CSV export
_NOTION_STATUS_MAP: dict[str, str] = {
    "watching": "Watching",
    "completed": "Completed",
    "plan to watch": "Plan to Watch",
    "dropped": "Dropped",
    "on hold": "On Hold",
    "not started": "Plan to Watch",
    "awaiting sequel": "Plan to Watch",
    "pending": "Plan to Watch",
    "upcoming": "Plan to Watch",
    "announced": "Plan to Watch",
    "simulcast": "Watching",
    "re-watching": "Watching",
}


def _normalize_column(raw: str) -> str | None:
    """Map a Notion CSV column name to an internal key, or None if unknown."""
    return _COLUMN_MAP.get(raw.strip().lower())


def _normalize_status(raw: str | None) -> str | None:
    """Map a Notion status value to the canonical user_status value."""
    if not raw:
        return None
    return _NOTION_STATUS_MAP.get(raw.strip().lower(), raw.strip())


def _normalize_score(raw: str | None) -> float | None:
    """Parse a Notion score value into a float, or None."""
    if not raw:
        return None
    try:
        return float(raw.strip())
    except (ValueError, TypeError):
        return None


def _clean_notion_urls(value: str) -> str:
    """Strip Notion page URLs embedded in a CSV cell value.

    Notion exports multi-select and relation fields with inline page links:
    "Action (https://www.notion.so/Action-...?pvs=21)"
    → "Action"
    """
    return _NOTION_URL_RE.sub("", value)


def _normalize_list(raw: str | None) -> list[str] | None:
    """Parse a comma-separated list string into a list, or None."""
    if not raw:
        return None
    cleaned = _clean_notion_urls(raw)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts or None


def parse_notion_csv(path: Path) -> list[dict[str, Any]]:
    """Read a Notion-exported CSV and return a list of normalized row dicts.

    Each returned dict contains only the keys that could be mapped from
    the CSV columns. Unmapped columns are silently ignored.
    """
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []

        # Build a mapping from column index to internal key
        col_map: dict[str, str] = {}
        for col in reader.fieldnames:
            key = _normalize_column(col)
            if key:
                col_map[col] = key

        rows: list[dict[str, Any]] = []
        for csv_row in reader:
            row: dict[str, Any] = {}
            for csv_col, internal_key in col_map.items():
                value = csv_row.get(csv_col, "").strip()
                if not value:
                    continue
                row[internal_key] = value

            # Normalize known fields
            if "user_status" in row:
                row["user_status"] = _normalize_status(row.get("user_status"))
            if "user_score" in row:
                row["user_score"] = _normalize_score(row.get("user_score"))
                if row["user_score"] is None:
                    del row["user_score"]
            if "studios" in row and isinstance(row["studios"], str):
                row["studios"] = _normalize_list(row["studios"])
            if "genres" in row and isinstance(row["genres"], str):
                row["genres"] = _normalize_list(row["genres"])
            if "debut_year" in row:
                try:
                    row["debut_year"] = int(row["debut_year"])
                except (ValueError, TypeError):
                    del row["debut_year"]
            if "episodes" in row:
                try:
                    row["episodes"] = int(row["episodes"])
                except (ValueError, TypeError):
                    del row["episodes"]
            if "anilist_id" in row:
                try:
                    row["anilist_id"] = int(row["anilist_id"])
                except (ValueError, TypeError):
                    del row["anilist_id"]

            # Only include rows with at least a title
            if row.get("english_title") or row.get("romaji_title") or row.get("native_title"):
                rows.append(row)

        return rows


def match_rows_to_collection(
    rows: list[dict[str, Any]],
    collection: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Match Notion CSV rows against existing collection entries.

    Matches by anilist_id first, then by canonical_title fuzzy.
    Returns list of (notion_row, matched_entry_or_None) pairs.
    """
    # Build lookup maps from collection
    by_anilist: dict[int, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    by_romaji: dict[str, dict[str, Any]] = {}
    by_raw: dict[str, dict[str, Any]] = {}

    for entry in collection:
        aid = entry.get("anilist_id")
        if aid is not None:
            by_anilist[int(aid)] = entry
        ct = (entry.get("canonical_title") or "").lower().strip()
        if ct:
            by_canonical[ct] = entry
        rt = (entry.get("romaji") or "").lower().strip()
        if rt:
            by_romaji[rt] = entry
        raw = (entry.get("raw_title") or "").lower().strip()
        if raw:
            by_raw[raw] = entry

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for row in rows:
        match = None

        # 1. AniList ID exact match
        aid = row.get("anilist_id")
        if aid is not None and aid in by_anilist:
            match = by_anilist[aid]

        # 2. English title exact match
        if match is None:
            et = (row.get("english_title") or "").lower().strip()
            if et and et in by_canonical:
                match = by_canonical[et]
            elif et and et in by_raw:
                match = by_raw[et]

        # 3. Romaji title exact match
        if match is None:
            rt = (row.get("romaji_title") or "").lower().strip()
            if rt and rt in by_romaji:
                match = by_romaji[rt]
            elif rt and rt in by_canonical:
                match = by_canonical[rt]
            elif rt and rt in by_raw:
                match = by_raw[rt]

        pairs.append((row, match))

    return pairs


def merge_notion_rows(
    db: Any,
    rows: list[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, int]:
    """Merge Notion CSV rows into the dashboard database.

    Returns stats: {matched, updated, created, skipped}.
    The full implementation is in Task 3 — this stub delegates to the
    Database.merge_notion_import method when available.
    """
    if dry_run:
        collection = db.list_anime(per_page=10000).items
        collection_dicts = [
            {
                "anilist_id": e.anilist_id,
                "canonical_title": e.canonical_title,
                "romaji": e.romaji,
                "raw_title": e.raw_title,
            }
            for e in collection
        ]
        pairs = match_rows_to_collection(rows, collection_dicts)
        matched = sum(1 for _, m in pairs if m is not None)
        return {
            "matched": matched,
            "updated": 0,
            "created": len(rows) - matched,
            "skipped": 0,
        }

    # Real merge — delegates to DB method (Task 3)
    if hasattr(db, "merge_notion_import"):
        return db.merge_notion_import(rows, dry_run=False)

    # Stub fallback
    return {"matched": 0, "updated": 0, "created": 0, "skipped": len(rows)}
