#!/usr/bin/env python3
"""Revert 'none'->GONE and 'real'->Ream garbage matches written by re-enrichment scripts.

Finds all output/*.json where canonical_title in {GONE, Ream} or raw_title matches
/^\s*none\s*$/i and reverts the AniList enrichment fields. Adds entries back to
review_queue.json with reason="garbage_title" so they get manual attention.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_GARBAGE_CANONICAL = {"GONE", "Ream"}
_GARBAGE_RAW_RE = re.compile(r"^\s*none\s*$", re.IGNORECASE)


def _atomic_write(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _revert_json(data: dict) -> dict:
    """Strip AniList enrichment from a garbage JSON, keep extraction provenance."""
    raw = data.get("raw_title") or ""
    data["canonical_title"] = None
    data["native_title"] = None
    data["romaji"] = None
    data["media_type"] = None
    data["media_source"] = "unknown"
    data["episodes"] = None
    data["status"] = None
    data["genres"] = []
    data["score"] = None
    data["anilist_id"] = None
    data["anilist_url"] = None
    data["cover_image"] = None
    data["banner_image"] = None
    data["media_format"] = None
    data["source"] = None
    data["country_of_origin"] = None
    data["debut_year"] = None
    data["studios"] = []
    data["dedup_key"] = raw.lower().strip()
    data["levenshtein_ratio"] = None
    data["confidence"] = 0.0
    data["needs_review"] = True
    return data


def _make_queue_entry(data: dict, json_path: Path) -> dict:
    return {
        "reason": "garbage_title",
        "raw_title": data.get("raw_title"),
        "source_screenshot": data.get("source_screenshot", ""),
        "screenshot": str(json_path.stem) + ".json",
        "extraction_context": data.get("extraction_context"),
        "title_pattern": data.get("title_pattern"),
        "note": (
            "raw_title is a garbage value (e.g. 'none', 'real') that "
            "fuzzy-matched a wrong AniList entry — needs manual review"
        ),
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    out_dir = ROOT / "output"
    queue_path = out_dir / "review_queue.json"

    # Load existing queue; build set of known sources to avoid duplicates
    queue: list[dict] = json.loads(queue_path.read_text(encoding="utf-8"))
    existing_sources = {
        e.get("source_screenshot", "") for e in queue
        if e.get("reason") == "garbage_title"
    }

    fixed = 0
    added_to_queue = 0

    for p in sorted(out_dir.glob("*.json")):
        if p.name == "review_queue.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] cannot read {p.name}: {e}")
            continue

        canonical = data.get("canonical_title") or ""
        raw = data.get("raw_title") or ""

        is_garbage = canonical in _GARBAGE_CANONICAL or _GARBAGE_RAW_RE.match(raw)
        if not is_garbage:
            continue

        print(f"[fix] {p.name}: raw={raw!r} -> canonical={canonical!r}")
        reverted = _revert_json(dict(data))
        _atomic_write(p, reverted)
        fixed += 1

        src = data.get("source_screenshot", "")
        if src not in existing_sources:
            queue.append(_make_queue_entry(data, p))
            existing_sources.add(src)
            added_to_queue += 1

    if fixed:
        _atomic_write(queue_path, queue)

    print(f"\nfixed={fixed}, added_to_queue={added_to_queue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
