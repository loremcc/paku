#!/usr/bin/env python3
"""Re-enrich review_queue entries flagged as network_error.

Retries AniList for each entry's raw_title. When an AniList match is found
and the enhanced ratio >= 0.6, merges the enriched AnimeExtractionResult
into the corresponding per-image JSON and drops the entry from
review_queue.json. Entries that still fail are preserved unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paku.extractors.anime import (  # noqa: E402
    _COUNTRY_MAP,
    _assign_confidence,
    _query_anilist_best,
)
from paku.models import AnimeExtractionResult  # noqa: E402


def _build_enriched(
    existing: dict[str, Any],
    media: dict,
    ratio: float,
) -> dict[str, Any]:
    titles = media.get("title", {}) or {}
    anilist_id = media.get("id")
    canonical_title = titles.get("english") or titles.get("romaji")
    native_title = titles.get("native")
    romaji = titles.get("romaji")
    media_type = media.get("type")
    media_source = (
        "anime" if media_type == "ANIME"
        else "manga" if media_type == "MANGA"
        else "unknown"
    )
    raw_score = media.get("averageScore")
    score = raw_score / 10.0 if raw_score is not None else None
    cover_image = (
        (media.get("coverImage") or {}).get("extraLarge")
        or (media.get("coverImage") or {}).get("large")
    )
    studio_edges = (media.get("studios") or {}).get("edges") or []
    _animation = sorted({
        e["node"]["name"] for e in studio_edges
        if e.get("node", {}).get("isAnimationStudio")
    })
    if _animation:
        studios = _animation
    else:
        _main = sorted({
            e["node"]["name"] for e in studio_edges
            if e.get("isMain")
        })
        studios = _main if _main else sorted({
            e["node"]["name"] for e in studio_edges
            if (e.get("node") or {}).get("name")
        })

    title_pattern = existing.get("title_pattern")
    extraction_context = existing.get("extraction_context") or "recommendation"
    confidence, needs_review = _assign_confidence(
        ratio, title_pattern, extraction_context
    )
    if existing.get("multi_title_detected") and ratio < 0.8:
        needs_review = True

    dedup_key = (
        str(anilist_id) if anilist_id
        else (canonical_title or existing.get("raw_title") or "").lower().strip()
    )

    result = AnimeExtractionResult(
        extractor="anime",
        confidence=confidence,
        needs_review=needs_review,
        source_screenshot=existing["source_screenshot"],
        extracted_at=existing["extracted_at"],
        raw_title=existing["raw_title"],
        canonical_title=canonical_title,
        native_title=native_title,
        romaji=romaji,
        media_type=media_type,
        media_source=media_source,
        episodes=media.get("episodes"),
        status=media.get("status"),
        genres=media.get("genres") or [],
        score=score,
        anilist_id=anilist_id,
        anilist_url=media.get("siteUrl"),
        cover_image=cover_image,
        banner_image=media.get("bannerImage"),
        media_format=media.get("format"),
        source=media.get("source"),
        country_of_origin=_COUNTRY_MAP.get(media.get("countryOfOrigin") or "", ""),
        debut_year=(media.get("startDate") or {}).get("year"),
        studios=studios,
        extraction_mode=existing.get("extraction_mode") or "fast",
        title_pattern=title_pattern,
        extraction_context=extraction_context,
        multi_title_detected=bool(existing.get("multi_title_detected", False)),
        dedup_key=dedup_key,
        levenshtein_ratio=ratio,
    )
    return result.model_dump()


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True,
                    help="Directory containing per-image JSON files")
    ap.add_argument("--queue", type=Path, required=True,
                    help="Path to review_queue.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report resolutions without writing files")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("reenrich")

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    targets = [
        (i, e) for i, e in enumerate(queue)
        if e.get("reason") == "network_error"
        and (e.get("raw_title") or "").strip()
    ]
    logger.info(f"scanning {len(targets)} network_error entries")

    resolved_indices: set[int] = set()
    errors = 0

    for idx, entry in targets:
        raw_title = entry["raw_title"]
        stem = Path(entry.get("screenshot") or "").stem
        json_path = args.output / f"{stem}.json"

        if not stem or not json_path.exists():
            logger.warning(
                f"[skip] per-image JSON missing: {json_path.name} "
                f"(raw_title={raw_title!r})"
            )
            errors += 1
            continue

        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[error] cannot read {json_path.name}: {e}")
            errors += 1
            continue

        if existing.get("anilist_id"):
            logger.info(
                f"[already-enriched] {stem}: anilist_id={existing['anilist_id']}"
            )
            resolved_indices.add(idx)
            continue

        try:
            media, ratio, err = _query_anilist_best(raw_title, raw_title, logger)
        except Exception as e:  # pragma: no cover — AniList client raised
            logger.warning(f"[error] AniList raised for {raw_title!r}: {e}")
            errors += 1
            continue

        if err == "network_error":
            logger.info(f"[still-network] {raw_title!r}")
            continue

        if not media or ratio < 0.6:
            logger.info(f"[no-match] {raw_title!r} ratio={ratio:.2f}")
            continue

        updated = _build_enriched(existing, media, ratio)
        canonical = updated.get("canonical_title")

        if args.dry_run:
            logger.info(
                f"[dry] {stem}: {raw_title!r} -> {canonical!r} "
                f"ratio={ratio:.2f}"
            )
        else:
            _atomic_write_json(json_path, updated)
            logger.info(
                f"[resolved] {stem}: {raw_title!r} -> {canonical!r} "
                f"ratio={ratio:.2f}"
            )

        resolved_indices.add(idx)

    still_queued = len(targets) - len(resolved_indices)

    if not args.dry_run and resolved_indices:
        remaining = [e for i, e in enumerate(queue) if i not in resolved_indices]
        _atomic_write_json(args.queue, remaining)

    print(
        f"resolved={len(resolved_indices)}, "
        f"still_queued={still_queued}, errors={errors}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
