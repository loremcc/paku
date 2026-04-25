#!/usr/bin/env python3
"""Back-fill missing studios for anime JSONs that already have an anilist_id.

Targets per-image JSON files where:
  - anilist_id is set
  - studios list is empty
  - media_format is not MANGA (MANGA entries have no animation studio)

Queries AniList by ID (reliable, non-fuzzy) and writes back only the studios
field. Does not alter any other enrichment data.

Usage:
    python scripts/reenrich_studios.py --output output/ [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ANILIST_URL = "https://graphql.anilist.co"

_STUDIOS_QUERY = """
query ($id: Int) {
  Media(id: $id) {
    studios {
      edges {
        isMain
        node { name isAnimationStudio }
      }
    }
  }
}
"""


def _fetch_studios(anilist_id: int, logger: logging.Logger) -> list[str] | None:
    """Return sorted animation-studio names for anilist_id, or None on error."""
    for attempt in range(3):
        try:
            resp = requests.post(
                _ANILIST_URL,
                json={"query": _STUDIOS_QUERY, "variables": {"id": anilist_id}},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.warning(f"[network] id={anilist_id} attempt={attempt+1}: {exc}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.info(f"[rate-limit] sleeping {retry_after}s")
            time.sleep(retry_after)
            continue

        if resp.status_code == 404:
            return []

        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"[parse-error] id={anilist_id}: {exc}")
            return None

        media = (data.get("data") or {}).get("Media") or {}
        edges = (media.get("studios") or {}).get("edges") or []
        animation = sorted({
            e["node"]["name"] for e in edges
            if e.get("node", {}).get("isAnimationStudio")
        })
        if animation:
            return animation
        main = sorted({
            e["node"]["name"] for e in edges
            if e.get("isMain")
        })
        if main:
            return main
        return sorted({
            e["node"]["name"] for e in edges
            if (e.get("node") or {}).get("name")
        })

    return None  # exhausted retries


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True,
                    help="Directory containing per-image JSON files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing files")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("reenrich_studios")

    json_files = sorted(args.output.glob("*.json"))
    targets: list[tuple[Path, dict]] = []

    for p in json_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"[skip] cannot read {p.name}: {exc}")
            continue

        if not isinstance(data, dict):
            continue  # review_queue.json and similar list files

        anilist_id = data.get("anilist_id")
        if not anilist_id:
            continue
        if data.get("studios"):
            continue
        # MANGA entries have no animation studio — skip without querying
        if data.get("media_format") == "MANGA" or data.get("media_source") == "manga":
            continue

        targets.append((p, data))

    logger.info(f"found {len(targets)} targets (anilist_id set, studios empty, non-MANGA)")

    updated = 0
    no_studios_on_anilist = 0
    errors = 0

    for p, data in targets:
        anilist_id = data["anilist_id"]
        studios = _fetch_studios(anilist_id, logger)

        if studios is None:
            logger.warning(f"[error] {p.name}: id={anilist_id} — network failure")
            errors += 1
            continue

        if not studios:
            logger.info(f"[no-studios] {p.name}: id={anilist_id} — AniList has no animation studio")
            no_studios_on_anilist += 1
            continue

        if args.dry_run:
            logger.info(f"[dry] {p.name}: id={anilist_id} -> studios={studios}")
        else:
            data["studios"] = studios
            _atomic_write_json(p, data)
            logger.info(f"[updated] {p.name}: id={anilist_id} -> studios={studios}")

        updated += 1

    print(f"updated={updated}, no_studios_on_anilist={no_studios_on_anilist}, errors={errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
