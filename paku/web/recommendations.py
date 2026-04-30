from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import requests

_RECS_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS recommendation_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_hash TEXT UNIQUE,
    suggestions TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

_REFRESH_INTERVAL = 3600  # regenerate recs at most once per hour


def build_collection_context(db: Any) -> dict[str, Any]:
    """Aggregate the user's collection into a structured context dict for the LLM."""
    all_entries = db.list_anime(per_page=100000).items
    if not all_entries:
        return {"empty": True}

    # Top rated by user_score (falling back to community score)
    scored = sorted(
        [e for e in all_entries if (e.user_score or e.score)],
        key=lambda e: e.user_score or e.score or 0,
        reverse=True,
    )
    top_rated = scored[:20]

    # Genre distribution
    genre_counts: dict[str, int] = {}
    for e in all_entries:
        for g in e.genres or []:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]

    # Studio distribution
    studio_counts: dict[str, int] = {}
    for e in all_entries:
        for s in e.studios or []:
            studio_counts[s] = studio_counts.get(s, 0) + 1
    top_studios = sorted(studio_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # Format distribution
    format_counts: dict[str, int] = {}
    for e in all_entries:
        fmt = e.media_format or "Unknown"
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    top_formats = sorted(format_counts.items(), key=lambda kv: kv[1], reverse=True)

    # Status distribution
    status_counts: dict[str, int] = {}
    for e in all_entries:
        s = e.user_status or "Unknown"
        status_counts[s] = status_counts.get(s, 0) + 1

    # Already-collected AniList IDs (to exclude from suggestions)
    seen_ids: set[int] = {e.anilist_id for e in all_entries if e.anilist_id is not None}
    seen_titles: set[str] = {
        (e.canonical_title or e.raw_title or "").lower().strip() for e in all_entries
    }

    return {
        "empty": False,
        "total_entries": len(all_entries),
        "top_rated": [
            {
                "title": e.canonical_title or e.raw_title,
                "score": e.user_score or e.score,
                "genres": e.genres or [],
                "studios": e.studios or [],
                "format": e.media_format,
                "year": e.debut_year,
            }
            for e in top_rated
        ],
        "top_genres": [{"genre": g, "count": c} for g, c in top_genres],
        "top_studios": [{"studio": s, "count": c} for s, c in top_studios],
        "top_formats": [{"format": f, "count": c} for f, c in top_formats],
        "statuses": status_counts,
        "seen_anilist_ids": list(seen_ids),
        "seen_titles": list(seen_titles),
    }


def build_recommendation_prompt(context: dict[str, Any]) -> str:
    """Format collection context into a Gemma 4 prompt for personalized recs."""
    if context.get("empty"):
        return ""

    top_rated_text = "\n".join(
        f"- {e['title']} ({e.get('format', '?')}, {e.get('year', '?')}) "
        f"[genres: {', '.join(e.get('genres', []))}]"
        for e in context["top_rated"][:10]
    )

    genres_text = ", ".join(f"{g['genre']}({g['count']})" for g in context["top_genres"][:10])

    studios_text = ", ".join(f"{s['studio']}({s['count']})" for s in context["top_studios"][:6])

    formats_text = ", ".join(f"{f['format']}({f['count']})" for f in context["top_formats"])

    status_text = ", ".join(f"{k}: {v}" for k, v in context["statuses"].items())

    rules = (
        "Do NOT suggest titles already in the user's collection or "
        "from the Top rated list. Suggest lesser-known titles, not "
        "just the most popular ones."
    )
    prompt = (
        "You are an anime recommendation engine. Based on this "
        f"user's collection profile ({context['total_entries']} titles), "
        "suggest 10 anime they might enjoy but haven't watched yet.\n\n"
        f"Top rated:\n{top_rated_text}\n\n"
        f"Genre preferences: {genres_text}\n"
        f"Favorite studios: {studios_text}\n"
        f"Format preferences: {formats_text}\n"
        f"Watch status: {status_text}\n\n"
        f"RULES: {rules}\n\n"
        "Return ONLY a numbered list of 10 titles, one per line, like this:\n"
        "1. Title Name\n2. Another Title\n...\n10. Last Title\n\n"
        "Include no other text in your response."
    )

    return prompt


def parse_llm_suggestions(response: str) -> list[str]:
    """Extract anime title suggestions from an LLM text response."""
    titles: list[str] = []
    # Match numbered list items: "1. Title" or "1) Title" or "- Title"
    pattern = re.compile(r"(?:^\d+[.)]\s*|^[-•]\s*)(.+)$", re.MULTILINE)
    for match in pattern.finditer(response):
        title = match.group(1).strip()
        # Clean up common LLM artifacts
        title = re.sub(r"\s*\([^)]*\)\s*$", "", title)  # trailing parens
        title = re.sub(r'\s*"[^"]*"\s*$', "", title)  # trailing quotes
        title = title.strip("\"'*_")
        if title and len(title) > 1:
            titles.append(title)
    return titles[:10]


def _anilist_search(query: str) -> dict[str, Any] | None:
    """Lightweight AniList search for resolving a title suggestion."""
    try:
        resp = requests.post(
            "https://graphql.anilist.co",
            json={
                "query": """
                query ($search: String) {
                  Page(page: 1, perPage: 3) {
                    media(search: $search, sort: SEARCH_MATCH) {
                      id
                      title { english romaji }
                      coverImage { extraLarge large }
                      averageScore
                      format
                      status
                      genres
                    }
                  }
                }
                """,
                "variables": {"search": query},
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def resolve_suggestions(titles: list[str], db: Any) -> list[dict[str, Any]]:
    """Search AniList for each suggested title, filter already-collected."""
    resolved: list[dict[str, Any]] = []
    seen_anilist_ids: set[int] = set()

    # Collect already-saved IDs
    all_entries = db.list_anime(per_page=100000).items
    saved_ids: set[int] = {e.anilist_id for e in all_entries if e.anilist_id is not None}

    for title in titles:
        data = _anilist_search(title)
        if data is None:
            continue
        media_list = ((data.get("data") or {}).get("Page") or {}).get("media") or []
        for media in media_list:
            aid = media.get("id")
            if aid is None or aid in seen_anilist_ids or aid in saved_ids:
                continue
            seen_anilist_ids.add(aid)
            title_obj = media.get("title") or {}
            cover = media.get("coverImage") or {}
            score = media.get("averageScore")
            resolved.append(
                {
                    "anilist_id": aid,
                    "english": title_obj.get("english"),
                    "romaji": title_obj.get("romaji"),
                    "cover_image": (cover.get("extraLarge") or cover.get("large")),
                    "average_score": score,
                    "media_format": media.get("format"),
                    "status": media.get("status"),
                    "genres": media.get("genres") or [],
                    "saved": False,
                    "matched_query": title,
                }
            )
            if len(resolved) >= 10:
                break
        if len(resolved) >= 10:
            break

    return resolved


def get_recommendations(
    db: Any, ollama_url: str, model: str, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Orchestrator: build context, call Ollama, resolve, cache, return."""
    # Ensure cache table exists
    db._connect().executescript(_RECS_CACHE_DDL)

    context = build_collection_context(db)
    if context.get("empty"):
        return {
            "recommendations": [],
            "reason": "empty_collection",
            "source": "ollama",
        }

    context_hash = hashlib.sha256(
        json.dumps(context, sort_keys=True, default=str).encode()
    ).hexdigest()

    # Check cache
    if not force_refresh:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT suggestions FROM recommendation_cache WHERE context_hash = ?",
                (context_hash,),
            ).fetchone()
            if row is not None:
                suggestions = json.loads(row["suggestions"])
                return {
                    "recommendations": suggestions,
                    "source": "ollama",
                    "cached": True,
                }

    # Build prompt and call Ollama
    prompt = build_recommendation_prompt(context)
    try:
        resp = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        llm_response = resp.json().get("response", "")
    except Exception:
        return {
            "recommendations": [],
            "reason": "ollama_error",
            "source": "ollama",
        }

    titles = parse_llm_suggestions(llm_response)
    suggestions = resolve_suggestions(titles, db)

    # Cache the resolved suggestions
    with db._connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO recommendation_cache "
            "(context_hash, suggestions, created_at) "
            "VALUES (?, ?, ?)",
            (
                context_hash,
                json.dumps(suggestions),
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            ),
        )
        conn.commit()

    return {
        "recommendations": suggestions,
        "source": "ollama",
        "cached": False,
    }
