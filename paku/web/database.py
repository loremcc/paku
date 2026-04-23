from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# --- Pydantic models ---

USER_STATUSES = ("Watching", "Completed", "Plan to Watch", "Dropped", "On Hold")


class AnimeEntry(BaseModel):
    """One row of the anime_entries table, exposed over the API."""

    id: int
    anilist_id: int | None = None
    canonical_title: str | None = None
    raw_title: str
    romaji: str | None = None
    native_title: str | None = None
    media_format: str | None = None
    source: str | None = None
    country_of_origin: str | None = None
    debut_year: int | None = None
    studios: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    score: float | None = None
    episodes: int | None = None
    status: str | None = None
    user_status: str = "Plan to Watch"
    cover_image: str | None = None
    banner_image: str | None = None
    anilist_url: str | None = None
    confidence: float = 0.0
    needs_review: bool = False
    source_screenshot: str | None = None
    extraction_mode: str | None = None
    added_at: str | None = None


class AnimeListResponse(BaseModel):
    items: list[AnimeEntry]
    total: int
    page: int
    per_page: int


class DashboardStats(BaseModel):
    total: int
    by_user_status: dict[str, int]
    by_genre: dict[str, int]
    recent_additions: list[AnimeEntry]
    needs_review_count: int


class StatusUpdate(BaseModel):
    user_status: str


# --- Schema ---

_SCHEMA = """
CREATE TABLE IF NOT EXISTS anime_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anilist_id INTEGER UNIQUE,
    canonical_title TEXT,
    raw_title TEXT NOT NULL,
    romaji TEXT,
    native_title TEXT,
    media_format TEXT,
    source TEXT,
    country_of_origin TEXT,
    debut_year INTEGER,
    studios TEXT,
    genres TEXT,
    score REAL,
    episodes INTEGER,
    status TEXT,
    user_status TEXT NOT NULL DEFAULT 'Plan to Watch',
    cover_image TEXT,
    banner_image TEXT,
    anilist_url TEXT,
    confidence REAL NOT NULL DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0,
    source_screenshot TEXT,
    extraction_mode TEXT,
    added_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_anime_user_status ON anime_entries(user_status);
CREATE INDEX IF NOT EXISTS idx_anime_canonical_title ON anime_entries(canonical_title);
CREATE INDEX IF NOT EXISTS idx_anime_raw_title ON anime_entries(raw_title);

CREATE TABLE IF NOT EXISTS screenshots_processed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    content_type TEXT,
    processed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    result_count INTEGER NOT NULL DEFAULT 0
);
"""


# --- Database access ---


class Database:
    """Thin sqlite3 wrapper. All methods are synchronous — FastAPI runs them in a threadpool."""

    def __init__(self, path: str | Path = "paku_web.db") -> None:
        self.path = str(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # --- anime_entries ---

    def insert_or_update_anime(self, extraction: dict[str, Any]) -> tuple[int, bool]:
        """Insert an anime extraction. Dedup: on anilist_id, else lowercased canonical/raw title.

        Returns (row_id, created). `created` is False when an existing row matched
        (row is updated only when new confidence is higher).
        """
        row = _extraction_to_row(extraction)
        with self._connect() as conn:
            cur = conn.cursor()

            existing = None
            if row["anilist_id"] is not None:
                cur.execute(
                    "SELECT id, confidence FROM anime_entries WHERE anilist_id = ?",
                    (row["anilist_id"],),
                )
                existing = cur.fetchone()
            else:
                dedup_key = (row["canonical_title"] or row["raw_title"] or "").lower().strip()
                if dedup_key:
                    cur.execute(
                        "SELECT id, confidence FROM anime_entries "
                        "WHERE anilist_id IS NULL AND "
                        "LOWER(COALESCE(canonical_title, raw_title)) = ?",
                        (dedup_key,),
                    )
                    existing = cur.fetchone()

            if existing is not None:
                existing_id = existing["id"]
                if row["confidence"] > (existing["confidence"] or 0):
                    cur.execute(
                        _UPDATE_SQL,
                        {**row, "id": existing_id},
                    )
                    conn.commit()
                return existing_id, False

            cur.execute(_INSERT_SQL, row)
            conn.commit()
            return cur.lastrowid, True

    def get_anime(self, anime_id: int) -> AnimeEntry | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM anime_entries WHERE id = ?", (anime_id,))
            row = cur.fetchone()
        return _row_to_entry(row) if row else None

    def update_user_status(self, anime_id: int, user_status: str) -> AnimeEntry | None:
        if user_status not in USER_STATUSES:
            raise ValueError(f"Invalid user_status: {user_status!r}. Must be one of {USER_STATUSES}")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE anime_entries SET user_status = ? WHERE id = ?",
                (user_status, anime_id),
            )
            if cur.rowcount == 0:
                return None
            conn.commit()
        return self.get_anime(anime_id)

    def clear_needs_review(self, anime_id: int) -> AnimeEntry | None:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE anime_entries SET needs_review = 0 WHERE id = ?",
                (anime_id,),
            )
            if cur.rowcount == 0:
                return None
            conn.commit()
        return self.get_anime(anime_id)

    def delete_anime(self, anime_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM anime_entries WHERE id = ?", (anime_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_anime(
        self,
        *,
        user_status: str | None = None,
        genre: str | None = None,
        needs_review: bool | None = None,
        search: str | None = None,
        sort: str = "added_at",
        order: str = "desc",
        page: int = 1,
        per_page: int = 60,
    ) -> AnimeListResponse:
        allowed_sort = {"title", "score", "debut_year", "added_at"}
        if sort not in allowed_sort:
            sort = "added_at"
        sort_col = "COALESCE(canonical_title, raw_title)" if sort == "title" else sort
        order_sql = "ASC" if order.lower() == "asc" else "DESC"

        where: list[str] = []
        params: list[Any] = []
        if user_status:
            where.append("user_status = ?")
            params.append(user_status)
        if needs_review is not None:
            where.append("needs_review = ?")
            params.append(1 if needs_review else 0)
        if genre:
            where.append("genres LIKE ?")
            params.append(f'%"{genre}"%')
        if search:
            where.append(
                "(LOWER(COALESCE(canonical_title, '')) LIKE ? "
                "OR LOWER(raw_title) LIKE ? "
                "OR LOWER(COALESCE(romaji, '')) LIKE ?)"
            )
            needle = f"%{search.lower()}%"
            params.extend([needle, needle, needle])

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        page = max(page, 1)
        per_page = max(min(per_page, 500), 1)
        offset = (page - 1) * per_page

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS c FROM anime_entries {where_sql}", params
            ).fetchone()["c"]
            rows = conn.execute(
                f"SELECT * FROM anime_entries {where_sql} "
                f"ORDER BY {sort_col} {order_sql} LIMIT ? OFFSET ?",
                [*params, per_page, offset],
            ).fetchall()

        return AnimeListResponse(
            items=[_row_to_entry(r) for r in rows],
            total=total,
            page=page,
            per_page=per_page,
        )

    def stats(self) -> DashboardStats:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM anime_entries").fetchone()["c"]
            status_rows = conn.execute(
                "SELECT user_status, COUNT(*) AS c FROM anime_entries GROUP BY user_status"
            ).fetchall()
            needs_review = conn.execute(
                "SELECT COUNT(*) AS c FROM anime_entries WHERE needs_review = 1"
            ).fetchone()["c"]
            recent = conn.execute(
                "SELECT * FROM anime_entries ORDER BY added_at DESC LIMIT 10"
            ).fetchall()
            genre_rows = conn.execute(
                "SELECT genres FROM anime_entries WHERE genres IS NOT NULL AND genres != ''"
            ).fetchall()

        by_status = {s: 0 for s in USER_STATUSES}
        for r in status_rows:
            by_status[r["user_status"]] = r["c"]

        by_genre: dict[str, int] = {}
        for r in genre_rows:
            for g in _parse_json_list(r["genres"]):
                by_genre[g] = by_genre.get(g, 0) + 1

        return DashboardStats(
            total=total,
            by_user_status=by_status,
            by_genre=dict(sorted(by_genre.items(), key=lambda kv: kv[1], reverse=True)),
            recent_additions=[_row_to_entry(r) for r in recent],
            needs_review_count=needs_review,
        )

    # --- screenshots_processed ---

    def record_screenshot(self, path: str, content_type: str, result_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO screenshots_processed (path, content_type, processed_at, result_count) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "content_type = excluded.content_type, "
                "processed_at = excluded.processed_at, "
                "result_count = excluded.result_count",
                (path, content_type, _utc_now(), result_count),
            )
            conn.commit()


# --- helpers ---


_ROW_COLUMNS = [
    "anilist_id",
    "canonical_title",
    "raw_title",
    "romaji",
    "native_title",
    "media_format",
    "source",
    "country_of_origin",
    "debut_year",
    "studios",
    "genres",
    "score",
    "episodes",
    "status",
    "user_status",
    "cover_image",
    "banner_image",
    "anilist_url",
    "confidence",
    "needs_review",
    "source_screenshot",
    "extraction_mode",
]

_INSERT_SQL = (
    f"INSERT INTO anime_entries ({', '.join(_ROW_COLUMNS)}) "
    f"VALUES ({', '.join(':' + c for c in _ROW_COLUMNS)})"
)

_UPDATE_SQL = (
    "UPDATE anime_entries SET "
    + ", ".join(f"{c} = :{c}" for c in _ROW_COLUMNS)
    + " WHERE id = :id"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    return []


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _extraction_to_row(extraction: dict[str, Any]) -> dict[str, Any]:
    """Flatten an AnimeExtractionResult-like dict into DB column values."""
    studios = extraction.get("studios") or []
    if isinstance(studios, str):
        studios = _split_csv(studios)
    genres = extraction.get("genres") or []
    if isinstance(genres, str):
        genres = _parse_json_list(genres)

    return {
        "anilist_id": extraction.get("anilist_id"),
        "canonical_title": extraction.get("canonical_title"),
        "raw_title": extraction.get("raw_title") or extraction.get("canonical_title") or "",
        "romaji": extraction.get("romaji"),
        "native_title": extraction.get("native_title"),
        "media_format": extraction.get("media_format"),
        "source": extraction.get("source"),
        "country_of_origin": extraction.get("country_of_origin"),
        "debut_year": extraction.get("debut_year"),
        "studios": ",".join(studios) if studios else None,
        "genres": json.dumps(list(genres)) if genres else None,
        "score": extraction.get("score"),
        "episodes": extraction.get("episodes"),
        "status": extraction.get("status"),
        "user_status": extraction.get("user_status", "Plan to Watch"),
        "cover_image": extraction.get("cover_image"),
        "banner_image": extraction.get("banner_image"),
        "anilist_url": extraction.get("anilist_url"),
        "confidence": float(extraction.get("confidence") or 0.0),
        "needs_review": 1 if extraction.get("needs_review") else 0,
        "source_screenshot": extraction.get("source_screenshot"),
        "extraction_mode": extraction.get("extraction_mode"),
    }


def _row_to_entry(row: sqlite3.Row) -> AnimeEntry:
    return AnimeEntry(
        id=row["id"],
        anilist_id=row["anilist_id"],
        canonical_title=row["canonical_title"],
        raw_title=row["raw_title"],
        romaji=row["romaji"],
        native_title=row["native_title"],
        media_format=row["media_format"],
        source=row["source"],
        country_of_origin=row["country_of_origin"],
        debut_year=row["debut_year"],
        studios=_split_csv(row["studios"]),
        genres=_parse_json_list(row["genres"]),
        score=row["score"],
        episodes=row["episodes"],
        status=row["status"],
        user_status=row["user_status"],
        cover_image=row["cover_image"],
        banner_image=row["banner_image"],
        anilist_url=row["anilist_url"],
        confidence=row["confidence"] or 0.0,
        needs_review=bool(row["needs_review"]),
        source_screenshot=row["source_screenshot"],
        extraction_mode=row["extraction_mode"],
        added_at=row["added_at"],
    )


def _iter_anime_extractions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the pipeline's plural-first-with-fallback anime extraction contract.

    Pipeline produces `extractions` (plural) only when >1 titles were matched;
    `extraction` (singular) is always set. Prefer plural when present and non-empty,
    otherwise fall back to singular. Non-list / non-dict entries are dropped rather
    than crashing the whole ingest.
    """
    plural = result.get("extractions")
    if isinstance(plural, list) and plural:
        return [e for e in plural if isinstance(e, dict) and e]

    singular = result.get("extraction")
    if isinstance(singular, dict) and singular:
        return [singular]
    return []


def ingest_pipeline_result(db: Database, result: dict[str, Any]) -> list[AnimeEntry]:
    """Take a `process_image()` result dict and write all anime extractions to the DB.

    Returns the list of AnimeEntry rows for the newly-stored or matched records.
    Non-anime content is ignored (dashboard is anime-first per spec).
    """
    stored: list[AnimeEntry] = []
    if result.get("content_type") != "anime":
        db.record_screenshot(
            path=result.get("screenshot", ""),
            content_type=result.get("content_type", "unknown"),
            result_count=0,
        )
        return stored

    for extraction in _iter_anime_extractions(result):
        anime_id, _created = db.insert_or_update_anime(extraction)
        entry = db.get_anime(anime_id)
        if entry is not None:
            stored.append(entry)

    db.record_screenshot(
        path=result.get("screenshot", ""),
        content_type="anime",
        result_count=len(stored),
    )
    return stored
