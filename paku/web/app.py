from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..context import AppContext
from ..extractors.anime import _ANILIST_URL, _COUNTRY_MAP
from ..pipeline import process_image
from .database import (
    USER_STATUSES,
    AnimeEntry,
    AnimeListResponse,
    Database,
    DashboardStats,
    StatusUpdate,
    ingest_pipeline_result,
)

_STATIC_DIR = Path(__file__).parent / "static"


class SearchResult(BaseModel):
    anilist_id: int
    english: str | None = None
    romaji: str | None = None
    native: str | None = None
    cover_image: str | None = None
    media_format: str | None = None
    episodes: int | None = None
    status: str | None = None
    genres: list[str] = []
    score: float | None = None
    debut_year: int | None = None
    anilist_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class SearchAddRequest(BaseModel):
    anilist_id: int


class DigestResponse(BaseModel):
    content_type: str
    stored: list[AnimeEntry]
    extraction: dict[str, Any] | None = None
    extractions: list[dict[str, Any]] | None = None
    needs_review: bool = False
    screen_type: str | None = None
    engine: str | None = None


def create_app(db_path: str | Path = "paku_web.db") -> FastAPI:
    """Build and return the FastAPI app. Factory is used so tests can pass an isolated DB path."""
    app = FastAPI(title="paku dashboard", version="0.6.0")
    db = Database(db_path)
    app.state.db = db

    # --- Frontend ---

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        index_html = _STATIC_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(500, "Frontend not installed")
        return FileResponse(str(index_html))

    # --- Digest ---

    @app.post("/api/digest", response_model=DigestResponse)
    async def digest_endpoint(
        file: UploadFile = File(...),
        mode: str = Query("auto", pattern="^(auto|anime|url|recipe)$"),
        smart: bool = Query(False),
    ) -> DigestResponse:
        try:
            contents = await file.read()
        except Exception as exc:
            raise HTTPException(422, f"Could not read upload: {exc}")
        if not contents:
            raise HTTPException(422, "Empty file upload")

        suffix = Path(file.filename or "upload.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            result = process_image(
                image_path=tmp_path,
                mode=mode,
                smart=smart,
                outputs=[],
            )
        except Exception as exc:
            raise HTTPException(422, f"Pipeline error: {exc}")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        if result is None:
            raise HTTPException(422, "Image could not be processed (unreadable or too dark)")

        # Use the original filename as the canonical screenshot path for dedup/display.
        result["screenshot"] = file.filename or result.get("screenshot", "")
        extraction = result.get("extraction") or {}
        for ex in (result.get("extractions") or [extraction]):
            if isinstance(ex, dict) and not ex.get("source_screenshot"):
                ex["source_screenshot"] = result["screenshot"]

        stored = ingest_pipeline_result(db, result)
        needs_review = any(e.needs_review for e in stored) or bool(
            extraction.get("needs_review")
        )

        return DigestResponse(
            content_type=result.get("content_type", "unknown"),
            stored=stored,
            extraction=result.get("extraction"),
            extractions=result.get("extractions"),
            needs_review=needs_review,
            screen_type=result.get("screen_type"),
            engine=result.get("engine"),
        )

    # --- Collection ---

    @app.get("/api/collection", response_model=AnimeListResponse)
    def list_collection(
        user_status: str | None = Query(None, alias="status"),
        genre: str | None = None,
        needs_review: bool | None = None,
        search: str | None = None,
        sort: str = Query("added_at"),
        order: str = Query("desc"),
        page: int = Query(1, ge=1),
        per_page: int = Query(60, ge=1, le=500),
    ) -> AnimeListResponse:
        return db.list_anime(
            user_status=user_status,
            genre=genre,
            needs_review=needs_review,
            search=search,
            sort=sort,
            order=order,
            page=page,
            per_page=per_page,
        )

    @app.get("/api/collection/{anime_id}", response_model=AnimeEntry)
    def get_collection_item(anime_id: int) -> AnimeEntry:
        entry = db.get_anime(anime_id)
        if entry is None:
            raise HTTPException(404, f"Anime id {anime_id} not found")
        return entry

    @app.patch("/api/collection/{anime_id}", response_model=AnimeEntry)
    def patch_collection_item(anime_id: int, update: StatusUpdate) -> AnimeEntry:
        if update.user_status not in USER_STATUSES:
            raise HTTPException(
                422,
                f"Invalid user_status: {update.user_status!r}. Must be one of {list(USER_STATUSES)}",
            )
        entry = db.update_user_status(anime_id, update.user_status)
        if entry is None:
            raise HTTPException(404, f"Anime id {anime_id} not found")
        return entry

    @app.post("/api/collection/{anime_id}/review/clear", response_model=AnimeEntry)
    def clear_review(anime_id: int) -> AnimeEntry:
        entry = db.clear_needs_review(anime_id)
        if entry is None:
            raise HTTPException(404, f"Anime id {anime_id} not found")
        return entry

    @app.delete("/api/collection/{anime_id}")
    def delete_collection_item(anime_id: int) -> JSONResponse:
        if not db.delete_anime(anime_id):
            raise HTTPException(404, f"Anime id {anime_id} not found")
        return JSONResponse({"deleted": anime_id})

    # --- Search ---

    @app.get("/api/search", response_model=SearchResponse)
    def search_anilist(q: str = Query(..., min_length=1)) -> SearchResponse:
        results = _search_anilist_page(q, limit=10)
        return SearchResponse(query=q, results=results)

    @app.post("/api/search/add", response_model=AnimeEntry)
    def add_from_search(request: SearchAddRequest) -> AnimeEntry:
        media = _fetch_anilist_by_id(request.anilist_id)
        if media is None:
            raise HTTPException(404, f"AniList id {request.anilist_id} not found")
        extraction = _media_to_extraction(media)
        anime_id, _created = db.insert_or_update_anime(extraction)
        entry = db.get_anime(anime_id)
        if entry is None:
            raise HTTPException(500, "Failed to persist entry")
        return entry

    # --- Stats ---

    @app.get("/api/stats", response_model=DashboardStats)
    def stats_endpoint() -> DashboardStats:
        return db.stats()

    return app


# --- AniList helpers (search endpoint, independent from extractor's single-best query) ---

_PAGE_QUERY = """
query ($search: String, $perPage: Int) {
  Page(page: 1, perPage: $perPage) {
    media(search: $search, sort: SEARCH_MATCH) {
      id
      title { english romaji native }
      type format
      episodes status genres averageScore siteUrl
      countryOfOrigin
      startDate { year }
      coverImage { extraLarge large }
      bannerImage
      source
      studios {
        edges {
          node { name isAnimationStudio }
        }
      }
    }
  }
}
"""

_ID_QUERY = """
query ($id: Int) {
  Media(id: $id) {
    id
    title { english romaji native }
    type format
    episodes status genres averageScore siteUrl
    countryOfOrigin
    startDate { year }
    coverImage { extraLarge large }
    bannerImage
    source
    studios {
      edges {
        node { name isAnimationStudio }
      }
    }
  }
}
"""


def _search_anilist_page(query: str, limit: int = 10) -> list[SearchResult]:
    try:
        resp = requests.post(
            _ANILIST_URL,
            json={"query": _PAGE_QUERY, "variables": {"search": query, "perPage": limit}},
            timeout=10,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(502, f"AniList request failed: {exc}")

    page = (data.get("data") or {}).get("Page") or {}
    media_list = page.get("media") or []
    return [_media_to_search_result(m) for m in media_list]


def _fetch_anilist_by_id(anilist_id: int) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            _ANILIST_URL,
            json={"query": _ID_QUERY, "variables": {"id": anilist_id}},
            timeout=10,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(502, f"AniList request failed: {exc}")

    return (data.get("data") or {}).get("Media")


def _media_to_search_result(media: dict[str, Any]) -> SearchResult:
    title = media.get("title") or {}
    cover = media.get("coverImage") or {}
    start = media.get("startDate") or {}
    score = media.get("averageScore")
    return SearchResult(
        anilist_id=media["id"],
        english=title.get("english"),
        romaji=title.get("romaji"),
        native=title.get("native"),
        cover_image=cover.get("extraLarge") or cover.get("large"),
        media_format=media.get("format"),
        episodes=media.get("episodes"),
        status=media.get("status"),
        genres=media.get("genres") or [],
        score=(score / 10.0) if isinstance(score, (int, float)) else None,
        debut_year=start.get("year"),
        anilist_url=media.get("siteUrl"),
    )


def _media_to_extraction(media: dict[str, Any]) -> dict[str, Any]:
    """Convert an AniList media dict into an extraction-shaped dict for db.insert_or_update_anime."""
    title = media.get("title") or {}
    cover = media.get("coverImage") or {}
    start = media.get("startDate") or {}
    score = media.get("averageScore")
    canonical = title.get("english") or title.get("romaji") or title.get("native") or ""
    studios = [
        edge["node"]["name"]
        for edge in (media.get("studios") or {}).get("edges", [])
        if edge.get("node", {}).get("isAnimationStudio")
    ]
    country_code = media.get("countryOfOrigin")
    country = _COUNTRY_MAP.get(country_code, country_code) if country_code else None

    return {
        "anilist_id": media["id"],
        "canonical_title": canonical,
        "raw_title": canonical,
        "romaji": title.get("romaji"),
        "native_title": title.get("native"),
        "media_format": media.get("format"),
        "source": media.get("source"),
        "country_of_origin": country,
        "debut_year": start.get("year"),
        "studios": studios,
        "genres": media.get("genres") or [],
        "score": (score / 10.0) if isinstance(score, (int, float)) else None,
        "episodes": media.get("episodes"),
        "status": media.get("status"),
        "cover_image": cover.get("extraLarge") or cover.get("large"),
        "banner_image": media.get("bannerImage"),
        "anilist_url": media.get("siteUrl"),
        "confidence": 1.0,
        "needs_review": False,
        "source_screenshot": None,
        "extraction_mode": "manual_search",
    }


def run_server(host: str = "127.0.0.1", port: int = 8000, db_path: str = "paku_web.db") -> None:
    """Launch uvicorn. Called by `paku serve`."""
    import uvicorn

    AppContext.instance()  # validate config before binding
    app = create_app(db_path=db_path)
    uvicorn.run(app, host=host, port=port, log_level="info")
