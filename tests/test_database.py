from __future__ import annotations

from pathlib import Path

import pytest

from paku.web.database import (
    USER_STATUSES,
    Database,
    ingest_pipeline_result,
)


def _extraction(
    *,
    anilist_id: int | None = 1,
    canonical: str = "Frieren",
    raw: str | None = None,
    confidence: float = 0.95,
    needs_review: bool = False,
    score: float | None = 9.2,
    genres: list[str] | None = None,
    studios: list[str] | None = None,
    debut_year: int | None = 2023,
    user_status: str | None = None,
) -> dict:
    d: dict = {
        "anilist_id": anilist_id,
        "canonical_title": canonical,
        "raw_title": raw or canonical,
        "romaji": canonical,
        "native_title": None,
        "media_format": "TV",
        "source": "MANGA",
        "country_of_origin": "Japan",
        "debut_year": debut_year,
        "studios": studios or ["Madhouse"],
        "genres": genres or ["Adventure", "Fantasy"],
        "score": score,
        "episodes": 28,
        "status": "FINISHED",
        "cover_image": "https://example.com/cover.jpg",
        "banner_image": None,
        "anilist_url": f"https://anilist.co/anime/{anilist_id or 0}",
        "confidence": confidence,
        "needs_review": needs_review,
        "source_screenshot": "screen.png",
        "extraction_mode": "fast",
    }
    if user_status is not None:
        d["user_status"] = user_status
    return d


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


class TestInsertAndRetrieve:
    def test_insert_then_get(self, db: Database) -> None:
        anime_id, created = db.insert_or_update_anime(_extraction(anilist_id=101))
        assert created is True
        entry = db.get_anime(anime_id)
        assert entry is not None
        assert entry.anilist_id == 101
        assert entry.canonical_title == "Frieren"
        assert entry.genres == ["Adventure", "Fantasy"]
        assert entry.studios == ["Madhouse"]
        assert entry.user_status == "Plan to Watch"

    def test_get_missing_returns_none(self, db: Database) -> None:
        assert db.get_anime(999) is None


class TestDedup:
    def test_same_anilist_id_dedups(self, db: Database) -> None:
        id1, c1 = db.insert_or_update_anime(_extraction(anilist_id=42, confidence=0.8))
        id2, c2 = db.insert_or_update_anime(_extraction(anilist_id=42, confidence=0.9))
        assert id1 == id2
        assert c1 is True and c2 is False

        resp = db.list_anime()
        assert resp.total == 1

    def test_update_only_when_confidence_higher(self, db: Database) -> None:
        id1, _ = db.insert_or_update_anime(
            _extraction(anilist_id=7, canonical="Original", confidence=0.95)
        )
        db.insert_or_update_anime(
            _extraction(anilist_id=7, canonical="Lower conf", confidence=0.3)
        )
        entry = db.get_anime(id1)
        assert entry is not None
        assert entry.canonical_title == "Original"

        db.insert_or_update_anime(
            _extraction(anilist_id=7, canonical="Higher conf", confidence=0.99)
        )
        entry = db.get_anime(id1)
        assert entry is not None
        assert entry.canonical_title == "Higher conf"

    def test_null_anilist_id_dedups_on_title(self, db: Database) -> None:
        id1, c1 = db.insert_or_update_anime(
            _extraction(anilist_id=None, canonical="Unmatched Title")
        )
        id2, c2 = db.insert_or_update_anime(
            _extraction(anilist_id=None, canonical="unmatched title")
        )
        assert id1 == id2
        assert c1 is True and c2 is False

    def test_null_anilist_id_different_titles_both_inserted(self, db: Database) -> None:
        db.insert_or_update_anime(_extraction(anilist_id=None, canonical="Alpha"))
        db.insert_or_update_anime(_extraction(anilist_id=None, canonical="Beta"))
        assert db.list_anime().total == 2


class TestUserStatus:
    def test_update_user_status(self, db: Database) -> None:
        anime_id, _ = db.insert_or_update_anime(_extraction(anilist_id=5))
        entry = db.update_user_status(anime_id, "Watching")
        assert entry is not None
        assert entry.user_status == "Watching"

    def test_update_user_status_invalid_raises(self, db: Database) -> None:
        anime_id, _ = db.insert_or_update_anime(_extraction(anilist_id=6))
        with pytest.raises(ValueError):
            db.update_user_status(anime_id, "NotAStatus")

    def test_update_user_status_missing_returns_none(self, db: Database) -> None:
        assert db.update_user_status(999, "Watching") is None

    def test_filter_by_user_status(self, db: Database) -> None:
        id1, _ = db.insert_or_update_anime(_extraction(anilist_id=1, canonical="A"))
        id2, _ = db.insert_or_update_anime(_extraction(anilist_id=2, canonical="B"))
        db.insert_or_update_anime(_extraction(anilist_id=3, canonical="C"))
        db.update_user_status(id1, "Watching")
        db.update_user_status(id2, "Watching")

        resp = db.list_anime(user_status="Watching")
        assert resp.total == 2
        assert {e.canonical_title for e in resp.items} == {"A", "B"}

        resp = db.list_anime(user_status="Plan to Watch")
        assert resp.total == 1
        assert resp.items[0].canonical_title == "C"


class TestSearchAndFilter:
    def test_search_title_substring(self, db: Database) -> None:
        db.insert_or_update_anime(_extraction(anilist_id=1, canonical="Attack on Titan"))
        db.insert_or_update_anime(_extraction(anilist_id=2, canonical="Frieren"))
        db.insert_or_update_anime(_extraction(anilist_id=3, canonical="Spy x Family"))

        resp = db.list_anime(search="titan")
        assert resp.total == 1
        assert resp.items[0].canonical_title == "Attack on Titan"

    def test_filter_by_genre(self, db: Database) -> None:
        db.insert_or_update_anime(
            _extraction(anilist_id=1, canonical="A", genres=["Action", "Drama"])
        )
        db.insert_or_update_anime(
            _extraction(anilist_id=2, canonical="B", genres=["Comedy", "Slice of Life"])
        )

        resp = db.list_anime(genre="Action")
        assert resp.total == 1
        assert resp.items[0].canonical_title == "A"

        resp = db.list_anime(genre="Slice of Life")
        assert resp.total == 1

    def test_filter_needs_review(self, db: Database) -> None:
        db.insert_or_update_anime(_extraction(anilist_id=1, canonical="Clean"))
        db.insert_or_update_anime(
            _extraction(anilist_id=2, canonical="Needs", needs_review=True)
        )

        assert db.list_anime(needs_review=True).total == 1
        assert db.list_anime(needs_review=False).total == 1

    def test_sort_order(self, db: Database) -> None:
        db.insert_or_update_anime(
            _extraction(anilist_id=1, canonical="B Second", score=7.0)
        )
        db.insert_or_update_anime(
            _extraction(anilist_id=2, canonical="A First", score=9.0)
        )

        resp = db.list_anime(sort="title", order="asc")
        assert [e.canonical_title for e in resp.items] == ["A First", "B Second"]

        resp = db.list_anime(sort="score", order="desc")
        assert resp.items[0].canonical_title == "A First"

    def test_pagination(self, db: Database) -> None:
        for i in range(5):
            db.insert_or_update_anime(
                _extraction(anilist_id=i + 1, canonical=f"Title {i:02d}")
            )

        page1 = db.list_anime(page=1, per_page=2)
        assert page1.total == 5
        assert len(page1.items) == 2
        page3 = db.list_anime(page=3, per_page=2)
        assert len(page3.items) == 1


class TestClearReview:
    def test_clear_needs_review(self, db: Database) -> None:
        anime_id, _ = db.insert_or_update_anime(
            _extraction(anilist_id=1, needs_review=True)
        )
        entry = db.clear_needs_review(anime_id)
        assert entry is not None
        assert entry.needs_review is False

    def test_clear_missing_returns_none(self, db: Database) -> None:
        assert db.clear_needs_review(999) is None


class TestDelete:
    def test_delete_anime(self, db: Database) -> None:
        anime_id, _ = db.insert_or_update_anime(_extraction(anilist_id=1))
        assert db.delete_anime(anime_id) is True
        assert db.get_anime(anime_id) is None

    def test_delete_missing_returns_false(self, db: Database) -> None:
        assert db.delete_anime(999) is False


class TestStats:
    def test_stats_counts(self, db: Database) -> None:
        id1, _ = db.insert_or_update_anime(
            _extraction(anilist_id=1, canonical="A", genres=["Action"])
        )
        db.insert_or_update_anime(
            _extraction(
                anilist_id=2, canonical="B", needs_review=True, genres=["Action", "Drama"]
            )
        )
        db.update_user_status(id1, "Completed")

        stats = db.stats()
        assert stats.total == 2
        assert stats.needs_review_count == 1
        assert stats.by_user_status["Completed"] == 1
        assert stats.by_user_status["Plan to Watch"] == 1
        assert all(s in stats.by_user_status for s in USER_STATUSES)
        assert stats.by_genre["Action"] == 2
        assert stats.by_genre["Drama"] == 1
        assert len(stats.recent_additions) == 2


class TestIngestPipelineResult:
    def test_anime_single_extraction(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "anime",
            "extraction": _extraction(anilist_id=500),
        }
        stored = ingest_pipeline_result(db, result)
        assert len(stored) == 1
        assert stored[0].anilist_id == 500

    def test_anime_multi_extractions(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "anime",
            "extractions": [
                _extraction(anilist_id=10, canonical="One"),
                _extraction(anilist_id=11, canonical="Two"),
            ],
        }
        stored = ingest_pipeline_result(db, result)
        assert len(stored) == 2
        assert db.list_anime().total == 2

    def test_non_anime_is_skipped(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "url",
            "extraction": {"resolved_url": "https://example.com"},
        }
        stored = ingest_pipeline_result(db, result)
        assert stored == []
        assert db.list_anime().total == 0

    def test_idempotent_same_screenshot(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "anime",
            "extraction": _extraction(anilist_id=77),
        }
        ingest_pipeline_result(db, result)
        ingest_pipeline_result(db, result)
        assert db.list_anime().total == 1

    def test_empty_plural_falls_back_to_singular(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "anime",
            "extraction": _extraction(anilist_id=800, canonical="Solo"),
            "extractions": [],
        }
        stored = ingest_pipeline_result(db, result)
        assert len(stored) == 1
        assert stored[0].anilist_id == 800

    def test_plural_preferred_over_singular(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "anime",
            "extraction": _extraction(anilist_id=900, canonical="First"),
            "extractions": [
                _extraction(anilist_id=901, canonical="Alpha"),
                _extraction(anilist_id=902, canonical="Beta"),
            ],
        }
        stored = ingest_pipeline_result(db, result)
        assert {e.anilist_id for e in stored} == {901, 902}
        assert db.list_anime().total == 2

    def test_malformed_entries_are_skipped(self, db: Database) -> None:
        result = {
            "screenshot": "shot.png",
            "content_type": "anime",
            "extractions": [
                None,
                {},
                "not a dict",
                _extraction(anilist_id=950, canonical="Valid"),
            ],
        }
        stored = ingest_pipeline_result(db, result)
        assert len(stored) == 1
        assert stored[0].anilist_id == 950

    def test_no_extraction_keys_ingests_nothing(self, db: Database) -> None:
        result = {"screenshot": "shot.png", "content_type": "anime"}
        stored = ingest_pipeline_result(db, result)
        assert stored == []
        assert db.list_anime().total == 0
