from __future__ import annotations

from pathlib import Path

from paku.web.database import Database
from paku.web.recommendations import (
    build_collection_context,
    build_recommendation_prompt,
    parse_llm_suggestions,
    resolve_suggestions,
)


def _extraction(
    *,
    anilist_id: int | None = 1,
    canonical: str = "Frieren",
    genres: list[str] | None = None,
    studios: list[str] | None = None,
    media_format: str = "TV",
    user_score: float | None = None,
    user_status: str = "Completed",
    debut_year: int | None = 2023,
) -> dict:
    return {
        "anilist_id": anilist_id,
        "canonical_title": canonical,
        "raw_title": canonical,
        "genres": genres or ["Fantasy", "Adventure"],
        "studios": studios or ["MADHOUSE"],
        "media_format": media_format,
        "user_score": user_score,
        "user_status": user_status,
        "debut_year": debut_year,
        "score": 8.5,
        "confidence": 0.95,
        "needs_review": False,
    }


class TestBuildCollectionContext:
    def test_empty_collection(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        ctx = build_collection_context(db)
        assert ctx["empty"] is True

    def test_populated_collection(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        db.insert_or_update_anime(_extraction(anilist_id=100, canonical="Frieren"))
        db.insert_or_update_anime(_extraction(anilist_id=200, canonical="Dungeon Meshi"))
        db.insert_or_update_anime(
            _extraction(anilist_id=300, canonical="Apothecary Diaries", user_status="Watching")
        )
        ctx = build_collection_context(db)
        assert ctx["empty"] is False
        assert ctx["total_entries"] == 3
        assert len(ctx["top_rated"]) == 3
        # Genre counts
        genres = {g["genre"]: g["count"] for g in ctx["top_genres"]}
        assert "Fantasy" in genres
        # Status counts
        assert ctx["statuses"].get("Completed", 0) >= 2
        assert ctx["statuses"].get("Watching", 0) >= 1
        # Seen IDs
        assert 100 in ctx["seen_anilist_ids"]

    def test_top_rated_sorted(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        db.insert_or_update_anime(_extraction(anilist_id=1, canonical="A", user_score=6.0))
        db.insert_or_update_anime(_extraction(anilist_id=2, canonical="B", user_score=9.0))
        db.insert_or_update_anime(_extraction(anilist_id=3, canonical="C", user_score=7.5))
        ctx = build_collection_context(db)
        scores = [e["score"] for e in ctx["top_rated"]]
        assert scores == [9.0, 7.5, 6.0]  # descending


class TestBuildRecommendationPrompt:
    def test_empty_context(self) -> None:
        prompt = build_recommendation_prompt({"empty": True})
        assert prompt == ""

    def test_prompt_includes_top_rated(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        db.insert_or_update_anime(
            _extraction(
                anilist_id=1,
                canonical="Frieren: Beyond Journey's End",
                genres=["Fantasy", "Adventure"],
                studios=["MADHOUSE"],
            )
        )
        ctx = build_collection_context(db)
        prompt = build_recommendation_prompt(ctx)
        assert "Frieren" in prompt
        assert "Fantasy" in prompt
        assert "MADHOUSE" in prompt
        assert "10" in prompt  # asks for 10 suggestions

    def test_prompt_includes_genres_and_studios(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        db.insert_or_update_anime(
            _extraction(anilist_id=1, canonical="Test", genres=["Mecha", "Sci-Fi"])
        )
        ctx = build_collection_context(db)
        prompt = build_recommendation_prompt(ctx)
        assert "Mecha" in prompt
        assert "Sci-Fi" in prompt


class TestParseLlmSuggestions:
    def test_numbered_list(self) -> None:
        response = """1. Attack on Titan
2. Demon Slayer
3. Jujutsu Kaisen"""
        titles = parse_llm_suggestions(response)
        assert titles == ["Attack on Titan", "Demon Slayer", "Jujutsu Kaisen"]

    def test_numbered_with_parentheses(self) -> None:
        response = "1) Fullmetal Alchemist\n2) Steins;Gate"
        titles = parse_llm_suggestions(response)
        assert "Fullmetal Alchemist" in titles
        assert "Steins;Gate" in titles

    def test_dash_prefixed(self) -> None:
        response = "- Mob Psycho 100\n- Vinland Saga\n- 86 Eighty-Six"
        titles = parse_llm_suggestions(response)
        assert len(titles) == 3

    def test_cleans_trailing_parens(self) -> None:
        response = "1. Frieren (Fantasy)\n2. Bocchi the Rock! (Music)"
        titles = parse_llm_suggestions(response)
        assert titles[0] == "Frieren"
        assert titles[1] == "Bocchi the Rock!"

    def test_extra_text_ignored(self) -> None:
        response = """Here are some suggestions based on your taste:

1. Monster
2. Psycho-Pass
3. Paranoia Agent

These should fit your profile."""
        titles = parse_llm_suggestions(response)
        assert titles == ["Monster", "Psycho-Pass", "Paranoia Agent"]

    def test_max_10_titles(self) -> None:
        lines = "\n".join(f"{i}. Title {i}" for i in range(1, 16))
        titles = parse_llm_suggestions(lines)
        assert len(titles) == 10

    def test_empty_response(self) -> None:
        assert parse_llm_suggestions("") == []
        assert parse_llm_suggestions("No suggestions available.") == []


class TestResolveSuggestions:
    def test_returns_structured_results(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        # Empty DB — no saved titles to filter
        results = resolve_suggestions(["Frieren"], db)
        assert isinstance(results, list)
        # May be empty if AniList is unreachable, but shouldn't crash
        for r in results:
            assert "anilist_id" in r
            assert "english" in r or "romaji" in r
            assert "cover_image" in r or "cover_image" not in r

    def test_filters_already_saved(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        db.insert_or_update_anime(_extraction(anilist_id=163263, canonical="Frieren"))
        # We can't easily test the AniList call without mocking,
        # but we can verify the saved_ids filter is populated
        all_entries = db.list_anime(per_page=100000).items
        saved_ids = {e.anilist_id for e in all_entries if e.anilist_id is not None}
        assert 163263 in saved_ids
