from __future__ import annotations

import csv
from pathlib import Path

from paku.inputs.notion_import import (
    _clean_notion_urls,
    _normalize_column,
    _normalize_list,
    _normalize_score,
    _normalize_status,
    match_rows_to_collection,
    parse_notion_csv,
)


class TestNormalizeColumn:
    def test_english_title(self):
        assert _normalize_column("English Title") == "english_title"

    def test_romaji_title(self):
        assert _normalize_column("Romaji Title") == "romaji_title"

    def test_status(self):
        assert _normalize_column("Status") == "user_status"

    def test_score(self):
        assert _normalize_column("Score") == "user_score"

    def test_case_insensitive_and_whitespace(self):
        assert _normalize_column("  ENGLISH TITLE  ") == "english_title"

    def test_unknown_column_returns_none(self):
        assert _normalize_column("Unknown Column!") is None


class TestNormalizeStatus:
    def test_watching(self):
        assert _normalize_status("Watching") == "Watching"

    def test_completed(self):
        assert _normalize_status("Completed") == "Completed"

    def test_lowercase_input(self):
        assert _normalize_status("plan to watch") == "Plan to Watch"

    def test_not_started_maps_to_plan_to_watch(self):
        assert _normalize_status("Not Started") == "Plan to Watch"

    def test_unknown_status_passthrough(self):
        assert _normalize_status("Custom Status") == "Custom Status"

    def test_awaiting_sequel_maps_to_plan_to_watch(self):
        assert _normalize_status("Awaiting Sequel") == "Plan to Watch"

    def test_empty_returns_none(self):
        assert _normalize_status("") is None
        assert _normalize_status(None) is None


class TestCleanNotionUrls:
    def test_strips_single_url(self):
        result = _clean_notion_urls(
            "Action (https://www.notion.so/Action-bd460470cc5e4546862e2bdf19662377?pvs=21)"
        )
        assert result == "Action"

    def test_strips_multiple_urls_in_csv_list(self):
        result = _clean_notion_urls(
            "Action (https://www.notion.so/Action-abc?pvs=21), "
            "Mecha (https://www.notion.so/Mecha-def?pvs=21)"
        )
        assert result == "Action, Mecha"

    def test_passthrough_clean_value(self):
        assert _clean_notion_urls("MADHOUSE") == "MADHOUSE"


class TestNormalizeScore:
    def test_integer_score(self):
        assert _normalize_score("8") == 8.0

    def test_decimal_score(self):
        assert _normalize_score("7.5") == 7.5

    def test_empty_returns_none(self):
        assert _normalize_score("") is None

    def test_non_numeric_returns_none(self):
        assert _normalize_score("N/A") is None


class TestNormalizeList:
    def test_comma_separated(self):
        assert _normalize_list("MADHOUSE, MAPPA") == ["MADHOUSE", "MAPPA"]

    def test_single_value(self):
        assert _normalize_list("MADHOUSE") == ["MADHOUSE"]

    def test_empty(self):
        assert _normalize_list("") is None
        assert _normalize_list(None) is None

    def test_trims_whitespace(self):
        assert _normalize_list("  MADHOUSE ,  MAPPA  ") == ["MADHOUSE", "MAPPA"]


class TestParseNotionCsv:
    def _write_csv(self, tmp_path: Path, rows: list[dict], *, bom: bool = False) -> Path:
        p = tmp_path / "test.csv"
        encoding = "utf-8-sig" if bom else "utf-8"
        with open(p, "w", newline="", encoding=encoding) as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return p

    def test_parses_english_title_and_status(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {"English Title": "Frieren", "Status": "Completed", "Score": "9"},
            ],
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 1
        assert rows[0]["english_title"] == "Frieren"
        assert rows[0]["user_status"] == "Completed"
        assert rows[0]["user_score"] == 9.0

    def test_parses_romaji_title(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {"Romaji Title": "Sousou no Frieren", "Status": "Watching"},
            ],
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 1
        assert rows[0]["romaji_title"] == "Sousou no Frieren"

    def test_handles_bom(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {"English Title": "Bocchi the Rock!", "Status": "Completed"},
            ],
            bom=True,
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 1
        assert rows[0]["english_title"] == "Bocchi the Rock!"

    def test_ignores_unknown_columns(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {
                    "English Title": "Mob Psycho 100",
                    "My Custom Field": "ignored",
                    "Status": "Completed",
                },
            ],
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 1
        assert "my_custom_field" not in rows[0]
        assert rows[0]["english_title"] == "Mob Psycho 100"

    def test_skips_rows_without_title(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {"Status": "Completed", "Score": "8"},
            ],
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 0

    def test_empty_csv(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("English Title,Romaji Title,Status\n")
        rows = parse_notion_csv(p)
        assert len(rows) == 0

    def test_csv_with_only_header(self, tmp_path: Path):
        p = tmp_path / "header_only.csv"
        p.write_text("English Title,Status,Score\n")
        rows = parse_notion_csv(p)
        assert len(rows) == 0

    def test_multiple_rows(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {"English Title": "Frieren", "Status": "Completed", "Score": "9"},
                {"English Title": "Dungeon Meshi", "Status": "Watching", "Score": "8.5"},
                {"English Title": "Apothecary Diaries", "Status": "Plan to Watch", "Score": ""},
            ],
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 3
        assert rows[0]["user_score"] == 9.0
        assert rows[1]["user_score"] == 8.5
        assert "user_score" not in rows[2]

    def test_parses_anilist_id_and_debut_year(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {
                    "English Title": "Frieren",
                    "Status": "Completed",
                    "AniList ID": "163263",
                    "Debut Year": "2023",
                },
            ],
        )
        rows = parse_notion_csv(p)
        assert rows[0]["anilist_id"] == 163263
        assert rows[0]["debut_year"] == 2023

    def test_handles_invalid_numeric_fields(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {
                    "English Title": "Test",
                    "Status": "Watching",
                    "Score": "N/A",
                    "AniList ID": "abc",
                    "Debut Year": "TBD",
                },
            ],
        )
        rows = parse_notion_csv(p)
        assert len(rows) == 1
        assert "user_score" not in rows[0]
        assert "anilist_id" not in rows[0]
        assert "deselected_year" not in rows[0]

    def test_parses_studios_and_genres(self, tmp_path: Path):
        p = self._write_csv(
            tmp_path,
            [
                {
                    "English Title": "Frieren",
                    "Studios": "MADHOUSE",
                    "Genres": "Fantasy, Adventure, Drama",
                },
            ],
        )
        rows = parse_notion_csv(p)
        assert rows[0]["studios"] == ["MADHOUSE"]
        assert rows[0]["genres"] == ["Fantasy", "Adventure", "Drama"]


class TestMatchRowsToCollection:
    def test_match_by_anilist_id(self):
        rows = [{"anilist_id": 163263, "english_title": "Frieren", "user_status": "Completed"}]
        collection = [{"anilist_id": 163263, "canonical_title": "Frieren: Beyond Journey's End"}]
        pairs = match_rows_to_collection(rows, collection)
        assert pairs[0][1] is not None
        assert pairs[0][1]["canonical_title"] == "Frieren: Beyond Journey's End"

    def test_match_by_english_title(self):
        rows = [{"english_title": "Frieren", "user_status": "Completed"}]
        collection = [{"canonical_title": "frieren", "anilist_id": 163263}]
        pairs = match_rows_to_collection(rows, collection)
        assert pairs[0][1] is not None

    def test_match_by_romaji_title(self):
        rows = [{"romaji_title": "Sousou no Frieren", "user_status": "Watching"}]
        collection = [{"romaji": "Sousou no Frieren", "anilist_id": 163263}]
        pairs = match_rows_to_collection(rows, collection)
        assert pairs[0][1] is not None

    def test_no_match_returns_none(self):
        rows = [{"english_title": "Unknown Anime", "user_status": "Completed"}]
        collection = [{"canonical_title": "Frieren", "anilist_id": 163263}]
        pairs = match_rows_to_collection(rows, collection)
        assert pairs[0][1] is None

    def test_multiple_rows(self):
        rows = [
            {"english_title": "Frieren", "user_status": "Completed"},
            {"english_title": "Unknown", "user_status": "Watching"},
        ]
        collection = [{"canonical_title": "frieren", "anilist_id": 1}]
        pairs = match_rows_to_collection(rows, collection)
        assert pairs[0][1] is not None
        assert pairs[1][1] is None
