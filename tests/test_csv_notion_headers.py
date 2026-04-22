from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from paku.models import AnimeExtractionResult
from paku.outputs.csv_out import ANIME_CSV_HEADERS, write_anime_csv

NOW = "2026-04-21T00:00:00+00:00"


def _make_result(
    raw_title: str = "Test Anime",
    canonical_title: str | None = "Test Anime",
    confidence: float = 0.9,
    dedup_key: str | None = "123",
    media_format: str | None = "TV",
    source: str | None = "MANGA",
    debut_year: int | None = 2020,
    country_of_origin: str | None = "JP",
    studios: list[str] | None = None,
    anilist_id: int | None = 123,
) -> AnimeExtractionResult:
    return AnimeExtractionResult(
        extractor="anime",
        confidence=confidence,
        needs_review=False,
        source_screenshot="test.png",
        extracted_at=NOW,
        raw_title=raw_title,
        canonical_title=canonical_title,
        romaji="Test Anime",
        media_format=media_format,
        source=source,
        debut_year=debut_year,
        country_of_origin=country_of_origin,
        studios=studios if studios is not None else ["Madhouse"],
        dedup_key=dedup_key,
        anilist_id=anilist_id,
    )


class TestAnimeNotionHeaders:
    def test_exact_header_values(self):
        expected = [
            "English Title",
            "Romaji Title",
            "Cover",
            "Format",
            "Source",
            "Debut Year",
            "Status",
            "Country",
            "Studios",
        ]
        assert ANIME_CSV_HEADERS == expected

    def test_headers_are_strings(self):
        assert all(isinstance(h, str) for h in ANIME_CSV_HEADERS)

    def test_no_duplicate_headers(self):
        assert len(ANIME_CSV_HEADERS) == len(set(ANIME_CSV_HEADERS))


class TestWriteAnimeNotionCsv:
    def test_writes_file(self, tmp_path):
        result = _make_result()
        out = write_anime_csv([result], tmp_path / "out.csv")
        assert out.exists()

    def test_csv_headers_match_constants(self, tmp_path):
        result = _make_result()
        out = write_anime_csv([result], tmp_path / "out.csv")
        reader = csv.DictReader(out.read_text(encoding="utf-8").splitlines())
        assert reader.fieldnames == ANIME_CSV_HEADERS

    def test_english_title_from_canonical(self, tmp_path):
        result = _make_result(canonical_title="Canonical Anime", raw_title="Raw Anime")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["English Title"] == "Canonical Anime"

    def test_english_title_falls_back_to_raw(self, tmp_path):
        result = _make_result(canonical_title=None, raw_title="Raw Anime")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["English Title"] == "Raw Anime"

    def test_format_mapping_tv(self, tmp_path):
        result = _make_result(media_format="TV")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Format"] == "TV"

    def test_format_mapping_movie(self, tmp_path):
        result = _make_result(media_format="MOVIE")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Format"] == "Movie"

    def test_format_mapping_ova(self, tmp_path):
        result = _make_result(media_format="OVA")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Format"] == "OVA"

    def test_format_mapping_ona(self, tmp_path):
        result = _make_result(media_format="ONA")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Format"] == "ONA"

    def test_format_mapping_special(self, tmp_path):
        result = _make_result(media_format="SPECIAL")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Format"] == "Special"

    def test_format_mapping_tv_short(self, tmp_path):
        result = _make_result(media_format="TV_SHORT")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Format"] == "TV Short"

    def test_source_mapping_manga(self, tmp_path):
        result = _make_result(source="MANGA")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Source"] == "Manga"

    def test_source_mapping_light_novel(self, tmp_path):
        result = _make_result(source="LIGHT_NOVEL")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Source"] == "Light Novel"

    def test_source_mapping_original(self, tmp_path):
        result = _make_result(source="ORIGINAL")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Source"] == "Original"

    def test_status_always_not_started(self, tmp_path):
        result = _make_result()
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Status"] == "Not Started"

    def test_debut_year_populated(self, tmp_path):
        result = _make_result(debut_year=2021)
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Debut Year"] == "2021"

    def test_debut_year_empty_when_none(self, tmp_path):
        result = _make_result(debut_year=None)
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Debut Year"] == ""

    def test_country_from_model(self, tmp_path):
        result = _make_result(country_of_origin="KR")
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Country"] == "KR"

    def test_country_empty_when_none(self, tmp_path):
        result = _make_result(country_of_origin=None)
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Country"] == ""

    def test_studios_joined(self, tmp_path):
        result = _make_result(studios=["Madhouse", "Studio Pierrot"])
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Studios"] == "Madhouse, Studio Pierrot"

    def test_studios_empty(self, tmp_path):
        result = _make_result(studios=[])
        out = write_anime_csv([result], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["Studios"] == ""

    def test_dedup_same_key_keeps_higher_confidence(self, tmp_path):
        r1 = _make_result(raw_title="Anime A", dedup_key="key1", confidence=0.7)
        r2 = _make_result(raw_title="Anime A Better", dedup_key="key1", confidence=0.95, canonical_title="Anime A Better")
        out = write_anime_csv([r1, r2], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["English Title"] == "Anime A Better"

    def test_dedup_different_keys_both_written(self, tmp_path):
        r1 = _make_result(raw_title="Anime A", dedup_key="key1", canonical_title="Anime A")
        r2 = _make_result(raw_title="Anime B", dedup_key="key2", canonical_title="Anime B", anilist_id=456)
        out = write_anime_csv([r1, r2], tmp_path / "out.csv")
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2

    def test_empty_list_writes_header_only(self, tmp_path):
        out = write_anime_csv([], tmp_path / "out.csv")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1  # header only
        assert lines[0].startswith("English Title")

    def test_atomic_write_no_tmp_file_left(self, tmp_path):
        result = _make_result()
        out = write_anime_csv([result], tmp_path / "out.csv")
        tmp = tmp_path / "out.csv.tmp"
        assert not tmp.exists()
        assert out.exists()
