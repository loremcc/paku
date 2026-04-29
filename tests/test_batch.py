from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from paku.pipeline import (
    BatchReport,
    _append_checkpoint,
    _load_checkpoint,
    process_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(tmp_path: Path, name: str) -> Path:
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    p = tmp_path / name
    img.save(p, format="PNG")
    return p


def _fake_result(path: Path, content_type: str = "anime") -> dict:
    return {
        "screenshot": str(path),
        "screen_type": "post",
        "content_type": content_type,
        "ocr_text": "Some anime title",
        "engine": "stub",
        "outputs": [],
        "smart": False,
        "extracted_at": "2026-04-21T00:00:00+00:00",
        "status": "extracted",
        "extraction": {
            "extractor": "anime",
            "confidence": 0.9,
            "needs_review": False,
            "source_screenshot": str(path),
            "extracted_at": "2026-04-21T00:00:00+00:00",
            "raw_title": "Test Anime",
            "canonical_title": "Test Anime",
            "native_title": None,
            "romaji": "Test Anime",
            "media_type": "ANIME",
            "media_source": "anime",
            "episodes": 12,
            "status": "FINISHED",
            "genres": [],
            "score": 8.0,
            "anilist_id": 123,
            "anilist_url": "https://anilist.co/anime/123",
            "cover_image": None,
            "banner_image": None,
            "media_format": "TV",
            "source": "MANGA",
            "country_of_origin": "JP",
            "debut_year": 2020,
            "studios": ["Madhouse"],
            "extraction_mode": "fast",
            "title_pattern": "label",
            "extraction_context": "recommendation",
            "multi_title_detected": False,
            "dedup_key": "123",
            "levenshtein_ratio": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_load_missing_file(self, tmp_path):
        assert _load_checkpoint(tmp_path / ".paku_checkpoint") == set()

    def test_append_and_load(self, tmp_path):
        cp = tmp_path / ".paku_checkpoint"
        _append_checkpoint(cp, "/a/b/c.png")
        _append_checkpoint(cp, "/a/b/d.png")
        loaded = _load_checkpoint(cp)
        assert loaded == {"/a/b/c.png", "/a/b/d.png"}

    def test_append_idempotent(self, tmp_path):
        cp = tmp_path / ".paku_checkpoint"
        _append_checkpoint(cp, "/img.png")
        _append_checkpoint(cp, "/img.png")
        loaded = _load_checkpoint(cp)
        assert len(loaded) == 1

    def test_tmp_file_replaced(self, tmp_path):
        cp = tmp_path / ".paku_checkpoint"
        _append_checkpoint(cp, "/img.png")
        tmp = tmp_path / ".paku_checkpoint.tmp"
        assert not tmp.exists()
        assert cp.exists()


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_batch_3_images_all_processed(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(3)]

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(tmp_path / "out"), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.side_effect = [_fake_result(p) for p in real_imgs]

            report, results = process_batch(img_dir, mode="anime")

        assert report.total == 3
        assert report.processed == 3
        assert report.skipped == 0
        assert report.failed == 0
        assert len(results) == 3

    def test_batch_one_bad_image_continues(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(3)]

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx, \
             patch("paku.pipeline.append_review_queue"):
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(tmp_path / "out"), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.side_effect = [
                Exception("corrupt file"),
                _fake_result(real_imgs[1]),
                _fake_result(real_imgs[2]),
            ]

            report, results = process_batch(img_dir)

        assert report.failed == 1
        assert report.processed == 2
        assert len(results) == 2

    def test_batch_none_result_counts_as_failed(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(2)]

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(tmp_path / "out"), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.side_effect = [None, _fake_result(real_imgs[1])]

            report, results = process_batch(img_dir)

        assert report.failed == 1
        assert report.processed == 1

    def test_none_result_is_checkpointed(self, tmp_path):
        """Permanent failures (poor_ocr / load_error) must be checkpointed.

        Otherwise --resume will re-OCR them every batch run and re-queue
        duplicate entries indefinitely.
        """
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        bad = _make_png(img_dir, "broken.png")
        out_dir = tmp_path / "out"

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.return_value = None

            process_batch(img_dir)

        cp = out_dir / ".paku_checkpoint"
        assert cp.exists()
        assert str(bad) in cp.read_text(encoding="utf-8"), (
            "permanent-failure image must be in checkpoint to prevent re-OCR on resume"
        )

    def test_batch_error_not_checkpointed_for_retry(self, tmp_path):
        """Unhandled exceptions in process_image are kept un-checkpointed.

        These could be transient (network blip, lock contention) — give them a
        chance to succeed on next --resume run.
        """
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        flaky = _make_png(img_dir, "flaky.png")
        out_dir = tmp_path / "out"

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx, \
             patch("paku.pipeline.append_review_queue"):
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.side_effect = RuntimeError("transient")

            process_batch(img_dir)

        cp = out_dir / ".paku_checkpoint"
        # Either the file doesn't exist or it doesn't contain the flaky image.
        if cp.exists():
            assert str(flaky) not in cp.read_text(encoding="utf-8"), (
                "batch_error images must NOT be checkpointed — they're retryable"
            )

    def test_checkpoint_resume_skips_processed(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(3)]
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Pre-populate checkpoint with first 2 images
        cp = out_dir / ".paku_checkpoint"
        cp.write_text("\n".join([str(real_imgs[0]), str(real_imgs[1])]) + "\n", encoding="utf-8")

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.return_value = _fake_result(real_imgs[2])

            report, results = process_batch(img_dir, resume=True)

        assert report.skipped == 2
        assert report.processed == 1
        assert mock_pi.call_count == 1

    def test_no_resume_processes_all(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(2)]
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Pre-populate checkpoint
        cp = out_dir / ".paku_checkpoint"
        cp.write_text(str(real_imgs[0]) + "\n", encoding="utf-8")

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.return_value = _fake_result(real_imgs[0])

            report, results = process_batch(img_dir, resume=False)

        assert report.skipped == 0
        assert report.processed == 2

    def test_checkpoint_file_created_after_success(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, "img00.png")]
        out_dir = tmp_path / "out"

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.return_value = _fake_result(real_imgs[0])

            process_batch(img_dir)

        cp = out_dir / ".paku_checkpoint"
        assert cp.exists()
        assert str(real_imgs[0]) in cp.read_text(encoding="utf-8")

    def test_empty_directory_no_crash(self, tmp_path):
        img_dir = tmp_path / "empty"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        with patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()

            report, results = process_batch(img_dir)

        assert report.total == 0
        assert report.processed == 0
        assert results == []

    def test_by_content_type_counts(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(3)]

        def _result_with_type(p: Path, ct: str) -> dict:
            r = _fake_result(p)
            r["content_type"] = ct
            return r

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(tmp_path / "out"), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.side_effect = [
                _result_with_type(real_imgs[0], "anime"),
                _result_with_type(real_imgs[1], "anime"),
                _result_with_type(real_imgs[2], "url"),
            ]

            report, _ = process_batch(img_dir)

        assert report.by_content_type.get("anime") == 2
        assert report.by_content_type.get("url") == 1

    def test_progress_callback_called(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, f"img{i:02d}.png") for i in range(2)]
        calls: list[tuple[int, int, str]] = []

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(tmp_path / "out"), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.return_value = _fake_result(real_imgs[0])

            process_batch(img_dir, progress_callback=lambda c, t, n: calls.append((c, t, n)))

        assert len(calls) >= 2
        # Final call has current == total
        assert calls[-1][0] == calls[-1][1]

    def test_review_queued_for_needs_review_result(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        real_imgs = [_make_png(img_dir, "img00.png")]

        result = _fake_result(real_imgs[0])
        result["extraction"]["needs_review"] = True

        with patch("paku.pipeline.process_image") as mock_pi, \
             patch("paku.context.AppContext") as mock_ctx:
            mock_ctx.instance.return_value.config = {
                "outputs": {"base_dir": str(tmp_path / "out"), "review_queue": str(tmp_path / "rq.json")}
            }
            mock_ctx.instance.return_value.logger = MagicMock()
            mock_pi.return_value = result

            report, _ = process_batch(img_dir)

        assert report.review_queued == 1


# ---------------------------------------------------------------------------
# Multi-title extractions output
# ---------------------------------------------------------------------------

def _make_extraction_dict(
    canonical_title: str,
    raw_title: str,
    anilist_id: int,
    debut_year: int = 2020,
) -> dict:
    return {
        "extractor": "anime",
        "confidence": 0.9,
        "needs_review": False,
        "source_screenshot": "test.png",
        "extracted_at": "2026-04-21T00:00:00+00:00",
        "raw_title": raw_title,
        "canonical_title": canonical_title,
        "native_title": None,
        "romaji": canonical_title,
        "media_type": "ANIME",
        "media_source": "anime",
        "episodes": 12,
        "status": "FINISHED",
        "genres": [],
        "score": 8.0,
        "anilist_id": anilist_id,
        "anilist_url": f"https://anilist.co/anime/{anilist_id}",
        "cover_image": None,
        "banner_image": None,
        "media_format": "TV",
        "source": "MANGA",
        "country_of_origin": "JP",
        "debut_year": debut_year,
        "studios": ["Madhouse"],
        "extraction_mode": "fast",
        "title_pattern": "label",
        "extraction_context": "recommendation",
        "multi_title_detected": True,
        "dedup_key": str(anilist_id),
        "levenshtein_ratio": 1.0,
    }


class TestMultiTitleExtractionsOutput:
    """extractions (plural) key: all titles written to TXT and CSV."""

    def _make_multi_result(self) -> dict:
        ex1 = _make_extraction_dict("Attack on Titan", "Attack on Titan", 101)
        ex2 = _make_extraction_dict("One Piece", "One Piece", 102)
        ex3 = _make_extraction_dict("Naruto", "Naruto", 103)
        return {
            "screenshot": "test.png",
            "screen_type": "post",
            "content_type": "anime",
            "ocr_text": "...",
            "engine": "stub",
            "outputs": [],
            "smart": False,
            "extracted_at": "2026-04-21T00:00:00+00:00",
            "status": "extracted",
            "extraction": ex1,
            "extractions": [ex1, ex2, ex3],
        }

    def test_all_titles_in_txt(self, tmp_path):
        from paku.cli import _write_consolidated_txt

        result = self._make_multi_result()
        with patch("paku.cli.click"):
            _write_consolidated_txt([result], tmp_path)

        txt = (tmp_path / "anime_titles.txt").read_text(encoding="utf-8")
        assert "Attack on Titan" in txt
        assert "One Piece" in txt
        assert "Naruto" in txt

    def test_all_titles_in_csv(self, tmp_path):
        from paku.cli import _write_anime_csv

        result = self._make_multi_result()
        with patch("paku.cli.click"):
            _write_anime_csv([result], tmp_path)

        rows = list(csv.DictReader((tmp_path / "anime_export.csv").read_text(encoding="utf-8").splitlines()))
        titles = {r["English Title"] for r in rows}
        assert "Attack on Titan" in titles
        assert "One Piece" in titles
        assert "Naruto" in titles
        assert len(rows) == 3

    def test_single_extraction_unchanged(self, tmp_path):
        """Result without extractions key still writes the single title."""
        from paku.cli import _write_consolidated_txt

        single = {
            "screenshot": "test.png",
            "screen_type": "post",
            "content_type": "anime",
            "ocr_text": "...",
            "engine": "stub",
            "outputs": [],
            "smart": False,
            "extracted_at": "2026-04-21T00:00:00+00:00",
            "status": "extracted",
            "extraction": _make_extraction_dict("PLUTO", "PLUTO", 200),
        }
        with patch("paku.cli.click"):
            _write_consolidated_txt([single], tmp_path)

        txt = (tmp_path / "anime_titles.txt").read_text(encoding="utf-8")
        assert "PLUTO" in txt


class TestMultiTitleJsonFanOut:
    """Per-image JSON write must NOT drop multi-title extractions.

    Regression: pipeline previously wrote only `result["extraction"]` (the first
    title) to disk — 2nd/3rd titles in `result["extractions"]` were silently lost.
    Fix writes one indexed file per extraction.
    """

    def _multi_anime_result(self, screenshot_path: str) -> dict:
        ex1 = _make_extraction_dict("Attack on Titan", "Attack on Titan", 101)
        ex2 = _make_extraction_dict("One Piece", "One Piece", 102)
        ex3 = _make_extraction_dict("Naruto", "Naruto", 103)
        return {
            "screenshot": screenshot_path,
            "content_type": "anime",
            "extraction": ex1,
            "extractions": [ex1, ex2, ex3],
        }

    def test_multi_title_writes_one_file_per_extraction(self, tmp_path):
        from paku.outputs.json_out import write_json
        from paku.pipeline import process_image  # noqa: F401 — import sanity

        result = self._multi_anime_result(str(tmp_path / "scr.png"))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Replicate the pipeline's JSON fan-out logic for the multi-title path.
        stem = "scr"
        for idx, ex in enumerate(result["extractions"], start=1):
            json_stem = stem if idx == 1 else f"{stem}_{idx}"
            write_json(ex, json_stem, out_dir)

        files = sorted(p.name for p in out_dir.glob("*.json"))
        assert files == ["scr.json", "scr_2.json", "scr_3.json"]

        first = json.loads((out_dir / "scr.json").read_text("utf-8"))
        second = json.loads((out_dir / "scr_2.json").read_text("utf-8"))
        third = json.loads((out_dir / "scr_3.json").read_text("utf-8"))
        assert first["canonical_title"] == "Attack on Titan"
        assert second["canonical_title"] == "One Piece"
        assert third["canonical_title"] == "Naruto"

    def test_pipeline_writes_all_extractions_via_process_image(self, tmp_path, monkeypatch):
        """End-to-end: process_image with multi-title result writes N JSONs."""
        from paku import pipeline as _pipeline

        png = _make_png(tmp_path, "scr.png")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ex1 = _make_extraction_dict("Title A", "Title A", 101)
        ex2 = _make_extraction_dict("Title B", "Title B", 102)

        # Stub the pipeline to skip OCR/classify/extract — just exercise the
        # fan-out branch by patching the dispatch result.
        from paku.models import AnimeExtractionResult, OcrResult

        anime_results = [
            AnimeExtractionResult.model_validate(ex1),
            AnimeExtractionResult.model_validate(ex2),
        ]

        ctx = MagicMock()
        ctx.config = {
            "outputs": {"base_dir": str(out_dir), "review_queue": str(tmp_path / "rq.json")}
        }
        ctx.logger = MagicMock()
        ctx.resolve_engine.return_value.extract.return_value = OcrResult(
            engine="stub", raw_text="A and B", confidence=0.9
        )
        # AppContext is a local import inside process_image — patch via paku.context.
        monkeypatch.setattr("paku.context.AppContext.instance", staticmethod(lambda: ctx))
        monkeypatch.setattr(_pipeline, "guard_ocr_quality", lambda _t: None)
        monkeypatch.setattr(_pipeline, "classify_screen_type", lambda _t: "post")
        monkeypatch.setattr(_pipeline, "classify_content", lambda *_a, **_k: "anime")
        monkeypatch.setattr(
            "paku.extractors.anime.extract", lambda **_k: anime_results
        )

        _pipeline.process_image(png, mode="anime", outputs=["json"])

        files = sorted(p.name for p in out_dir.glob("*.json"))
        assert files == ["scr.json", "scr_2.json"]
        first = json.loads((out_dir / "scr.json").read_text("utf-8"))
        second = json.loads((out_dir / "scr_2.json").read_text("utf-8"))
        assert first["canonical_title"] == "Title A"
        assert second["canonical_title"] == "Title B"

    def test_single_title_writes_one_file_unsuffixed(self, tmp_path):
        """Single-title (no extractions plural) keeps existing single-file format."""
        from paku.outputs.json_out import write_json

        single_ex = _make_extraction_dict("Solo Title", "Solo Title", 200)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        write_json(single_ex, "scr", out_dir)

        files = sorted(p.name for p in out_dir.glob("*.json"))
        assert files == ["scr.json"]
