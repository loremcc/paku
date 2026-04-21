from __future__ import annotations

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
