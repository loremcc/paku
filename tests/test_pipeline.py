from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from paku.pipeline import (
    PoorOCRQuality,
    append_review_queue,
    classify_content,
    classify_screen_type,
    discover_images,
    guard_ocr_quality,
    preprocess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(tmp_path: Path, name: str = "img.png", width: int = 100, height: int = 100) -> Path:
    """Create a minimal valid PNG file."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    p = tmp_path / name
    img.save(p, format="PNG")
    return p


def _make_rgb_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height))


def _make_rgba_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGBA", (width, height))


# ---------------------------------------------------------------------------
# discover_images
# ---------------------------------------------------------------------------

class TestDiscoverImages:
    def test_finds_single_image(self, tmp_path):
        p = _make_png(tmp_path, "test.png")
        result = discover_images(tmp_path)
        assert p in result

    def test_single_file_path(self, tmp_path):
        p = _make_png(tmp_path, "single.png")
        result = discover_images(p)
        assert result == [p]

    def test_skips_non_image_file(self, tmp_path):
        _make_png(tmp_path, "img.png")
        (tmp_path / "notes.txt").write_text("hello")
        result = discover_images(tmp_path)
        assert all(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"} for p in result)

    def test_recurses_subdirectories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        p = _make_png(sub, "nested.png")
        result = discover_images(tmp_path)
        assert p in result

    def test_returns_empty_list_for_empty_dir(self, tmp_path):
        result = discover_images(tmp_path)
        assert result == []

    def test_returns_empty_for_non_image_file(self, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        result = discover_images(txt)
        assert result == []

    def test_result_is_sorted_by_name(self, tmp_path):
        names = ["c.png", "a.png", "b.png"]
        for n in names:
            _make_png(tmp_path, n)
        result = discover_images(tmp_path)
        assert [p.name for p in result] == sorted(names)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_converts_rgba_to_rgb(self):
        img = _make_rgba_image()
        result = preprocess(img)
        assert result.mode == "RGB"

    def test_rgb_unchanged_mode(self):
        img = _make_rgb_image()
        result = preprocess(img)
        assert result.mode == "RGB"

    def test_resizes_wide_image(self):
        img = _make_rgb_image(width=3000, height=1000)
        result = preprocess(img)
        assert result.width == 2160
        assert result.height < 1000  # proportionally scaled

    def test_does_not_resize_narrow_image(self):
        img = _make_rgb_image(width=1080, height=1920)
        result = preprocess(img)
        assert result.width == 1080


# ---------------------------------------------------------------------------
# guard_ocr_quality
# ---------------------------------------------------------------------------

class TestGuardOcrQuality:
    def test_passes_on_sufficient_text(self):
        guard_ocr_quality("This is enough text to pass the quality check.")

    def test_raises_on_empty_string(self):
        with pytest.raises(PoorOCRQuality):
            guard_ocr_quality("")

    def test_raises_on_short_text(self):
        with pytest.raises(PoorOCRQuality):
            guard_ocr_quality("abc")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(PoorOCRQuality):
            guard_ocr_quality("   \n\t  ")

    def test_boundary_exactly_15_chars(self):
        # 15 alpha chars — must pass
        guard_ocr_quality("abcdefghijklmno")

    def test_boundary_14_chars_fails(self):
        with pytest.raises(PoorOCRQuality):
            guard_ocr_quality("abcdefghijklmn")


# ---------------------------------------------------------------------------
# classify_screen_type
# ---------------------------------------------------------------------------

class TestClassifyScreenType:
    def test_story_from_send_message(self):
        assert classify_screen_type("Send message... story chrome") == "story"

    def test_story_from_see_translation(self):
        assert classify_screen_type("Caption text\nSee translation > more") == "story"

    def test_feed_card_with_github_heading(self):
        text = "\nAwesome Repository Title Here\nGitHub"
        assert classify_screen_type(text) == "feed_card"

    def test_default_post(self):
        assert classify_screen_type("Some random caption without signals") == "post"


# ---------------------------------------------------------------------------
# classify_content
# ---------------------------------------------------------------------------

class TestClassifyContent:
    def test_explicit_mode_overrides_auto(self):
        assert classify_content("some anime text", "post", mode="url") == "url"
        assert classify_content("some text", "post", mode="recipe") == "recipe"
        assert classify_content("some text", "post", mode="anime") == "anime"

    def test_detects_github_url(self):
        text = "Check out github.com/user/repo for the source code."
        assert classify_content(text, "post") == "url"

    def test_detects_ingredient_anchor(self):
        text = "Ingredients:\n200g flour\n100ml milk"
        assert classify_content(text, "post") == "recipe"

    def test_detects_italian_signal(self):
        text = "Ho guardato un nuovo anime ieri sera!"
        assert classify_content(text, "post") == "anime"

    def test_story_defaults_to_anime(self):
        assert classify_content("Nessun segnale riconosciuto", "story") == "anime"

    def test_unknown_for_unrecognized_post(self):
        assert classify_content("Random text without signals", "post") == "unknown"


# ---------------------------------------------------------------------------
# append_review_queue
# ---------------------------------------------------------------------------

class TestAppendReviewQueue:
    def test_creates_new_queue_file(self, tmp_path):
        queue = str(tmp_path / "output" / "review_queue.json")
        entry = {"screenshot": "img.png", "reason": "test", "confidence": "low"}
        append_review_queue(entry, queue)
        with open(queue) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["reason"] == "test"

    def test_appends_to_existing_queue(self, tmp_path):
        queue = str(tmp_path / "review_queue.json")
        e1 = {"id": 1}
        e2 = {"id": 2}
        append_review_queue(e1, queue)
        append_review_queue(e2, queue)
        with open(queue) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[1]["id"] == 2

    def test_no_tmp_file_left_behind(self, tmp_path):
        queue = str(tmp_path / "review_queue.json")
        append_review_queue({"x": 1}, queue)
        tmp_file = Path(queue + ".tmp")
        assert not tmp_file.exists()

    def test_idempotent_on_multiple_writes(self, tmp_path):
        queue = str(tmp_path / "review_queue.json")
        for i in range(5):
            append_review_queue({"i": i}, queue)
        with open(queue) as f:
            data = json.load(f)
        assert len(data) == 5
