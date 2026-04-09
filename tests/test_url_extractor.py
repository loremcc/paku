"""Tests for URL extractor (v0.2) — 4-tier cascade, noise stripping, output writers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from paku.extractors.url import (
    GITHUB_CONTEXT_SIGNALS,
    SOCIAL_DOMAIN_BLOCKLIST,
    TLD_ALLOWLIST,
    _count_github_signals,
    _extract_project_name,
    _has_chrome_adjacency,
    _is_social_domain,
    _tier1,
    _tier2,
    _tier3,
    _tier4,
    extract,
    strip_noise,
)
from paku.models import ExtractionResult, URLExtractionResult
from paku.outputs.json_out import write_json
from paku.outputs.txt_out import write_txt

_LOGGER = logging.getLogger("test")


# ─── Model tests ───


class TestModels:
    def test_extraction_result_base(self):
        r = ExtractionResult(
            extractor="url",
            confidence=0.5,
            source_screenshot="IMG.PNG",
            extracted_at="2026-04-01T00:00:00",
        )
        assert r.extractor == "url"
        assert r.needs_review is False

    def test_url_extraction_result_defaults(self):
        r = URLExtractionResult(
            confidence=0.9,
            source_screenshot="IMG.PNG",
            extracted_at="2026-04-01T00:00:00",
            extraction_tier=1,
        )
        assert r.extractor == "url"
        assert r.extraction_mode == "direct"
        assert r.resolved_url is None
        assert r.raw_text_snippet == ""

    def test_url_extraction_result_serialization(self):
        r = URLExtractionResult(
            confidence=0.9,
            source_screenshot="IMG.PNG",
            extracted_at="2026-04-01T00:00:00",
            extraction_tier=1,
            resolved_url="https://github.com/user/repo",
        )
        d = r.model_dump()
        assert d["extractor"] == "url"
        assert d["extraction_tier"] == 1
        assert d["resolved_url"] == "https://github.com/user/repo"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            URLExtractionResult(
                confidence=1.5,
                source_screenshot="X",
                extracted_at="X",
                extraction_tier=1,
            )


# ─── Noise stripping tests ───


class TestNoiseStripping:
    def test_strips_reel_header(self):
        text = "For you    Friends\nhttps://example.com\nSome content"
        cleaned = strip_noise(text)
        assert "For you" not in cleaned
        assert "https://example.com" in cleaned

    def test_strips_pure_numeric(self):
        text = "Some text\n1,234\n56.7K\nhttps://example.com"
        cleaned = strip_noise(text)
        assert "1,234" not in cleaned
        assert "56.7K" not in cleaned.split("\n")  # cleaned as individual line

    def test_strips_engagement_lines(self):
        text = "Liked by user1 and others\nReal content here"
        cleaned = strip_noise(text)
        assert "Liked by" not in cleaned

    def test_strips_action_prompts(self):
        text = "Content\nAdd comment...\nSee translation >\nMore content"
        cleaned = strip_noise(text)
        assert "Add comment" not in cleaned
        assert "See translation" not in cleaned

    def test_strips_hashtag_heavy_lines(self):
        text = "URL here\n#github #opensource #dev #tools\nMore text"
        cleaned = strip_noise(text)
        assert "#github" not in cleaned

    def test_strips_slide_indicators(self):
        text = "Content\n1/7\n3/5\nMore content"
        cleaned = strip_noise(text)
        assert "1/7" not in cleaned

    def test_strips_bottom_nav(self):
        text = "Content\nHome\nInbox\nExplore\nProfile"
        cleaned = strip_noise(text)
        assert "Home\n" not in cleaned
        assert "Inbox" not in cleaned

    def test_strips_notification_lines(self):
        text = "WhatsApp message from John now\nSecond line\nhttps://github.com/user/repo"
        cleaned = strip_noise(text)
        assert "WhatsApp" not in cleaned

    def test_preserves_content_lines(self):
        text = "Meet Pretext!\nhttps://github.com/chenglou/pretext\nA great tool"
        cleaned = strip_noise(text)
        assert "Meet Pretext!" in cleaned
        assert "https://github.com/chenglou/pretext" in cleaned

    def test_preserves_empty_lines(self):
        text = "Line one\n\nLine two"
        cleaned = strip_noise(text)
        assert cleaned == text


# ─── Tier 1 tests ───


class TestTier1:
    def test_full_https_url(self):
        text = "Check this out\nhttps://github.com/chenglou/pretext\n#opensource"
        result = _tier1(text)
        assert result is not None
        assert result.resolved_url == "https://github.com/chenglou/pretext"
        assert result.confidence == 0.9
        assert result.needs_review is False
        assert result.extraction_tier == 1

    def test_domain_pattern_without_scheme(self):
        text = "Found at github.com/user/cool-repo today"
        result = _tier1(text)
        assert result is not None
        assert result.resolved_url == "https://github.com/user/cool-repo"

    def test_git_clone_suffix_stripped(self):
        text = "Run:\ngit clone https://github.com/user/repo.git\nto install"
        result = _tier1(text)
        assert result is not None
        assert result.resolved_url == "https://github.com/user/repo"

    def test_social_domain_filtered(self):
        text = "https://www.instagram.com/p/ABC123\nhttps://github.com/user/repo"
        result = _tier1(text)
        assert result is not None
        assert "instagram.com" not in result.resolved_url
        assert "github.com/user/repo" in result.resolved_url

    def test_social_domain_only_returns_none(self):
        text = "https://www.instagram.com/p/ABC123"
        result = _tier1(text)
        assert result is None

    def test_truncated_url_falls_through(self):
        text = "github.com/agucova/aweso..."
        result = _tier1(text)
        # Should return None — truncated + final segment < 6 chars.
        assert result is None

    def test_truncated_url_long_segment_kept(self):
        text = "https://github.com/user/longname..."
        result = _tier1(text)
        # "longname" is >= 6 chars so Tier 1 keeps it despite truncation.
        assert result is not None
        assert "longname" in result.resolved_url

    def test_short_segment_no_ellipsis_github_context_falls_through(self):
        # Regression: IMG_4965. Browser bar shows github.com/agucova/aweso (no ellipsis).
        # "aweso" = 5 chars < 6 AND 2+ GitHub signals present → must fall through to Tier 3.
        text = (
            "github.com/agucova/aweso\n"
            "agucova / awesome-esp\n"
            "Issues 5\n"
            "Pull requests 3\n"
        )
        result = _tier1(text)
        assert result is None, (
            "Tier 1 must not return a short-segment URL when 2+ GitHub signals are present — "
            "fall through to Tier 3 for full reconstruction."
        )

    def test_short_segment_no_ellipsis_without_github_signals_kept(self):
        # Short segment without GitHub context signals → NOT treated as truncated.
        text = "github.com/user/rs"
        result = _tier1(text)
        assert result is not None
        assert "github.com/user/rs" in result.resolved_url

    def test_short_segment_full_https_url_kept_despite_github_signals(self):
        # A full https:// URL with a short repo name is NOT browser-bar truncation —
        # browser bars never show the scheme. Short path is legitimate here.
        text = "https://github.com/user/repo\nStars 50\nForks 10"
        result = _tier1(text)
        assert result is not None
        assert result.resolved_url == "https://github.com/user/repo"

    def test_trailing_punctuation_stripped(self):
        text = 'See https://example.com/page). More text'
        result = _tier1(text)
        assert result is not None
        assert result.resolved_url == "https://example.com/page"

    def test_snippet_populated(self):
        text = "prefix text " * 5 + "https://github.com/user/repo" + " suffix text" * 5
        result = _tier1(text)
        assert result is not None
        assert len(result.raw_text_snippet) > 0
        assert "github.com" in result.raw_text_snippet


# ─── Tier 2 tests ───


class TestTier2:
    def test_domain_with_allowed_tld(self):
        text = "Check out openml.org for datasets"
        result = _tier2(text)
        assert result is not None
        assert result.resolved_url == "https://openml.org"
        assert result.confidence == 0.7
        assert result.needs_review is True
        assert result.extraction_tier == 2

    def test_domain_with_path(self):
        text = "Visit algorithm-visualizer.org/demo for more"
        result = _tier2(text)
        assert result is not None
        assert "algorithm-visualizer.org" in result.resolved_url

    def test_file_extension_rejected(self):
        """File extensions like .py, .json, .png should NOT match Tier 2."""
        text = "Load model.py and data.json and image.png for training"
        result = _tier2(text)
        # .py, .json, .png are not in TLD_ALLOWLIST
        assert result is None

    def test_abbreviation_rejected(self):
        """Common abbreviations like e.g should not match."""
        text = "Use tools e.g for automation"
        result = _tier2(text)
        assert result is None

    def test_social_domain_rejected(self):
        text = "Follow us on instagram.com for updates"
        result = _tier2(text)
        assert result is None

    def test_chrome_adjacency_rejects_username(self):
        """Instagram usernames with dots near timestamps should be rejected."""
        text = "smartworld.it\n3h\nSuggested for you"
        result = _tier2(text)
        assert result is None

    def test_github_context_suppression(self):
        """When 2+ GitHub signals present, Tier 2 skips entirely."""
        text = "kubecm.cloud\nStars 1.2k\nForks 234\nIssues 12"
        result = _tier2(text)
        assert result is None

    def test_no_github_suppression_without_signals(self):
        """Domain match should work when no GitHub signals present."""
        text = "Visit kubecm.cloud for more info\nGreat documentation"
        result = _tier2(text)
        assert result is not None
        assert "kubecm.cloud" in result.resolved_url


# ─── Tier 3 tests ───


class TestTier3:
    def test_author_repo_with_signals(self):
        text = "sunny0826 / kubecm\nStars 1.2k\nForks 234\nMIT license"
        result = _tier3(text)
        assert result is not None
        assert result.resolved_url == "https://github.com/sunny0826/kubecm"
        assert result.confidence == 0.75
        assert result.needs_review is True
        assert result.extraction_tier == 3

    def test_rejects_pure_numeric(self):
        """Slide indicators like 1/7 should not match."""
        text = "1 / 7\nStars 100\nForks 50"
        result = _tier3(text)
        assert result is None

    def test_rejects_domain_fragment_as_author(self):
        """github.com/user/repo should not parse as author=github.com, repo=user."""
        text = "github.com/agucova/aweso\nagucova / awesome-esp\nStars 50\nForks 10"
        result = _tier3(text)
        assert result is not None
        # Must reconstruct from the clean author/repo, not the domain fragment.
        assert result.resolved_url == "https://github.com/agucova/awesome-esp"

    def test_rejects_without_signals(self):
        """author/repo without GitHub signals should not match."""
        text = "someuser / somerepo\nJust a random page"
        result = _tier3(text)
        assert result is None

    def test_context_window_signals(self):
        """Signals must be within 500-char window around the match."""
        text = "A" * 300 + "\nuser / repo\nStars 500\nForks 100" + "B" * 300
        result = _tier3(text)
        assert result is not None
        assert result.resolved_url == "https://github.com/user/repo"


# ─── Tier 4 tests ───


class TestTier4:
    def test_always_returns_result(self):
        text = "Some random text with no URLs at all"
        result = _tier4(text)
        assert result is not None
        assert result.confidence == 0.0
        assert result.needs_review is True
        assert result.resolved_url is None
        assert result.extraction_tier == 4

    def test_extracts_project_name_near_signal(self):
        text = "Check out this open source tool\nCool Project Name here"
        name = _extract_project_name(text)
        assert name is not None
        assert "Cool Project" in name

    def test_extracts_allcaps_word(self):
        text = "No signal phrases but PRETEXT is mentioned"
        name = _extract_project_name(text)
        assert name == "PRETEXT"

    def test_excludes_nav_allcaps(self):
        text = "HOME INBOX EXPLORE PROFILE some text"
        name = _extract_project_name(text)
        assert name is None or name not in {"HOME", "INBOX", "EXPLORE", "PROFILE"}

    def test_snippet_is_last_200_chars(self):
        text = "A" * 300 + "important tail"
        result = _tier4(text)
        assert len(result.raw_text_snippet) <= 200
        assert "important tail" in result.raw_text_snippet


# ─── Helper tests ───


class TestHelpers:
    def test_is_social_domain(self):
        assert _is_social_domain("https://www.instagram.com/p/123") is True
        assert _is_social_domain("https://twitter.com/user") is True
        assert _is_social_domain("https://github.com/user/repo") is False

    def test_count_github_signals(self):
        text = "Stars 100\nForks 50\nIssues 12"
        assert _count_github_signals(text) == 3

    def test_count_github_signals_zero(self):
        assert _count_github_signals("No signals here") == 0

    def test_has_chrome_adjacency(self):
        lines = ["smartworld.it", "3h", "Follow"]
        assert _has_chrome_adjacency(lines, 0) is True

    def test_no_chrome_adjacency(self):
        lines = ["openml.org", "Great tool for ML", "Try it now"]
        assert _has_chrome_adjacency(lines, 0) is False


# ─── Integration: extract() ───


class TestExtractIntegration:
    def test_tier1_wins_over_lower(self):
        text = "https://github.com/user/repo\nsome-domain.io\nauthor / repo\nStars 50\nForks 10"
        result = extract(text, "IMG.PNG", {}, _LOGGER)
        assert result.extraction_tier == 1
        assert result.resolved_url == "https://github.com/user/repo"
        assert result.source_screenshot == "IMG.PNG"
        assert result.extracted_at != ""

    def test_tier2_when_no_full_url(self):
        text = "Check out openml.org for datasets\nGreat resource"
        result = extract(text, "IMG.PNG", {}, _LOGGER)
        assert result.extraction_tier == 2
        assert "openml.org" in result.resolved_url

    def test_tier3_when_github_signals_suppress_tier2(self):
        text = "kubecm.cloud\nsunny0826 / kubecm\nStars 1.2k\nForks 234\nMIT license"
        result = extract(text, "IMG.PNG", {}, _LOGGER)
        assert result.extraction_tier == 3
        assert result.resolved_url == "https://github.com/sunny0826/kubecm"

    def test_tier4_fallback(self):
        text = "Random text with no URLs or domains"
        result = extract(text, "IMG.PNG", {}, _LOGGER)
        assert result.extraction_tier == 4
        assert result.resolved_url is None

    def test_noise_stripped_before_extraction(self):
        text = "For you    Friends\n1,234\nhttps://github.com/user/repo\nHome\nInbox"
        result = extract(text, "IMG.PNG", {}, _LOGGER)
        assert result.extraction_tier == 1
        assert result.resolved_url == "https://github.com/user/repo"

    def test_tier1_truncated_no_ellipsis_falls_to_tier3(self):
        # Regression: IMG_4965. Browser bar truncates without visible ellipsis.
        # Tier 1 should skip the partial URL; Tier 3 reconstructs from author/repo.
        text = (
            "github.com/agucova/aweso\n"
            "agucova / awesome-esp\n"
            "Issues 5\n"
            "Pull requests 3\n"
        )
        result = extract(text, "IMG_4965.PNG", {}, _LOGGER)
        assert result.extraction_tier == 3, (
            f"Expected Tier 3 reconstruction, got Tier {result.extraction_tier}"
        )
        assert result.resolved_url == "https://github.com/agucova/awesome-esp"
        assert result.needs_review is True


# ─── Output writer tests ───


class TestJsonOutput:
    def test_write_json_creates_file(self, tmp_path: Path):
        data = {"extractor": "url", "resolved_url": "https://example.com"}
        path = write_json(data, "IMG_001", tmp_path)
        assert path.exists()
        assert path.name == "IMG_001.json"
        content = json.loads(path.read_text())
        assert content["resolved_url"] == "https://example.com"

    def test_write_json_creates_directory(self, tmp_path: Path):
        out_dir = tmp_path / "nested" / "dir"
        path = write_json({"key": "val"}, "test", out_dir)
        assert path.exists()

    def test_write_json_pretty_printed(self, tmp_path: Path):
        path = write_json({"k": "v"}, "test", tmp_path)
        text = path.read_text()
        assert "\n" in text  # pretty-printed, not single line


class TestTxtOutput:
    def test_write_txt_with_url(self, tmp_path: Path):
        path = write_txt("https://github.com/user/repo", "IMG_001", tmp_path)
        assert path.exists()
        assert path.read_text().strip() == "https://github.com/user/repo"

    def test_write_txt_without_url(self, tmp_path: Path):
        path = write_txt(None, "IMG_001", tmp_path)
        assert "[no URL resolved" in path.read_text()

    def test_write_txt_creates_directory(self, tmp_path: Path):
        out_dir = tmp_path / "nested"
        path = write_txt("https://example.com", "test", out_dir)
        assert path.exists()


# ─── Pipeline integration tests ───


class TestPipelineIntegration:
    """Test URL extractor wiring in process_image()."""

    def test_url_content_triggers_extraction(self, tmp_path: Path):
        """When content_type is 'url', extraction result should be present."""
        from unittest.mock import MagicMock

        from paku.context import AppContext
        from paku.models import OcrResult
        from paku.ocr.stub import StubOCREngine

        ocr_text = "Meet Pretext!\nhttps://github.com/chenglou/pretext\n#opensource"
        stub = StubOCREngine(config={}, logger=logging.getLogger("test"))
        mock_result = OcrResult(engine="stub", raw_text=ocr_text)

        # Patch the stub to return our specific OCR text.
        stub.extract = MagicMock(return_value=mock_result)

        config = {"outputs": {"review_queue": str(tmp_path / "review.json")}}
        from paku.ocr.router import EngineRouter

        router = EngineRouter({"stub": stub})
        AppContext._instance = AppContext(
            config=config,
            logger=logging.getLogger("test"),
            ocr_engines={"stub": stub},
            router=router,
        )

        # Create a tiny test image.
        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        from paku.pipeline import process_image

        result = process_image(image_path=img_path, mode="url")
        assert result is not None
        assert result["status"] == "extracted"
        assert "extraction" in result
        assert result["extraction"]["extraction_tier"] == 1
        assert "github.com/chenglou/pretext" in result["extraction"]["resolved_url"]

    def test_anime_mode_produces_extracted_status(self, tmp_path: Path):
        """Anime mode extracts a result (status=extracted) and routes to review when low confidence."""
        from unittest.mock import MagicMock, patch

        from paku.context import AppContext
        from paku.models import OcrResult
        from paku.ocr.router import EngineRouter
        from paku.ocr.stub import StubOCREngine

        stub = StubOCREngine(config={}, logger=logging.getLogger("test"))
        mock_result = OcrResult(
            engine="stub",
            raw_text="episodio di anime guardato oggi rilasciato ieri stagione nuova",
        )
        stub.extract = MagicMock(return_value=mock_result)

        config = {"outputs": {"review_queue": str(tmp_path / "review.json")}}
        router = EngineRouter({"stub": stub})
        AppContext._instance = AppContext(
            config=config,
            logger=logging.getLogger("test"),
            ocr_engines={"stub": stub},
            router=router,
        )

        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        from paku.pipeline import process_image

        import requests as req_mod
        with patch("requests.post", side_effect=req_mod.exceptions.ConnectionError("network down")):
            result = process_image(image_path=img_path, mode="anime")
        assert result is not None
        assert result["status"] == "extracted"
        assert "extraction" in result

    def test_needs_review_writes_to_queue(self, tmp_path: Path):
        """Tier 2/3/4 results with needs_review should write to review_queue."""
        from unittest.mock import MagicMock

        from paku.context import AppContext
        from paku.models import OcrResult
        from paku.ocr.router import EngineRouter
        from paku.ocr.stub import StubOCREngine

        # Text that only matches Tier 2 (domain-only, no full URL, no GitHub signals).
        ocr_text = "Visit openml.org for great datasets\nMachine learning resources"
        stub = StubOCREngine(config={}, logger=logging.getLogger("test"))
        mock_result = OcrResult(engine="stub", raw_text=ocr_text)
        stub.extract = MagicMock(return_value=mock_result)

        queue_path = tmp_path / "review.json"
        config = {"outputs": {"review_queue": str(queue_path)}}
        router = EngineRouter({"stub": stub})
        AppContext._instance = AppContext(
            config=config,
            logger=logging.getLogger("test"),
            ocr_engines={"stub": stub},
            router=router,
        )

        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        from paku.pipeline import process_image

        result = process_image(image_path=img_path, mode="url")
        assert result is not None
        assert result["extraction"]["needs_review"] is True

        # review_queue.json should have been written.
        assert queue_path.exists()
        queue = json.loads(queue_path.read_text())
        assert len(queue) == 1
        assert queue[0]["extractor"] == "url"
        assert "domain-only" in queue[0]["reason"]
