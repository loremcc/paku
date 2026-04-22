"""Tests for OllamaVLMEngine — stream parsing, health check, pipeline integration."""

from __future__ import annotations

import json
import logging
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from paku.ocr.ollama import OllamaVLMEngine


def _make_engine(config: dict | None = None) -> OllamaVLMEngine:
    if config is None:
        config = {"ollama": {"base_url": "http://localhost:11434", "model": "test-model"}}
    return OllamaVLMEngine(config=config, logger=logging.getLogger("test"))


def _make_ndjson_response(chunks: list[dict]) -> MagicMock:
    """Create a mock response that yields NDJSON lines via iter_lines."""
    lines = [json.dumps(c) for c in chunks]
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status = MagicMock()
    return resp


def _test_image() -> Image.Image:
    return Image.new("RGB", (800, 600), color="white")


class TestExtractStreaming:
    """Stream parsing: concatenation, done termination, error handling."""

    @patch("requests.post")
    def test_extract_streams_and_concatenates(self, mock_post):
        chunks = [
            {"response": "Attack on ", "done": False},
            {"response": "Titan\n", "done": False},
            {"response": "One Piece", "done": False},
            {"done": True},
        ]
        mock_post.return_value = _make_ndjson_response(chunks)

        engine = _make_engine()
        result = engine.extract(_test_image())

        assert result.raw_text == "Attack on Titan\nOne Piece"
        assert result.engine == "ollama_vlm"
        assert result.meta["model"] == "test-model"
        assert result.meta["streamed_chunks"] == 3

    @patch("requests.post")
    def test_extract_done_true_terminates(self, mock_post):
        """Chunks after done:true are NOT included."""
        chunks = [
            {"response": "Naruto", "done": False},
            {"done": True},
            {"response": "SHOULD NOT APPEAR", "done": False},
        ]
        mock_post.return_value = _make_ndjson_response(chunks)

        engine = _make_engine()
        result = engine.extract(_test_image())

        assert result.raw_text == "Naruto"
        assert "SHOULD NOT APPEAR" not in result.raw_text
        assert result.meta["streamed_chunks"] == 1

    @patch("requests.post")
    def test_extract_connection_error(self, mock_post):
        """ConnectionError returns empty OcrResult, never raises."""
        import requests

        mock_post.side_effect = requests.ConnectionError("refused")

        engine = _make_engine()
        result = engine.extract(_test_image())

        assert result.raw_text == ""
        assert result.engine == "ollama_vlm"
        assert "error" in result.meta


class TestHealthCheck:
    """Lazy health check with caching."""

    @patch("requests.get")
    def test_is_healthy_false_on_timeout(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("timed out")

        engine = _make_engine()
        assert engine.is_healthy() is False
        # Cached — second call doesn't retry
        assert engine.is_healthy() is False
        assert mock_get.call_count == 1

    @patch("requests.get")
    def test_is_healthy_true_on_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)

        engine = _make_engine()
        assert engine.is_healthy() is True
        # Cached
        assert engine.is_healthy() is True
        assert mock_get.call_count == 1


class TestPipelineSmartRerun:
    """Pipeline integration: smart re-run gating."""

    @patch("requests.post")
    @patch("requests.get")
    def test_pipeline_smart_reruns_on_low_confidence(self, mock_get, mock_post):
        """Low-confidence fast-path triggers smart re-run with Ollama."""
        from paku.context import AppContext

        AppContext.reset()

        # Mock Ollama health check
        mock_get.return_value = MagicMock(status_code=200)

        # Mock Ollama generate — returns a better title
        smart_chunks = [
            {"response": "Frieren: Beyond Journey's End", "done": False},
            {"done": True},
        ]
        mock_post.return_value = _make_ndjson_response(smart_chunks)

        # Config with ollama section
        config = {
            "paku": {"log_level": "DEBUG"},
            "outputs": {"base_dir": "./output", "review_queue": "./output/review_queue.json"},
            "ollama": {"base_url": "http://localhost:11434", "model": "test-model"},
        }

        with patch("paku.context.load_config", return_value=config), \
             patch("paku.context.validate_config"), \
             patch("paku.pipeline.load_image") as mock_load, \
             patch("paku.pipeline.append_review_queue"), \
             patch("paku.pipeline.guard_ocr_quality"), \
             patch("paku.extractors.anime.extract") as mock_anime_extract:

            # Fast-path returns low confidence
            from paku.models import AnimeExtractionResult

            fast_result = AnimeExtractionResult(
                extractor="anime",
                confidence=0.3,
                needs_review=True,
                source_screenshot="test.png",
                extracted_at="2026-04-20T00:00:00",
                raw_title="Frieren",
                extraction_mode="fast",
            )
            # Smart re-run returns higher confidence
            smart_result = AnimeExtractionResult(
                extractor="anime",
                confidence=0.85,
                needs_review=False,
                source_screenshot="test.png",
                extracted_at="2026-04-20T00:00:00",
                raw_title="Frieren: Beyond Journey's End",
                canonical_title="Frieren: Beyond Journey's End",
                extraction_mode="fast",  # will be overwritten to "smart"
            )
            mock_anime_extract.side_effect = [fast_result, smart_result]
            mock_load.return_value = Image.new("RGB", (400, 300))

            from paku.pipeline import process_image

            result = process_image("test.png", mode="anime", smart=True)

            assert result is not None
            assert result["extraction"]["extraction_mode"] == "smart"
            assert result["extraction"]["confidence"] == 0.85
            # anime_extract called twice: fast + smart
            assert mock_anime_extract.call_count == 2

        AppContext.reset()

    @patch("requests.get")
    def test_pipeline_smart_skipped_on_high_confidence(self, mock_get):
        """High-confidence fast-path does NOT trigger smart re-run."""
        from paku.context import AppContext

        AppContext.reset()

        config = {
            "paku": {"log_level": "DEBUG"},
            "outputs": {"base_dir": "./output", "review_queue": "./output/review_queue.json"},
            "ollama": {"base_url": "http://localhost:11434", "model": "test-model"},
        }

        with patch("paku.context.load_config", return_value=config), \
             patch("paku.context.validate_config"), \
             patch("paku.pipeline.load_image") as mock_load, \
             patch("paku.pipeline.append_review_queue"), \
             patch("paku.pipeline.guard_ocr_quality"), \
             patch("paku.extractors.anime.extract") as mock_anime_extract, \
             patch("paku.ocr.ollama.requests.post") as mock_post:

            from paku.models import AnimeExtractionResult

            fast_result = AnimeExtractionResult(
                extractor="anime",
                confidence=0.85,
                needs_review=False,
                source_screenshot="test.png",
                extracted_at="2026-04-20T00:00:00",
                raw_title="One Piece",
                canonical_title="One Piece",
                extraction_mode="fast",
            )
            mock_anime_extract.return_value = fast_result
            mock_load.return_value = Image.new("RGB", (400, 300))

            from paku.pipeline import process_image

            result = process_image("test.png", mode="anime", smart=True)

            assert result is not None
            assert result["extraction"]["extraction_mode"] == "fast"
            # Ollama engine never called
            mock_post.assert_not_called()
            # anime_extract called only once
            assert mock_anime_extract.call_count == 1

        AppContext.reset()
