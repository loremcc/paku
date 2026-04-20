from __future__ import annotations

from pathlib import Path

import pytest

from paku.context import AppContext
from paku.models import OcrResult
from paku.ocr.base import OCREngine
from paku.ocr.router import EngineRouter
from paku.ocr.stub import StubOCREngine


class TestStubOCREngine:
    def test_name(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        assert engine.name() == "stub"

    def test_kind_default(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        assert engine.kind() == "light"

    def test_is_healthy(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        assert engine.is_healthy() is True

    def test_extract_returns_ocr_result(self, tmp_path):
        fake_img = tmp_path / "test.png"
        fake_img.write_bytes(b"fake")
        engine = StubOCREngine(config={}, logger=_make_logger())
        result = engine.extract(fake_img)
        assert isinstance(result, OcrResult)
        assert result.engine == "stub"
        assert fake_img.name in result.raw_text
        assert len(result.raw_text) > 0

    def test_extract_meta_contains_path(self, tmp_path):
        fake_img = tmp_path / "sample.png"
        fake_img.write_bytes(b"fake")
        engine = StubOCREngine(config={}, logger=_make_logger())
        result = engine.extract(fake_img)
        assert "path" in result.meta
        assert str(fake_img) == result.meta["path"]


class TestEngineRouter:
    def test_select_auto_returns_light_engine_when_only_light_available(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        router = EngineRouter(engines={engine.name(): engine})
        selected = router.select("auto")
        assert selected is engine

    def test_select_light(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        router = EngineRouter(engines={engine.name(): engine})
        selected = router.select("light")
        assert selected is engine

    def test_select_heavy_falls_back_to_light(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        router = EngineRouter(engines={engine.name(): engine})
        # No heavy engine registered — should fall back to light (stub)
        selected = router.select("heavy")
        assert selected is engine

    def test_select_raises_on_empty_engines(self):
        router = EngineRouter(engines={})
        with pytest.raises(RuntimeError, match="No OCR engines available"):
            router.select("auto")

    def test_select_raises_on_unknown_strategy(self):
        engine = StubOCREngine(config={}, logger=_make_logger())
        router = EngineRouter(engines={engine.name(): engine})
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            router.select("turbo")


class TestAppContextEngineRegistry:
    def test_stub_engine_always_registered(self):
        ctx = AppContext.instance()
        assert "stub" in ctx.ocr_engines

    def test_get_ocr_returns_stub(self):
        ctx = AppContext.instance()
        engine = ctx.get_ocr("stub")
        assert isinstance(engine, StubOCREngine)

    def test_get_ocr_raises_on_unknown(self):
        ctx = AppContext.instance()
        with pytest.raises(ValueError, match="not registered"):
            ctx.get_ocr("nonexistent_engine")

    def test_resolve_engine_by_name(self):
        ctx = AppContext.instance()
        engine = ctx.resolve_engine("stub")
        assert engine.name() == "stub"

    def test_resolve_engine_by_strategy(self):
        ctx = AppContext.instance()
        engine = ctx.resolve_engine("auto")
        assert isinstance(engine, OCREngine)

    def test_resolve_engine_raises_on_unknown(self):
        ctx = AppContext.instance()
        with pytest.raises(ValueError, match="Unknown engine or strategy"):
            ctx.resolve_engine("nonexistent")

    def test_list_ocr_engines_returns_copy(self):
        ctx = AppContext.instance()
        engines = ctx.list_ocr_engines()
        assert isinstance(engines, dict)
        # Modifying the returned dict must not affect the registry
        engines["injected"] = None  # type: ignore[assignment]
        assert "injected" not in ctx.ocr_engines


def _make_logger():
    import logging

    return logging.getLogger("test")
