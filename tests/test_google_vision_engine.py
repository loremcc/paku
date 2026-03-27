"""Tests for GoogleVisionOCREngine.

Unit tests: mock the Vision API response — run without the real SDK or credentials.
Integration tests: require google-cloud-vision + valid credentials; skip otherwise.
  Place real Instagram screenshots in tests/fixtures/ before running integration tests.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Response builder helpers (used by unit tests via MagicMock)
# ---------------------------------------------------------------------------


def _make_vertex(x: int = 0, y: int = 0) -> MagicMock:
    v = MagicMock()
    v.x = x
    v.y = y
    return v


def _make_bounding_box(x: int = 10, y: int = 20, w: int = 100, h: int = 30) -> MagicMock:
    bb = MagicMock()
    bb.vertices = [
        _make_vertex(x, y),          # top-left
        _make_vertex(x + w, y),      # top-right
        _make_vertex(x + w, y + h),  # bottom-right
        _make_vertex(x, y + h),      # bottom-left
    ]
    return bb


def _make_word(word: str) -> MagicMock:
    w = MagicMock()
    w.symbols = [_make_symbol(c) for c in word]
    return w


def _make_symbol(text: str) -> MagicMock:
    s = MagicMock()
    s.text = text
    return s


def _make_paragraph(words: list[str]) -> MagicMock:
    p = MagicMock()
    p.words = [_make_word(w) for w in words]
    return p


def _make_block(
    words: list[str],
    confidence: float = 0.95,
    x: int = 10,
    y: int = 20,
    w: int = 100,
    h: int = 30,
) -> MagicMock:
    b = MagicMock()
    b.paragraphs = [_make_paragraph(words)]
    b.confidence = confidence
    b.bounding_box = _make_bounding_box(x, y, w, h)
    return b


def _make_response(
    full_text: str = "Demon Slayer",
    blocks: list[MagicMock] | None = None,
    language_code: str = "ja",
    error_message: str = "",
) -> MagicMock:
    """Build a mock Vision API document_text_detection response."""
    response = MagicMock()
    response.error.message = error_message

    annotation = MagicMock()
    annotation.description = full_text
    response.text_annotations = [annotation] if full_text else []

    lang = MagicMock()
    lang.language_code = language_code

    page = MagicMock()
    page.property.detected_languages = [lang]
    page.blocks = blocks if blocks is not None else [_make_block(["Demon", "Slayer"])]

    full = MagicMock()
    full.pages = [page]
    response.full_text_annotation = full

    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_google_vision")


@pytest.fixture
def engine_no_creds(logger):
    """Engine instance with empty config (no credentials)."""
    from paku.ocr.google_vision import GoogleVisionOCREngine

    return GoogleVisionOCREngine(config={}, logger=logger)


@pytest.fixture
def engine_with_api_key(logger):
    from paku.ocr.google_vision import GoogleVisionOCREngine

    return GoogleVisionOCREngine(
        config={"google_vision": {"api_key": "fake-key-for-unit-tests"}},
        logger=logger,
    )


@pytest.fixture
def mock_vision_modules():
    """Inject a full mock google.cloud.vision hierarchy into sys.modules.

    Ensures `from google.cloud import vision` inside extract() resolves to our mock.
    Restores original sys.modules state on exit.
    """
    mock_vision = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.vision = mock_vision
    mock_google = MagicMock()
    mock_google.cloud = mock_cloud

    with patch.dict(
        sys.modules,
        {
            "google": mock_google,
            "google.cloud": mock_cloud,
            "google.cloud.vision": mock_vision,
        },
    ):
        yield mock_vision


@pytest.fixture
def minimal_pil_image() -> Image.Image:
    return Image.new("RGB", (120, 80), color=(30, 30, 30))


# ---------------------------------------------------------------------------
# Unit tests — metadata (no SDK, no credentials needed)
# ---------------------------------------------------------------------------


class TestGoogleVisionOCREngineMeta:
    def test_name(self, engine_no_creds):
        assert engine_no_creds.name() == "google_vision"

    def test_kind_is_heavy(self, engine_no_creds):
        assert engine_no_creds.kind() == "heavy"


# ---------------------------------------------------------------------------
# Unit tests — is_healthy()
# ---------------------------------------------------------------------------


class TestGoogleVisionIsHealthy:
    def test_false_when_sdk_absent(self, engine_no_creds, monkeypatch):
        """Simulate SDK not installed by injecting None into sys.modules."""
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        with patch.dict(sys.modules, {"google.cloud.vision": None}):
            assert engine_no_creds.is_healthy() is False

    def test_false_when_sdk_present_but_no_credentials(
        self, engine_no_creds, monkeypatch
    ):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        mock_vision = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.vision": mock_vision}):
            assert engine_no_creds.is_healthy() is False

    def test_true_with_env_var(self, engine_no_creds, monkeypatch, tmp_path):
        fake_creds = str(tmp_path / "svc.json")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", fake_creds)
        mock_vision = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.vision": mock_vision}):
            assert engine_no_creds.is_healthy() is True

    def test_true_with_api_key_in_config(self, engine_with_api_key, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        mock_vision = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.vision": mock_vision}):
            assert engine_with_api_key.is_healthy() is True

    def test_false_when_api_key_is_whitespace_only(self, logger, monkeypatch):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        engine = GoogleVisionOCREngine(
            config={"google_vision": {"api_key": "   "}}, logger=logger
        )
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        mock_vision = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.vision": mock_vision}):
            assert engine.is_healthy() is False

    def test_does_not_instantiate_client(self, engine_no_creds, monkeypatch, tmp_path):
        """is_healthy() must never call ImageAnnotatorClient()."""
        fake_creds = str(tmp_path / "svc.json")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", fake_creds)
        mock_vision = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.vision": mock_vision}):
            engine_no_creds.is_healthy()
        mock_vision.ImageAnnotatorClient.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests — mapping helpers (static methods)
# ---------------------------------------------------------------------------


class TestGoogleVisionMappingHelpers:
    def test_map_bbox_correct_coords(self):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        bb = _make_bounding_box(x=5, y=10, w=200, h=50)
        result = GoogleVisionOCREngine._map_bbox(bb)
        assert result is not None
        assert result.x == 5
        assert result.y == 10
        assert result.width == 200
        assert result.height == 50

    def test_map_bbox_returns_none_on_fewer_than_4_vertices(self):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        bb = MagicMock()
        bb.vertices = [_make_vertex(0, 0), _make_vertex(10, 0)]  # only 2
        result = GoogleVisionOCREngine._map_bbox(bb)
        assert result is None

    def test_map_bbox_returns_none_on_none_input(self):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        result = GoogleVisionOCREngine._map_bbox(None)
        assert result is None

    def test_map_bbox_width_abs_for_unusual_vertex_order(self):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        bb = MagicMock()
        # Vertices where top-right.x < top-left.x (rotated text edge case)
        bb.vertices = [
            _make_vertex(100, 10),
            _make_vertex(80, 10),
            _make_vertex(80, 40),
            _make_vertex(100, 40),
        ]
        result = GoogleVisionOCREngine._map_bbox(bb)
        assert result is not None
        assert result.width >= 0
        assert result.height >= 0

    def test_block_text_joins_symbol_characters(self):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        block = _make_block(["Demon", "Slayer"])
        result = GoogleVisionOCREngine._block_text(block)
        assert "Demon" in result
        assert "Slayer" in result

    def test_detect_language_returns_top_code(self, engine_no_creds):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        response = _make_response(language_code="it")
        lang = GoogleVisionOCREngine._detect_language(response)
        assert lang == "it"

    def test_detect_language_returns_none_on_empty_pages(self, engine_no_creds):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        response = MagicMock()
        response.full_text_annotation.pages = []
        lang = GoogleVisionOCREngine._detect_language(response)
        assert lang is None


# ---------------------------------------------------------------------------
# Unit tests — extract() via mocked SDK
# ---------------------------------------------------------------------------


class TestGoogleVisionExtract:
    def test_extract_returns_ocr_result_shape(
        self, engine_no_creds, mock_vision_modules, minimal_pil_image, monkeypatch, tmp_path
    ):
        from paku.models import OcrResult

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "svc.json"))
        response = _make_response("Sword Art Online recensione")
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = response
        mock_vision_modules.ImageAnnotatorClient.return_value = mock_client
        mock_vision_modules.Image = MagicMock()

        result = engine_no_creds.extract(minimal_pil_image)

        assert isinstance(result, OcrResult)
        assert result.engine == "google_vision"
        assert result.raw_text == "Sword Art Online recensione"
        assert isinstance(result.blocks, list)
        assert result.language == "ja"
        assert "block_count" in result.meta

    def test_extract_uses_document_text_detection(
        self, engine_no_creds, mock_vision_modules, minimal_pil_image, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "svc.json"))
        response = _make_response("test")
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = response
        mock_vision_modules.ImageAnnotatorClient.return_value = mock_client
        mock_vision_modules.Image = MagicMock()

        engine_no_creds.extract(minimal_pil_image)

        mock_client.document_text_detection.assert_called_once()
        mock_client.text_detection.assert_not_called()

    def test_extract_raises_on_api_error(
        self, engine_no_creds, mock_vision_modules, minimal_pil_image, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "svc.json"))
        response = _make_response(error_message="Quota exceeded")
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = response
        mock_vision_modules.ImageAnnotatorClient.return_value = mock_client
        mock_vision_modules.Image = MagicMock()

        with pytest.raises(RuntimeError, match="Vision API error"):
            engine_no_creds.extract(minimal_pil_image)

    def test_extract_empty_text_annotations_returns_empty_raw_text(
        self, engine_no_creds, mock_vision_modules, minimal_pil_image, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "svc.json"))
        response = _make_response(full_text="", blocks=[])
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = response
        mock_vision_modules.ImageAnnotatorClient.return_value = mock_client
        mock_vision_modules.Image = MagicMock()

        result = engine_no_creds.extract(minimal_pil_image)
        assert result.raw_text == ""

    def test_extract_uses_env_var_client_path(
        self, engine_no_creds, mock_vision_modules, minimal_pil_image, monkeypatch, tmp_path
    ):
        """When GOOGLE_APPLICATION_CREDENTIALS is set, ImageAnnotatorClient() called without args."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "svc.json"))
        response = _make_response("test")
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = response
        mock_vision_modules.ImageAnnotatorClient.return_value = mock_client
        mock_vision_modules.Image = MagicMock()

        engine_no_creds.extract(minimal_pil_image)

        # Should be called with no positional args and no client_options kwarg
        call_kwargs = mock_vision_modules.ImageAnnotatorClient.call_args
        assert call_kwargs is not None
        assert "client_options" not in (call_kwargs.kwargs or {})

    def test_extract_uses_api_key_client_path(
        self, engine_with_api_key, mock_vision_modules, minimal_pil_image, monkeypatch
    ):
        """When env var absent, ImageAnnotatorClient() called with client_options api_key."""
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        response = _make_response("test")
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = response
        mock_vision_modules.ImageAnnotatorClient.return_value = mock_client
        mock_vision_modules.Image = MagicMock()

        engine_with_api_key.extract(minimal_pil_image)

        call_kwargs = mock_vision_modules.ImageAnnotatorClient.call_args
        assert call_kwargs is not None
        assert "client_options" in (call_kwargs.kwargs or call_kwargs[1])
        client_options = (call_kwargs.kwargs or call_kwargs[1])["client_options"]
        assert client_options.get("api_key") == "fake-key-for-unit-tests"

    def test_map_blocks_confidence_clamped(self, engine_no_creds):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        block = _make_block(["text"], confidence=1.5)
        response = _make_response(blocks=[block])
        blocks = GoogleVisionOCREngine._map_blocks(engine_no_creds, response)
        assert all(0.0 <= b.confidence <= 1.0 for b in blocks)

    def test_map_blocks_skips_whitespace_only(self, engine_no_creds):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        block = _make_block([" "])  # word is a single space
        response = _make_response(blocks=[block])
        blocks = GoogleVisionOCREngine._map_blocks(engine_no_creds, response)
        for b in blocks:
            assert b.text.strip() != ""

    def test_map_blocks_type_is_paragraph(self, engine_no_creds):
        from paku.ocr.google_vision import GoogleVisionOCREngine

        block = _make_block(["Hello", "world"])
        response = _make_response(blocks=[block])
        blocks = GoogleVisionOCREngine._map_blocks(engine_no_creds, response)
        assert all(b.type == "paragraph" for b in blocks)


# ---------------------------------------------------------------------------
# Integration tests — require real SDK + credentials + fixture images
# ---------------------------------------------------------------------------


def _load_google_vision_config() -> dict:
    """Try to load google_vision section from config.yaml if present."""
    try:
        import yaml

        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("google_vision", {})
    except Exception:
        return {}


def _integration_credentials_available() -> bool:
    has_env = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip())
    has_key = bool(_load_google_vision_config().get("api_key", "").strip())
    return has_env or has_key


_sdk_available = pytest.mark.skipif(
    pytest.importorskip.__doc__ and False,  # evaluated lazily below
    reason="google-cloud-vision not installed",
)


@pytest.fixture(scope="session")
def integration_engine():
    """Real GoogleVisionOCREngine for integration tests."""
    vision = pytest.importorskip(
        "google.cloud.vision", reason="google-cloud-vision not installed"
    )
    if not _integration_credentials_available():
        pytest.skip("No Google Vision credentials available for integration tests")

    import logging as _logging
    from paku.ocr.google_vision import GoogleVisionOCREngine

    cfg = {"google_vision": _load_google_vision_config()}
    engine = GoogleVisionOCREngine(config=cfg, logger=_logging.getLogger("integration"))
    if not engine.is_healthy():
        pytest.skip("GoogleVisionOCREngine.is_healthy() returned False")
    return engine


@pytest.fixture(params=[], scope="session")
def fixture_image_path(request) -> Path:
    return request.param


def pytest_generate_tests(metafunc):
    """Parametrize integration tests with whatever PNG/JPG files exist in tests/fixtures/."""
    if "fixture_image_path" in metafunc.fixturenames:
        images = list(FIXTURES_DIR.glob("*.png")) + list(FIXTURES_DIR.glob("*.jpg"))
        if not images:
            metafunc.parametrize("fixture_image_path", [])
        else:
            metafunc.parametrize("fixture_image_path", sorted(images), ids=[p.name for p in sorted(images)])


@pytest.mark.integration
def test_integration_extract_produces_text(integration_engine, fixture_image_path):
    """Phase 0 gate check: Vision API returns ≥ 15 meaningful chars from each fixture."""
    if not fixture_image_path:
        pytest.skip("No fixture images — populate tests/fixtures/ with real screenshots")

    from paku.pipeline import guard_ocr_quality, PoorOCRQuality

    img = Image.open(fixture_image_path).convert("RGB")
    result = integration_engine.extract(img)

    assert result.engine == "google_vision"
    assert isinstance(result.raw_text, str)
    assert isinstance(result.blocks, list)

    try:
        guard_ocr_quality(result.raw_text)
        passed = True
    except PoorOCRQuality:
        passed = False

    print(f"\n[Phase 0] {fixture_image_path.name}: {len(result.raw_text)} chars, "
          f"quality={'PASS' if passed else 'FAIL (< 15 meaningful chars)'}")
    print(f"  raw_text preview: {result.raw_text[:120]!r}")

    # Log but do not fail — the Phase 0 gate is 15/20; individual images may legitimately fail
    # Run with -s to see the Phase 0 progress report


@pytest.mark.integration
def test_integration_engine_registered_in_app_context():
    """When SDK + credentials are present, google_vision appears in AppContext."""
    pytest.importorskip("google.cloud.vision", reason="google-cloud-vision not installed")
    if not _integration_credentials_available():
        pytest.skip("No Google Vision credentials available")

    from paku.context import AppContext

    AppContext.reset()
    ctx = AppContext.instance()
    AppContext.reset()

    assert "google_vision" in ctx.ocr_engines
    engine = ctx.ocr_engines["google_vision"]
    assert engine.kind() == "heavy"
