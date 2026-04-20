from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from .config import load_config, validate_config
from .logging_utils import get_logger
from .ocr.base import OCREngine
from .ocr.router import EngineRouter
from .ocr.stub import StubOCREngine


@dataclass
class AppContext:
    """Singleton that holds config, logger, and OCR engine registry."""

    _instance: ClassVar[AppContext | None] = None

    config: dict
    logger: logging.Logger
    ocr_engines: dict[str, OCREngine]
    router: EngineRouter

    @classmethod
    def instance(cls) -> "AppContext":
        if cls._instance is None:
            cls._instance = cls._build()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        cls._instance = None

    @classmethod
    def _build(cls) -> "AppContext":
        config = load_config()
        validate_config(config)

        log_level = config.get("paku", {}).get("log_level", "INFO")
        logger = get_logger(log_level)

        engines: dict[str, OCREngine] = {}

        # Stub is always available — required for testing and dev
        stub = StubOCREngine(config=config, logger=logger)
        engines[stub.name()] = stub

        # Google Cloud Vision — optional, registered only when SDK is installed and credentials present
        try:
            from .ocr.google_vision import GoogleVisionOCREngine

            gv = GoogleVisionOCREngine(config=config, logger=logger)
            if gv.is_healthy():
                engines[gv.name()] = gv
                logger.debug("[AppContext] Registered engine: google_vision")
            else:
                logger.debug(
                    "[AppContext] google_vision not healthy — skipping registration "
                    "(install google-cloud-vision and set credentials)"
                )
        except ImportError:
            logger.debug(
                "[AppContext] google-cloud-vision not installed — google_vision engine unavailable"
            )

        # Ollama VLM — registered when ollama config section is present.
        # Health check is lazy (deferred to first is_healthy() call at runtime).
        if config.get("ollama"):
            from .ocr.ollama import OllamaVLMEngine

            ollama_engine = OllamaVLMEngine(config=config, logger=logger)
            engines[ollama_engine.name()] = ollama_engine
            logger.debug("[AppContext] Registered engine: ollama_vlm (health check deferred)")

        router = EngineRouter(engines=engines)
        logger.debug(f"[AppContext] Registered engines: {', '.join(engines)}")

        return cls(config=config, logger=logger, ocr_engines=engines, router=router)

    def get_ocr(self, name: str) -> OCREngine:
        if name not in self.ocr_engines:
            raise ValueError(
                f"OCR engine not registered: {name!r}. "
                f"Available: {', '.join(self.ocr_engines)}"
            )
        return self.ocr_engines[name]

    def list_ocr_engines(self) -> dict[str, OCREngine]:
        return dict(self.ocr_engines)

    def resolve_engine(self, name_or_strategy: str) -> OCREngine:
        """Accept a concrete engine name or a routing strategy (light/heavy/auto)."""
        key = name_or_strategy.lower()
        if key in self.ocr_engines:
            return self.ocr_engines[key]
        if key in {"light", "heavy", "auto"}:
            return self.router.select(key)
        raise ValueError(
            f"Unknown engine or strategy: {name_or_strategy!r}. "
            f"Engines: {', '.join(self.ocr_engines)}; strategies: light, heavy, auto"
        )
