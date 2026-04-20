from __future__ import annotations

from dataclasses import dataclass

from .base import OCREngine


@dataclass
class EngineRouter:
    """
    Routing logic for OCR engines.

    Strategies:
    - 'light'  → prefer light engines, fall back to heavy if none available
    - 'heavy'  → prefer heavy engines, fall back to light if none available
    - 'auto'   → prefer heavy if healthy, fall back to light
    - 'smart'  → ollama_vlm if healthy, else fall back to heavy/light
    """

    engines: dict[str, OCREngine]

    def _select_by_kind(self, kind: str) -> OCREngine | None:
        candidates = [
            e for e in self.engines.values() if e.kind() == kind and e.is_healthy()
        ]
        return candidates[0] if candidates else None

    def select(self, strategy: str) -> OCREngine:
        strategy = strategy.lower()

        if strategy == "light":
            engine = self._select_by_kind("light") or self._select_by_kind("heavy")
            if engine:
                return engine
            raise RuntimeError("No OCR engines available for strategy 'light'.")

        if strategy in {"heavy", "auto"}:
            engine = self._select_by_kind("heavy") or self._select_by_kind("light")
            if engine:
                return engine
            raise RuntimeError(f"No OCR engines available for strategy {strategy!r}.")

        if strategy == "smart":
            # Prefer ollama_vlm engine; fall back to heavy then light (same as auto)
            ollama = self.engines.get("ollama_vlm")
            if ollama and ollama.is_healthy():
                return ollama
            engine = self._select_by_kind("heavy") or self._select_by_kind("light")
            if engine:
                return engine
            raise RuntimeError("No OCR engines available for strategy 'smart'.")

        raise ValueError(f"Unknown routing strategy: {strategy!r}")
