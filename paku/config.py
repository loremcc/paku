from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG: dict = {
    "paku": {
        "log_level": "INFO",
    },
    "anilist": {
        "base_url": "https://graphql.anilist.co",
        "confidence_threshold": 0.6,
    },
    "notion": {
        "token": "",
        "anime_db_id": "",
        "url_db_id": "",
        "recipe_db_id": "",
    },
    "outputs": {
        "base_dir": "./output",
        "review_queue": "./output/review_queue.json",
        "checkpoint": "./output/.paku_checkpoint",
    },
    "langextract": {
        "api_key": "",
        "model_id": "gemini-2.5-flash",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml, merging with DEFAULT_CONFIG for any missing keys.

    Falls back to DEFAULT_CONFIG entirely if config.yaml does not exist.
    Never raises on a missing file — only raises on invalid YAML syntax.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return copy.deepcopy(DEFAULT_CONFIG)

    return _deep_merge(DEFAULT_CONFIG, user_config)


def validate_config(config: dict) -> None:
    """Raise ValueError with a clear message on invalid configuration.

    Only validates fields that are set — empty strings mean 'not configured'.
    """
    notion = config.get("notion", {})
    token = notion.get("token", "")
    if token:
        for db_key in ("anime_db_id", "url_db_id", "recipe_db_id"):
            if not notion.get(db_key, ""):
                raise ValueError(
                    f"notion.{db_key} is required when notion.token is set. "
                    f"Add it to config.yaml."
                )

    langextract = config.get("langextract", {})
    if langextract.get("api_key", ""):
        if not langextract.get("model_id", ""):
            raise ValueError(
                "langextract.model_id is required when langextract.api_key is set."
            )
