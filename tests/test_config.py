from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from paku.config import DEFAULT_CONFIG, load_config, validate_config


class TestLoadConfig:
    def test_returns_defaults_when_file_missing(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config["paku"]["log_level"] == "INFO"
        assert config["anilist"]["base_url"] == "https://graphql.anilist.co"

    def test_deep_merge_overrides_leaf_values(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"paku": {"log_level": "DEBUG"}}))
        config = load_config(cfg_file)
        assert config["paku"]["log_level"] == "DEBUG"
        # Sibling keys preserved after merge
        assert "anilist" in config

    def test_deep_merge_preserves_unset_keys(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"notion": {"token": "tok_123"}}))
        config = load_config(cfg_file)
        # Other notion keys still present from defaults
        assert "anime_db_id" in config["notion"]

    def test_empty_yaml_file_returns_defaults(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        config = load_config(cfg_file)
        assert config == DEFAULT_CONFIG

    def test_returns_deep_copy_of_defaults(self, tmp_path):
        c1 = load_config(tmp_path / "missing.yaml")
        c2 = load_config(tmp_path / "missing.yaml")
        c1["paku"]["log_level"] = "MUTATED"
        assert c2["paku"]["log_level"] == "INFO"

    def test_raises_on_invalid_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(": invalid: yaml: [unclosed")
        with pytest.raises(Exception):
            load_config(cfg_file)


class TestValidateConfig:
    def test_passes_on_empty_notion_token(self):
        config = load_config()  # defaults — no token
        validate_config(config)  # must not raise

    def test_passes_when_notion_fully_configured(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump(
                {
                    "notion": {
                        "token": "secret_abc",
                        "anime_db_id": "db1",
                        "url_db_id": "db2",
                        "recipe_db_id": "db3",
                    }
                }
            )
        )
        config = load_config(cfg_file)
        validate_config(config)  # must not raise

    def test_raises_when_token_set_but_db_id_missing(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"notion": {"token": "secret_abc"}}))
        config = load_config(cfg_file)
        with pytest.raises(ValueError, match="anime_db_id"):
            validate_config(config)

    def test_raises_for_each_missing_db_id(self, tmp_path):
        for missing_key in ("anime_db_id", "url_db_id", "recipe_db_id"):
            cfg_file = tmp_path / "config.yaml"
            notion = {
                "token": "tok",
                "anime_db_id": "db1",
                "url_db_id": "db2",
                "recipe_db_id": "db3",
            }
            del notion[missing_key]
            cfg_file.write_text(yaml.dump({"notion": notion}))
            config = load_config(cfg_file)
            with pytest.raises(ValueError, match=missing_key):
                validate_config(config)

    def test_passes_when_langextract_not_configured(self):
        config = load_config()
        validate_config(config)  # api_key is empty — no error

    def test_raises_when_langextract_api_key_set_but_model_missing(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump({"langextract": {"api_key": "key123", "model_id": ""}})
        )
        config = load_config(cfg_file)
        with pytest.raises(ValueError, match="model_id"):
            validate_config(config)
