from __future__ import annotations

import pytest

from paku.context import AppContext


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.integration (live API calls, fixtures).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="integration test — pass --run-integration to enable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def reset_app_context():
    """Reset AppContext singleton before every test to ensure isolation."""
    AppContext.reset()
    yield
    AppContext.reset()


@pytest.fixture(autouse=True)
def _reset_anilist_circuit():
    """Clear the AniList 403 circuit breaker before every test for isolation."""
    from paku.extractors.anime import reset_anilist_circuit

    reset_anilist_circuit()
    yield
    reset_anilist_circuit()
