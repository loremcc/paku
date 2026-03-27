from __future__ import annotations

import pytest

from paku.context import AppContext


@pytest.fixture(autouse=True)
def reset_app_context():
    """Reset AppContext singleton before every test to ensure isolation."""
    AppContext.reset()
    yield
    AppContext.reset()
