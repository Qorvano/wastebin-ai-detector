"""HA-suite bootstrap: repo paths plus custom-integration discovery.

Run this suite with its own configuration (the plugin's asyncio setup
conflicts with the plain core suite):

    pytest -c tests_ha/pytest.ini tests_ha
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))  # shared scene helpers


@pytest.fixture(autouse=True)
def _auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Let the HA test loader discover custom_components/."""
    return
