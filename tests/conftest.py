"""Core-suite bootstrap: import paths only, no Home Assistant.

The HA-layer tests live in ``tests_ha/`` with their own conftest and
pytest configuration; this suite runs with the pytest-homeassistant
plugin explicitly disabled (see pyproject addopts).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The core must stay importable WITHOUT Home Assistant: register a
# synthetic parent package that only carries the search path, so
# `wastebin_ai_detector.core` resolves without executing the real
# integration __init__.py (which imports homeassistant).
_pkg = types.ModuleType("wastebin_ai_detector")
_pkg.__path__ = [str(_REPO_ROOT / "custom_components" / "wastebin_ai_detector")]
sys.modules.setdefault("wastebin_ai_detector", _pkg)
