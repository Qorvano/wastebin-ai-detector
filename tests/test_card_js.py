"""Static checks on the calibration card.

There is no browser in this suite, and the card is the one part of the
integration a Python test cannot exercise - which is exactly why a
whole block of it (the pointer handlers) could be deleted by a careless
edit and still ship: every Python test stayed green while marking a lid
and drawing a region were both dead in the field.

These checks are cheap and catch that class of mistake: a method that
is called but no longer exists, and an element the code reaches for
that the template never renders.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CARD = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "wastebin_ai_detector"
    / "www"
    / "wastebin-calibration-card.js"
)

# The interactions the card exists for. Losing any of these means the
# user can no longer calibrate, which no other test would notice.
REQUIRED_METHODS = {
    "_onDown",
    "_onMove",
    "_onUp",
    "_pointerPos",
    "_vertexAt",
    "_patches",
    "_paintPoints",
    "_paintRegion",
    "_paintMarks",
    "_renderMarkRow",
    "_renderLabelRow",
    "_renderRunRow",
    "_saveSample",
    "_saveLabels",
    "_startRun",
    "_stopRun",
    "_capture",
    "_applyRegion",
    "_undoVertex",
    "_clearRegion",
    "_setMode",
    "_persist",
}


@pytest.fixture(scope="module")
def source() -> str:
    return CARD.read_text(encoding="utf-8")


def _defined(source: str) -> set[str]:
    return set(re.findall(r"^  (?:async )?(_\w+|\w+)\(", source, re.M))


def test_every_required_method_exists(source):
    missing = sorted(REQUIRED_METHODS - _defined(source))
    assert not missing, f"card methods lost: {missing}"


def test_no_method_is_defined_twice(source):
    defined = re.findall(r"^  (?:async )?(_?\w+)\(", source, re.M)
    duplicates = sorted({n for n in defined if defined.count(n) > 1})
    assert not duplicates, f"defined more than once: {duplicates}"


def test_every_called_method_is_defined(source):
    called = set(re.findall(r"this\.(_\w+)\(", source))
    missing = sorted(called - _defined(source))
    assert not missing, f"called but not defined: {missing}"


def test_every_element_the_code_uses_is_rendered(source):
    used = set(re.findall(r'getElementById\("([\w-]+)"\)', source))
    rendered = set(re.findall(r'id="([\w-]+)"', source))
    missing = sorted(used - rendered)
    assert not missing, f"element ids never rendered: {missing}"


def test_every_text_key_used_is_defined_in_both_languages(source):
    used = set(re.findall(r"this\._t\.(\w+)", source))
    blocks = re.findall(r"^  (en|de): \{(.*?)^  \},", source, re.M | re.S)
    assert {lang for lang, _body in blocks} == {"en", "de"}
    for lang, body in blocks:
        defined = set(re.findall(r"^    (\w+):", body, re.M))
        missing = sorted(used - defined)
        assert not missing, f"{lang} is missing text keys: {missing}"
