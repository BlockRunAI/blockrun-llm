"""The package version must be declared identically in both places.

Releases bump the version in two files — ``pyproject.toml`` (what PyPI ships)
and ``blockrun_llm/__init__.py`` (what ``blockrun_llm.__version__`` reports).
These drifted once (1.4.6 bumped pyproject but not __init__, so installed
copies under-reported as 1.4.5). This test fails CI if they ever diverge
again, instead of the mismatch shipping silently to PyPI.

Parsed with a regex rather than tomllib so it runs on Python 3.9 (no stdlib
TOML parser before 3.11) without adding a tomli dependency.
"""

import re
from pathlib import Path

import blockrun_llm

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    # First top-level `version = "..."` under [project] / [tool.poetry] etc.
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    assert match, "no version declared in pyproject.toml"
    return match.group(1)


def test_version_matches_pyproject():
    assert blockrun_llm.__version__ == _pyproject_version(), (
        f"version drift: __init__.py={blockrun_llm.__version__!r} "
        f"!= pyproject.toml={_pyproject_version()!r} — bump BOTH on release"
    )
