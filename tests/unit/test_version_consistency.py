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


_VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"


def test_version_matches_version_file():
    """The VERSION file is the third declaration and drifts the most quietly.

    It sat at 1.4.0 while pyproject.toml and __init__.py were both on 1.7.0,
    because the two-way check above cannot see it.
    """
    declared = _VERSION_FILE.read_text(encoding="utf-8").strip()
    assert declared == _pyproject_version(), (
        f"version drift: VERSION={declared!r} "
        f"!= pyproject.toml={_pyproject_version()!r} — bump ALL THREE on release"
    )
