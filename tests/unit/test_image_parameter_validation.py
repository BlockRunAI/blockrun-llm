"""
Unit tests for ImageClient parameter validation.

Ensures that unsupported parameters are caught early with helpful error messages
instead of confusing TypeErrors from the Python runtime.
"""

from __future__ import annotations

import pytest

from blockrun_llm import ImageClient

from ..helpers import TEST_PRIVATE_KEY


def _make_client() -> ImageClient:
    """Create a client with test private key."""
    return ImageClient(private_key=TEST_PRIVATE_KEY)


def test_generate_rejects_quality_parameter():
    """Unsupported quality parameter should raise TypeError with helpful message."""
    client = _make_client()

    with pytest.raises(TypeError) as excinfo:
        client.generate("A cat", quality="hd")

    error = str(excinfo.value)
    assert "quality" in error
    assert "Valid parameters are" in error
    assert "prompt, model, size, n" in error


def test_generate_rejects_multiple_invalid_parameters():
    """Multiple invalid parameters should all be listed in error message."""
    client = _make_client()

    with pytest.raises(TypeError) as excinfo:
        client.generate("A cat", quality="hd", style="realistic", foo="bar")

    error = str(excinfo.value)
    assert "foo" in error
    assert "quality" in error
    assert "style" in error


def test_generate_accepts_all_valid_parameters():
    """Valid parameters should not raise."""
    client = _make_client()

    # This should not raise a validation error (may raise network error, but that's OK).
    # We just want to verify the parameter validation passes.
    try:
        client.generate("A cat", model="google/nano-banana", size="1024x1024", n=2)
    except TypeError as e:
        # Should NOT be a parameter validation error
        if "Valid parameters are" in str(e):
            pytest.fail(f"Valid parameters rejected: {e}")
    except Exception:
        # Network/payment errors are fine for this test
        pass


def test_edit_rejects_quality_parameter():
    """Unsupported quality parameter in edit() should raise TypeError with helpful message."""
    client = _make_client()
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

    with pytest.raises(TypeError) as excinfo:
        client.edit("Make it red", image=data_uri, quality="hd")

    error = str(excinfo.value)
    assert "quality" in error
    assert "Valid parameters are" in error
    assert "prompt, image, model, mask, size, n" in error


def test_edit_accepts_all_valid_parameters():
    """Valid parameters should not raise parameter validation error."""
    client = _make_client()
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

    try:
        client.edit(
            "Make it red", image=data_uri, model="google/nano-banana", size="1024x1024", n=1
        )
    except TypeError as e:
        if "Valid parameters are" in str(e):
            pytest.fail(f"Valid parameters rejected: {e}")
    except Exception:
        # Network/payment errors are fine
        pass
