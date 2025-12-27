"""Unit tests for x402 payment protocol."""

import pytest
import base64
import json
from blockrun_llm.x402 import (
    create_nonce,
    create_payment_payload,
    parse_payment_required,
    extract_payment_details,
)
from ..helpers import TEST_ACCOUNT, TEST_RECIPIENT


class TestCreateNonce:
    def test_nonce_format(self):
        """Should generate nonce with correct format."""
        nonce = create_nonce()

        assert nonce.startswith("0x")
        assert len(nonce) == 66  # 0x + 64 hex chars

    def test_nonce_uniqueness(self):
        """Should generate unique nonces."""
        nonce1 = create_nonce()
        nonce2 = create_nonce()

        assert nonce1 != nonce2

    def test_nonce_is_hex(self):
        """Should contain only hex characters."""
        nonce = create_nonce()
        # Remove 0x prefix and check if valid hex
        hex_part = nonce[2:]
        assert all(c in "0123456789abcdef" for c in hex_part.lower())


class TestCreatePaymentPayload:
    def test_create_valid_payload(self):
        """Should create valid payment payload."""
        payload = create_payment_payload(
            account=TEST_ACCOUNT,
            recipient=TEST_RECIPIENT,
            amount="1000000",
            network="eip155:8453",
        )

        assert isinstance(payload, str)

        # Decode and verify structure
        decoded = json.loads(base64.b64decode(payload))
        assert decoded["x402Version"] == 2
        assert "payload" in decoded
        assert "signature" in decoded["payload"]
        assert decoded["payload"]["signature"].startswith("0x")

    def test_payload_includes_authorization(self):
        """Should include authorization details."""
        payload = create_payment_payload(
            account=TEST_ACCOUNT,
            recipient=TEST_RECIPIENT,
            amount="1000000",
        )

        decoded = json.loads(base64.b64decode(payload))
        auth = decoded["payload"]["authorization"]

        assert auth["from"] == TEST_ACCOUNT.address
        assert auth["to"] == TEST_RECIPIENT
        assert auth["value"] == "1000000"
        assert "validAfter" in auth
        assert "validBefore" in auth
        assert "nonce" in auth

    def test_payload_includes_resource_info(self):
        """Should include resource information."""
        payload = create_payment_payload(
            account=TEST_ACCOUNT,
            recipient=TEST_RECIPIENT,
            amount="1000000",
            resource_url="https://api.blockrun.ai/v1/test",
            resource_description="Test Resource",
        )

        decoded = json.loads(base64.b64decode(payload))
        assert decoded["resource"]["url"] == "https://api.blockrun.ai/v1/test"
        assert decoded["resource"]["description"] == "Test Resource"

    def test_payload_time_windows(self):
        """Should set valid time windows."""
        import time

        before = int(time.time())
        payload = create_payment_payload(
            account=TEST_ACCOUNT, recipient=TEST_RECIPIENT, amount="1000000"
        )
        after = int(time.time())

        decoded = json.loads(base64.b64decode(payload))
        auth = decoded["payload"]["authorization"]

        # Valid after should be in the past (allows clock skew)
        assert int(auth["validAfter"]) < before

        # Valid before should be in the future
        assert int(auth["validBefore"]) > after

    def test_custom_timeout(self):
        """Should use custom max timeout."""
        payload = create_payment_payload(
            account=TEST_ACCOUNT,
            recipient=TEST_RECIPIENT,
            amount="1000000",
            max_timeout_seconds=600,
        )

        decoded = json.loads(base64.b64decode(payload))
        assert decoded["accepted"]["maxTimeoutSeconds"] == 600


class TestParsePaymentRequired:
    def test_parse_valid_header(self):
        """Should parse valid payment required header."""
        data = {
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "amount": "1000000",
                    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "payTo": TEST_RECIPIENT,
                    "maxTimeoutSeconds": 300,
                }
            ],
        }

        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        result = parse_payment_required(encoded)

        assert result["x402Version"] == 2
        assert len(result["accepts"]) == 1

    def test_invalid_base64(self):
        """Should raise ValueError on invalid base64."""
        with pytest.raises(ValueError, match="invalid format"):
            parse_payment_required("invalid!!!")

    def test_invalid_json(self):
        """Should raise ValueError on invalid JSON."""
        with pytest.raises(ValueError, match="invalid format"):
            parse_payment_required(base64.b64encode(b"not json").decode())


class TestExtractPaymentDetails:
    def test_extract_details(self):
        """Should extract payment details."""
        payment_required = {
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "amount": "1000000",
                    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "payTo": TEST_RECIPIENT,
                    "maxTimeoutSeconds": 300,
                }
            ],
        }

        details = extract_payment_details(payment_required)

        assert details["amount"] == "1000000"
        assert details["recipient"] == TEST_RECIPIENT
        assert details["network"] == "eip155:8453"
        assert details["maxTimeoutSeconds"] == 300

    def test_empty_accepts(self):
        """Should raise ValueError on empty accepts."""
        with pytest.raises(ValueError, match="No payment options"):
            extract_payment_details({"x402Version": 2, "accepts": []})

    def test_default_timeout(self):
        """Should use default timeout if not specified."""
        payment_required = {
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "amount": "1000000",
                    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "payTo": TEST_RECIPIENT,
                    # maxTimeoutSeconds not specified
                }
            ],
        }

        details = extract_payment_details(payment_required)
        assert details["maxTimeoutSeconds"] == 300

    def test_include_resource(self):
        """Should include resource if present."""
        payment_required = {
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "amount": "1000000",
                    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "payTo": TEST_RECIPIENT,
                }
            ],
            "resource": {
                "url": "https://api.blockrun.ai/test",
                "description": "Test",
            },
        }

        details = extract_payment_details(payment_required)
        assert details["resource"]["url"] == "https://api.blockrun.ai/test"
