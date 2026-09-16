"""The EIP-712 domain a payment is signed against follows the 402's network.

Until 2.x's Arc release the chain table knew Base and Base Sepolia and fell
back to Base for anything else, while `asset` and `extra` were taken from the
402 as given. Against arc.blockrun.ai (eip155:5042, USDC at 0x3600…, domain
name "USDC") that produced a signature over chainId 8453 with Arc's contract
— invalid; the facilitator recovers a different signer and answers 401 after
the SDK has reported a payment.

The 402 now SELECTS a network from the SDK's own table, which supplies the
chainId, the USDC address and the domain; a hostile 402's `extra` cannot
steer a signature onto another contract, an unknown network is refused
naming what is supported, and a 402 whose `asset` is not that network's USDC
is refused before anything is signed. Mirrors @blockrun/llm 3.16.0.
"""

import base64
import json

import pytest
from eth_account import Account
from eth_account.messages import encode_typed_data

from blockrun_llm.x402 import EVM_NETWORKS, create_payment_payload, evm_network

from ..helpers import TEST_ACCOUNT, TEST_RECIPIENT

TYPES = {
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ],
}


def sign_and_decode(network: str, **kwargs) -> dict:
    payload = create_payment_payload(
        account=TEST_ACCOUNT, recipient=TEST_RECIPIENT, amount="2000", network=network, **kwargs
    )
    return json.loads(base64.b64decode(payload))


def recovered_signer(decoded: dict, domain: dict) -> str:
    a = decoded["payload"]["authorization"]
    message = {
        "from": a["from"],
        "to": a["to"],
        "value": int(a["value"]),
        "validAfter": int(a["validAfter"]),
        "validBefore": int(a["validBefore"]),
        "nonce": bytes.fromhex(a["nonce"][2:]),
    }
    signable = encode_typed_data(domain_data=domain, message_types=TYPES, message_data=message)
    return Account.recover_message(signable, signature=decoded["payload"]["signature"])


class TestSignedDomainFollowsNetwork:
    def test_knows_arc_base_and_base_sepolia(self):
        arc = evm_network("eip155:5042")
        assert arc["chain_id"] == 5042
        assert arc["usdc"] == "0x3600000000000000000000000000000000000000"
        assert arc["domain"]["name"] == "USDC"
        base = evm_network("eip155:8453")
        assert base["chain_id"] == 8453
        assert base["domain"]["name"] == "USD Coin"
        assert evm_network("eip155:84532")["domain"]["name"] == "USDC"
        # The old alias still resolves.
        assert evm_network("base-sepolia")["chain_id"] == 84532

    def test_arc_payment_signs_arc_domain_not_base(self):
        decoded = sign_and_decode("eip155:5042")
        assert (
            recovered_signer(decoded, EVM_NETWORKS["eip155:5042"]["domain"]) == TEST_ACCOUNT.address
        )
        assert (
            recovered_signer(decoded, EVM_NETWORKS["eip155:8453"]["domain"]) != TEST_ACCOUNT.address
        )
        assert decoded["accepted"]["network"] == "eip155:5042"
        assert decoded["accepted"]["asset"] == "0x3600000000000000000000000000000000000000"
        assert decoded["accepted"]["extra"] == {"name": "USDC", "version": "2"}

    def test_base_payment_unchanged(self):
        decoded = sign_and_decode("eip155:8453")
        assert (
            recovered_signer(decoded, EVM_NETWORKS["eip155:8453"]["domain"]) == TEST_ACCOUNT.address
        )
        assert decoded["accepted"]["asset"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert decoded["accepted"]["extra"] == {"name": "USD Coin", "version": "2"}

    def test_unknown_network_is_refused_not_signed_as_base(self):
        with pytest.raises(ValueError, match="eip155:1") as e:
            sign_and_decode("eip155:1")
        assert "eip155:5042" in str(e.value)  # names what it does know

    def test_asset_not_that_networks_usdc_is_refused(self):
        with pytest.raises(ValueError, match="(?i)asset"):
            sign_and_decode("eip155:5042", asset="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
        # Case-insensitive on the address; the gateway checksums, wallets often do not.
        ok = sign_and_decode(
            "eip155:5042", asset="0x3600000000000000000000000000000000000000".lower()
        )
        assert ok["accepted"]["asset"] == "0x3600000000000000000000000000000000000000"

    def test_402_extra_is_ignored_for_the_domain(self):
        decoded = sign_and_decode("eip155:5042", extra={"name": "USD Coin", "version": "9"})
        assert (
            recovered_signer(decoded, EVM_NETWORKS["eip155:5042"]["domain"]) == TEST_ACCOUNT.address
        )
        assert decoded["accepted"]["extra"] == {"name": "USDC", "version": "2"}
