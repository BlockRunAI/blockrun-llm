// Generates tests/unit/x402_upto_reference_vector.json from the official
// @x402/evm 2.28.0 upto client (UptoEvmScheme + trySignEip2612PermitExtension),
// which tests/unit/test_x402_upto.py checks the Python signer against.
// Deterministic: Date.now and crypto.getRandomValues are pinned.
//
// Not run by CI. To regenerate:
//   npm pack @x402/evm@2.28.0 && tar xzf x402-evm-2.28.0.tgz   # → ./package
//   (cd package && npm i viem@2 --no-save)
//   cp scripts/gen-upto-vector.mjs package/ && node package/gen-upto-vector.mjs \
//     > tests/unit/x402_upto_reference_vector.json
// The chunk file names below are those of the 2.28.0 build; the package index
// is avoided because it pulls in @x402/core.
import { privateKeyToAccount } from "viem/accounts";
import { hashTypedData } from "viem";
import { UptoEvmScheme } from "./dist/esm/chunk-3LEU2NZ6.mjs";
import { PERMIT2_ADDRESS, x402UptoPermit2ProxyAddress } from "./dist/esm/chunk-GPMYUGHX.mjs";

const NOW_MS = 1790000000000;
Date.now = () => NOW_MS;
Object.defineProperty(globalThis, "crypto", {
  value: { getRandomValues: (arr) => { arr.fill(0x11); return arr; } },
  configurable: true,
});

const account = privateKeyToAccount(
  "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
);

const requirements = {
  scheme: "upto",
  network: "eip155:8453",
  amount: "123456",
  asset: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
  payTo: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
  maxTimeoutSeconds: 300,
  extra: {
    name: "USD Coin",
    version: "2",
    facilitatorAddress: "0x97AcCe27D5069544480BDe0F04D9F47d7422a016",
  },
};

const reads = [];
const signer = {
  address: account.address,
  signTypedData: (msg) => account.signTypedData(msg),
  readContract: async (args) => {
    reads.push(args.functionName);
    if (args.functionName === "allowance") return 0n;
    if (args.functionName === "nonces") return 7n;
    throw new Error("unexpected read " + args.functionName);
  },
};

const scheme = new UptoEvmScheme(signer);

// 1) No gas sponsoring declared → bare Permit2 payload.
const bare = await scheme.createPaymentPayload(2, requirements, { extensions: {} });

// 2) Gas sponsoring declared, allowance 0 → EIP-2612 permit attached. Its
//    value is the per-call ceiling (permitted.amount): the upto proxy reverts
//    with Permit2612AmountMismatch on anything else.
const sponsored = await scheme.createPaymentPayload(2, requirements, {
  extensions: { eip2612GasSponsoring: { info: { version: "1" } } },
});

const a = bare.payload.permit2Authorization;
const permit2Digest = hashTypedData({
  domain: { name: "Permit2", chainId: 8453, verifyingContract: PERMIT2_ADDRESS },
  types: {
    PermitWitnessTransferFrom: [
      { name: "permitted", type: "TokenPermissions" },
      { name: "spender", type: "address" },
      { name: "nonce", type: "uint256" },
      { name: "deadline", type: "uint256" },
      { name: "witness", type: "Witness" },
    ],
    TokenPermissions: [
      { name: "token", type: "address" },
      { name: "amount", type: "uint256" },
    ],
    Witness: [
      { name: "to", type: "address" },
      { name: "facilitator", type: "address" },
      { name: "validAfter", type: "uint256" },
    ],
  },
  primaryType: "PermitWitnessTransferFrom",
  message: {
    permitted: { token: a.permitted.token, amount: BigInt(a.permitted.amount) },
    spender: a.spender,
    nonce: BigInt(a.nonce),
    deadline: BigInt(a.deadline),
    witness: {
      to: a.witness.to,
      facilitator: a.witness.facilitator,
      validAfter: BigInt(a.witness.validAfter),
    },
  },
});

const eip2612DigestOf = (info) => hashTypedData({
  domain: {
    name: "USD Coin",
    version: "2",
    chainId: 8453,
    verifyingContract: requirements.asset,
  },
  types: {
    Permit: [
      { name: "owner", type: "address" },
      { name: "spender", type: "address" },
      { name: "value", type: "uint256" },
      { name: "nonce", type: "uint256" },
      { name: "deadline", type: "uint256" },
    ],
  },
  primaryType: "Permit",
  message: {
    owner: info.from,
    spender: info.spender,
    value: BigInt(info.amount),
    nonce: BigInt(info.nonce),
    deadline: BigInt(info.deadline),
  },
});
const eip2612Digest = eip2612DigestOf(sponsored.extensions.eip2612GasSponsoring.info);

console.log(
  JSON.stringify(
    {
      source: "@x402/evm 2.28.0 UptoEvmScheme (viem " + (await import("viem/package.json", { with: { type: "json" } })).default.version + ")",
      inputs: { privateKey: "hardhat #0", nowSeconds: NOW_MS / 1000, randomBytes: "0x11 * 32", requirements, usdcNonce: "7", allowance: "0" },
      constants: { PERMIT2_ADDRESS, x402UptoPermit2ProxyAddress },
      bare,
      sponsored,
      permit2Digest,
      eip2612Digest,
      reads,
    },
    null,
    2,
  ),
);
