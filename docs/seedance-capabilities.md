# Seedance capabilities in the Python SDK

What `VideoClient.generate` (Base) and `SolanaLLMClient.video` /
`AsyncSolanaLLMClient.video` accept, per model and per rail. The SDK refuses
every combination below that the gateway would refuse, locally and before any
request is sent. On the account rail this is the only check that runs before
money moves, because there is no quote step there. The guards live in
`blockrun_llm/validation.py` (`validate_video_request`). They mirror the MCP's
table in `blockrun-mcp/src/tools/video.ts`, and both should change together.

## Per model

| Model | First + last frame | Reference images | Reference video / audio clips |
| --- | --- | --- | --- |
| seedance-1.5-pro | Yes | No | No |
| seedance-2.0 / 2.0-fast / 2.0-mini | Yes | 1–9 | 1–3 of each |
| seedance-2.5 | Yes (not on the Solana wallet gateway yet) | 1–30 | No |
| grok-imagine-video, sora-2 | No | No | No |

- Reference audio needs at least one reference image or video in the same request.
- Frame seeds (`image_url`, `last_frame_url`, `real_face_asset_id`) and
  reference inputs are mutually exclusive.
- Each clip is `{"url": "https://…"}`, optionally with `"role": "reference"`.
  No other keys are allowed.

## Per rail

| Rail | Reference media | seedance-2.5 last frame |
| --- | --- | --- |
| Account (`BLOCKRUN_API_KEY`, api.blockrun.ai) | Yes | Yes |
| Base wallet (blockrun.ai) | Refused (gateway 400s before quoting) | Yes |
| Solana wallet (sol.blockrun.ai) | Refused (gateway 400s before quoting) | Refused (gateway 400s before quoting) |

On the account rail the job is billed when it is accepted, not on completion.
A job that times out in the SDK has already been paid for. It stays claimable
for about 48h via the `poll_url` in the error.

## Cost of reference clips

Reference video and audio are billed per reference second, at the model's
15.2s reference ceiling, whatever the clip's real length. The gateway only
sees URLs, never durations. One clip on a 5s 720p seedance-2.0-mini render is
roughly 4x the price of the render alone. At 4K with three of each type the
price runs into the hundreds of dollars. Audio seconds bill at about 0.3x the
video rate.

## Output controls

| Field | Models | Values |
| --- | --- | --- |
| `bitrate_mode` | seedance-2.0 / 2.0-fast / 2.0-mini / 2.5 | `standard`, `high` |
| `output_format` | seedance-2.5 | `mp4`, `mov` |
| `camera_fixed` | seedance-1.5-pro | bool |
| `safety_identifier` | any Seedance | str |
| `return_last_frame` | any Seedance | bool |

When the upstream returns a last frame, `data[0].last_frame_url` and
`data[0].last_frame_backed_up` carry it.
