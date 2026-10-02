"""Unit tests for VideoClient.generate() parameter validation and body construction."""

import os

import pytest

from blockrun_llm import VideoClient
from blockrun_llm.types import VideoResponse


@pytest.fixture
def client():
    # Deterministic dummy key — never signs against a live endpoint in unit
    # tests; we only exercise local request/response paths.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return VideoClient()


@pytest.fixture
def account():
    # Reference media is an account-rail capability: both wallet gateways
    # refuse every reference_* field before quoting (blockrun#728).
    return VideoClient(private_key="brk_live_video_params_fixture")


def _capture(client, monkeypatch):
    captured = {}

    def fake_submit(body, budget_seconds):
        captured["body"] = body
        captured["budget"] = budget_seconds
        return VideoResponse(created=1, model=body["model"], data=[])

    monkeypatch.setattr(client, "_submit_and_poll", fake_submit)
    return captured


@pytest.fixture
def captured(client, monkeypatch):
    return _capture(client, monkeypatch)


@pytest.fixture
def account_captured(account, monkeypatch):
    return _capture(account, monkeypatch)


def test_first_last_frame_body(client, captured):
    client.generate(
        "the flower blooms",
        model="bytedance/seedance-1.5-pro",
        image_url="https://example.com/bud.jpg",
        last_frame_url="https://example.com/bloom.jpg",
    )
    assert captured["body"]["image_url"] == "https://example.com/bud.jpg"
    assert captured["body"]["last_frame_url"] == "https://example.com/bloom.jpg"


def test_reference_images_body(account, account_captured):
    urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
    account.generate(
        "the character from image 1 in the city from image 2",
        model="bytedance/seedance-2.0",
        reference_image_urls=urls,
    )
    assert account_captured["body"]["reference_image_urls"] == urls
    assert "image_url" not in account_captured["body"]


def test_reference_media_refused_on_the_wallet_rail(client, captured):
    for refs in (
        {"reference_image_urls": ["https://example.com/r.jpg"]},
        {"reference_videos": [{"url": "https://example.com/m.mp4"}]},
    ):
        with pytest.raises(ValueError, match="account rail"):
            client.generate("x", model="bytedance/seedance-2.0", **refs)
    assert captured == {}


def test_token360_passthroughs(client, captured):
    client.generate(
        "a calm lake at dawn",
        model="bytedance/seedance-2.0",
        aspect_ratio="16:9",
        seed=42,
        watermark=False,
        return_last_frame=True,
    )
    body = captured["body"]
    assert body["aspect_ratio"] == "16:9"
    assert body["seed"] == 42
    assert body["watermark"] is False
    assert body["return_last_frame"] is True


def test_last_frame_requires_image_url(client):
    with pytest.raises(ValueError, match="requires image_url"):
        client.generate("x", last_frame_url="https://example.com/last.jpg")


def test_last_frame_excludes_real_face(client):
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.generate(
            "x",
            image_url="https://example.com/first.jpg",
            last_frame_url="https://example.com/last.jpg",
            real_face_asset_id="ta_abc123",
        )


def test_reference_images_exclude_other_image_inputs(account):
    with pytest.raises(ValueError, match="mutually exclusive"):
        account.generate(
            "x",
            model="bytedance/seedance-2.0",
            image_url="https://example.com/seed.jpg",
            reference_image_urls=["https://example.com/r.jpg"],
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        account.generate(
            "x",
            model="bytedance/seedance-2.0",
            real_face_asset_id="ta_abc123",
            reference_image_urls=["https://example.com/r.jpg"],
        )


def test_reference_images_max_nine(account):
    with pytest.raises(ValueError, match="at most 9"):
        account.generate(
            "x",
            model="bytedance/seedance-2.0",
            reference_image_urls=[f"https://example.com/{i}.jpg" for i in range(10)],
        )


def test_image_url_and_real_face_still_exclusive(client):
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.generate(
            "x",
            image_url="https://example.com/a.jpg",
            real_face_asset_id="ta_abc123",
        )


# --- input_type ------------------------------------------------------------
# A declared seed mode the gateway cross-checks against the fields actually
# sent. Only the spelling is validated locally; the match is the gateway's
# call (400, unbilled) so the two can't drift.


def test_input_type_forwarded(client, captured):
    client.generate(
        "the flower blooms",
        model="bytedance/seedance-1.5-pro",
        image_url="https://example.com/bud.jpg",
        last_frame_url="https://example.com/bloom.jpg",
        input_type="first_last_frame",
    )
    assert captured["body"]["input_type"] == "first_last_frame"


def test_input_type_omitted_when_unset(client, captured):
    client.generate("a calm lake at dawn")
    assert "input_type" not in captured["body"]


@pytest.mark.parametrize("value", ["text", "image", "first_last_frame", "reference"])
def test_input_type_accepts_every_gateway_mode(client, captured, value):
    client.generate("x", input_type=value)
    assert captured["body"]["input_type"] == value


def test_input_type_rejects_unknown_value(client):
    with pytest.raises(ValueError, match="input_type must be one of"):
        client.generate("x", input_type="img")


def test_input_type_mismatch_is_left_to_the_gateway(client, captured):
    """Declaring a mode that contradicts the seed fields must still be sent.

    The gateway owns that check and answers 400 before charging; rejecting it
    here would fork the inference into a second copy that drifts.
    """
    client.generate("x", input_type="image")  # no image_url — gateway's call
    assert captured["body"]["input_type"] == "image"


def test_mixed_references_and_controls_reach_body(account, account_captured):
    account.generate(
        "follow the motion",
        model="bytedance/seedance-2.0",
        reference_image_urls=["https://example.com/person.png"],
        reference_videos=[{"url": "https://example.com/motion.mp4"}],
        reference_audios=[{"url": "https://example.com/music.mp3"}],
        bitrate_mode="high",
        safety_identifier="test",
        return_last_frame=True,
        input_type="reference",
    )
    body = account_captured["body"]
    assert body["reference_videos"] == [{"url": "https://example.com/motion.mp4"}]
    assert body["reference_audios"] == [{"url": "https://example.com/music.mp3"}]
    assert body["reference_image_urls"] == ["https://example.com/person.png"]
    assert body["bitrate_mode"] == "high"
    assert body["safety_identifier"] == "test"
    assert body["input_type"] == "reference"


def test_25_reference_limit_and_output_controls(account, account_captured):
    images = ["https://example.com/person.png"] * 30
    account.generate(
        "test", model="bytedance/seedance-2.5", reference_image_urls=images, output_format="mov"
    )
    assert account_captured["body"]["reference_image_urls"] == images
    assert account_captured["body"]["output_format"] == "mov"
    with pytest.raises(ValueError, match="at most 30"):
        account.generate(
            "test", model="bytedance/seedance-2.5", reference_image_urls=images + images
        )
    account.generate("test", model="bytedance/seedance-1.5-pro", camera_fixed=False)
    assert account_captured["body"]["camera_fixed"] is False


def test_reference_media_cannot_be_frame_seeds(account):
    with pytest.raises(ValueError, match="mutually exclusive"):
        account.generate(
            "test",
            model="bytedance/seedance-2.0",
            image_url="https://example.com/frame.png",
            reference_videos=[{"url": "https://example.com/motion.mp4"}],
        )


# Per-model guards — mirror the MCP's capability table (blockrun-mcp
# src/tools/video.ts). On the account rail there is no quote step, so these
# refusals are the only thing standing between a bad request and a charge.
_CLIP = [{"url": "https://example.com/motion.mp4"}]


@pytest.mark.parametrize(
    "model, kwargs, message",
    [
        # 2.5 takes reference images but not clips
        ("bytedance/seedance-2.5", {"reference_videos": _CLIP}, "2.5 takes reference IMAGES"),
        (
            "bytedance/seedance-2.5",
            {"reference_image_urls": ["https://e/x.png"], "reference_audios": _CLIP},
            "does not accept reference video or audio",
        ),
        (
            "bytedance/seedance-1.5-pro",
            {"reference_image_urls": ["https://e/x.png"]},
            "does not accept reference images",
        ),
        (
            "xai/grok-imagine-video",
            {"reference_image_urls": ["https://e/x.png"]},
            "does not accept reference images",
        ),
        (
            "bytedance/seedance-2.0",
            {"reference_audios": _CLIP},
            "requires a reference image or video",
        ),
        ("bytedance/seedance-2.0", {"reference_videos": _CLIP * 4}, "at most 3 clips"),
        (
            "bytedance/seedance-2.0",
            {"reference_videos": [{"url": "ftp://e/x.mp4"}]},
            "entries must be",
        ),
        (
            "bytedance/seedance-2.0",
            {"reference_videos": [{"url": "https://e/x.mp4", "start": 3}]},
            "no other keys",
        ),
        (
            "bytedance/seedance-2.0",
            {"reference_image_urls": ["data:image/png;base64,AA=="]},
            "http\\(s\\) URLs",
        ),
        ("bytedance/seedance-1.5-pro", {"bitrate_mode": "high"}, "requires a Seedance 2.x"),
        ("bytedance/seedance-2.0", {"bitrate_mode": "ultra"}, "bitrate_mode must be one of"),
        ("bytedance/seedance-2.0", {"output_format": "mov"}, "requires bytedance/seedance-2.5"),
        ("bytedance/seedance-2.5", {"output_format": "webm"}, "output_format must be one of"),
        ("bytedance/seedance-2.0", {"camera_fixed": True}, "requires bytedance/seedance-1.5-pro"),
        ("xai/grok-imagine-video", {"safety_identifier": "u1"}, "requires a Seedance model"),
    ],
)
def test_per_model_guards_refuse_before_submit(account, account_captured, model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        account.generate("x", model=model, **kwargs)
    assert account_captured == {}


def test_empty_reference_lists_are_omitted(account, account_captured):
    account.generate(
        "x", model="bytedance/seedance-2.0", reference_image_urls=[], reference_videos=[]
    )
    assert "reference_image_urls" not in account_captured["body"]
    assert "reference_videos" not in account_captured["body"]


def test_last_frame_response_is_not_dropped():
    result = VideoResponse(
        created=1,
        model="bytedance/seedance-2.0",
        data=[
            {
                "url": "https://example.com/movie.mp4",
                "last_frame_url": "https://example.com/last.png",
                "last_frame_backed_up": True,
            }
        ],
    )
    assert result.data[0].last_frame_url == "https://example.com/last.png"
    assert result.data[0].last_frame_backed_up is True
