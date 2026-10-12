"""An API key must never follow a server-supplied ``poll_url`` off-origin.

Every poll on the account rail carries ``Authorization: Bearer brk_...``.
``VideoClient`` and the shared ``jobs.py`` poller (images, music) used to return
an absolute ``poll_url`` as-is, before ``resolve_poll_url`` could pin it to the
gateway's origin, so a response naming ``https://evil.example/...`` received
the key. Absolute poll URLs now go through the same pin as relative ones; a
refusal is an ``APIError`` carrying the accepted job's id and the refused URL.

``httpx.MockTransport`` keeps the network out; any request to a foreign host
is recorded so the tests can assert none was made.
"""

from __future__ import annotations

import httpx
import pytest

from blockrun_llm import MusicClient, VideoClient
from blockrun_llm.apikey import DEFAULT_API_KEY_URL
from blockrun_llm.types import APIError

KEY = "brk_live_poll_origin_fixture"
FOREIGN = "https://evil.example/v1/jobs/J1"


def _handler(poll_url: str, done: dict, seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host != httpx.URL(DEFAULT_API_KEY_URL).host:
            return httpx.Response(200, json=done)
        if request.method == "POST":
            return httpx.Response(202, json={"id": "J1", "poll_url": poll_url, "status": "queued"})
        return httpx.Response(200, json=done)

    return handler


_VIDEO_DONE = {
    "status": "completed",
    "created": 1,
    "model": "bytedance/seedance-2.0",
    "data": [{"url": "https://cdn/v.mp4"}],
}
_MUSIC_DONE = {
    "id": "J1",
    "status": "completed",
    "created": 1,
    "model": "minimax/music-2.5+",
    "data": [{"url": "https://cdn/t.mp3"}],
}


def _video(handler, monkeypatch: pytest.MonkeyPatch) -> VideoClient:
    monkeypatch.setattr(VideoClient, "POLL_INTERVAL_SECONDS", 0.0)
    client = VideoClient(private_key=KEY)
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler), headers=client._client.headers
    )
    return client


def _music(handler, monkeypatch: pytest.MonkeyPatch) -> MusicClient:
    monkeypatch.setattr(MusicClient, "MUSIC_POLL_INTERVAL_SECONDS", 0.0)
    client = MusicClient(private_key=KEY)
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler), headers=client._client.headers
    )
    return client


@pytest.mark.parametrize(
    "poll_url",
    [FOREIGN, "http://api.blockrun.ai/v1/jobs/J1", "//evil.example/v1/jobs/J1"],
)
def test_video_never_sends_the_key_to_a_foreign_poll_origin(monkeypatch, poll_url):
    seen: list[httpx.Request] = []
    client = _video(_handler(poll_url, _VIDEO_DONE, seen), monkeypatch)
    with pytest.raises(APIError, match="different polling origin") as exc:
        client.generate("x", model="bytedance/seedance-2.0")
    assert [r.method for r in seen] == ["POST"]
    assert exc.value.response == {"id": "J1", "poll_url": poll_url}


def test_video_same_origin_absolute_poll_url_still_polls(monkeypatch):
    seen: list[httpx.Request] = []
    same = f"{DEFAULT_API_KEY_URL}/v1/videos/generations/J1?token=a"
    client = _video(_handler(same, _VIDEO_DONE, seen), monkeypatch)
    resp = client.generate("x", model="bytedance/seedance-2.0")
    assert resp.data[0].url == "https://cdn/v.mp4"
    assert [r.method for r in seen] == ["POST", "GET"]
    assert str(seen[1].url) == same
    assert seen[1].headers["authorization"] == f"Bearer {KEY}"


def test_music_never_sends_the_key_to_a_foreign_poll_origin(monkeypatch):
    seen: list[httpx.Request] = []
    client = _music(_handler(FOREIGN, _MUSIC_DONE, seen), monkeypatch)
    with pytest.raises(APIError, match="different polling origin") as exc:
        client.generate("x")
    assert [r.method for r in seen] == ["POST"]
    assert exc.value.response == {"id": "J1", "poll_url": FOREIGN}


def test_music_same_origin_absolute_poll_url_still_polls(monkeypatch):
    seen: list[httpx.Request] = []
    same = f"{DEFAULT_API_KEY_URL}/v1/audio/generations/J1"
    client = _music(_handler(same, _MUSIC_DONE, seen), monkeypatch)
    assert client.generate("x").data[0].url == "https://cdn/t.mp3"
    assert [str(r.url) for r in seen[1:]] == [same]


def test_wallet_rail_absolute_poll_url_is_unchanged():
    # No key, nothing to leak: the wallet rail keeps taking an absolute
    # poll_url as given (its signature is bound to the job, not the host).
    from blockrun_llm.jobs import absolute_poll_url

    assert absolute_poll_url(FOREIGN, "https://blockrun.ai/api", None) == FOREIGN
