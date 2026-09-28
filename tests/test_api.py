import asyncio
import json
import pytest
import httpx
from app.main import create_app
from app.config import Settings
from tests.fakes import FakeWorker


@pytest.fixture
async def client():
    app = create_app(
        Settings(batch_wait_ms=5, max_output_tokens=32, max_prompt_chars=1000),
        FakeWorker(),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": "x", "max_new_tokens": 0},
        {"prompt": "x", "temperature": -1},
        {"prompt": "x", "top_p": 0},
        {"prompt": "x", "unexpected": 1},
    ],
)
async def test_validation(client, body):
    assert (await client.post("/generate", json=body)).status_code == 422


async def test_limits_health_metrics(client):
    assert (await client.get("/health")).json() == {
        "status": "ok",
        "model_loaded": True,
    }
    assert (
        await client.post("/generate", json={"prompt": "x", "max_new_tokens": 33})
    ).status_code == 422
    assert (
        await client.post("/generate", json={"prompt": "x" * 1001, "max_new_tokens": 1})
    ).status_code == 422
    assert (await client.post("/generate", content=b"x" * 140000)).status_code == 413
    assert "llm_queue_size" in (await client.get("/metrics")).text


async def test_api_concurrency_and_streaming(client):
    responses = await asyncio.gather(
        *(
            client.post(
                "/generate",
                json={"prompt": f"p{i}", "max_new_tokens": 3, "temperature": 0},
            )
            for i in range(16)
        )
    )
    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["request_id"] for r in responses}) == 16
    assert all(r.json()["text"].startswith(f"p{i}:") for i, r in enumerate(responses))
    response = await client.post(
        "/generate", json={"prompt": "stream", "max_new_tokens": 3, "stream": True}
    )
    blocks = [b for b in response.text.strip().split("\n\n") if b.startswith("event:")]
    assert len(blocks) == 4 and blocks[-1].startswith("event: done")
    payloads = [json.loads(b.split("data: ", 1)[1]) for b in blocks]
    assert "".join(p["text"] for p in payloads[:-1]) == payloads[-1]["text"]
    assert payloads[-1]["ttft_ms"] < payloads[-1]["latency_ms"]


async def test_error_response_and_stream_terminal(client):
    a = await client.post("/generate", json={"prompt": "RAISE", "max_new_tokens": 2})
    assert a.status_code == 500 and a.json() == {"error": "model_error"}
    b = await client.post(
        "/generate", json={"prompt": "RAISE", "max_new_tokens": 2, "stream": True}
    )
    assert b.status_code == 200 and "event: error" in b.text and "model_error" in b.text
