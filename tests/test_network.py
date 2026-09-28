"""Real sockets: ASGITransport alone buffers streams and cannot prove early delivery."""

import asyncio
import socket
import httpx
import uvicorn
from app.main import create_app
from app.config import Settings
from tests.fakes import FakeWorker


async def test_real_sse_arrives_early_and_disconnect_cancels():
    app = create_app(Settings(batch_wait_ms=0, cache_size=0), FakeWorker(0.025))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        assert server.started
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False
        ) as client:
            async with client.stream(
                "POST",
                "/generate",
                json={"prompt": "network", "max_new_tokens": 100, "stream": True},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        assert app.state.scheduler.metrics.counts["completed"] == 0
                        break
            for _ in range(100):
                if not app.state.scheduler.jobs:
                    break
                await asyncio.sleep(0.01)
            assert not app.state.scheduler.jobs
            assert app.state.scheduler.metrics.counts["failed"] == 1
            assert app.state.scheduler.metrics.counts["completed"] == 0
            assert (await client.get("/health")).status_code == 200
    finally:
        server.should_exit = True
        await task
        sock.close()
