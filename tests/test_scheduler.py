import asyncio
import time
import pytest
from app.config import Settings
from app.inference.scheduler import Scheduler
from app.inference.request import InferenceError
from app.api.schemas import GenerateRequest
from tests.fakes import FakeWorker


def params(prompt="x", **kw):
    return GenerateRequest(
        prompt=prompt, max_new_tokens=kw.pop("max_new_tokens", 2), **kw
    )


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
async def test_concurrent_mapping(concurrency):
    worker = FakeWorker()
    scheduler = Scheduler(
        Settings(max_batch_size=8, batch_wait_ms=20, cache_size=0), worker
    )
    await scheduler.start()
    try:
        jobs = [scheduler.submit(params(str(i))) for i in range(concurrency)]
        results = await asyncio.gather(*(j.future for j in jobs))
        assert len({r.request_id for r in results}) == concurrency
        assert all(r.text == f"{i}:0 {i}:1 " for i, r in enumerate(results))
        assert sum(map(len, worker.batches)) == concurrency
        assert all(len(b) <= 8 for b in worker.batches)
        if concurrency >= 8:
            assert len(worker.batches[0]) == 8
    finally:
        await scheduler.close()
    assert worker.closed


async def test_oldest_request_window():
    scheduler = Scheduler(Settings(batch_wait_ms=70, max_batch_size=8), FakeWorker())
    await scheduler.start()
    try:
        start = time.perf_counter()
        a = scheduler.submit(params("a"))
        await asyncio.sleep(0.04)
        b = scheduler.submit(params("b"))
        await asyncio.gather(a.future, b.future)
        assert 0.055 <= a.started - start < 0.16
        assert a.started == b.started
    finally:
        await scheduler.close()


async def test_full_batch_does_not_wait_for_window():
    s = Scheduler(Settings(batch_wait_ms=1000, max_batch_size=2), FakeWorker())
    await s.start()
    try:
        a = s.submit(params("a"))
        b = s.submit(params("b"))
        await asyncio.wait_for(asyncio.gather(a.future, b.future), 0.5)
    finally:
        await s.close()


async def test_backpressure_and_cancel_reclaims_queue_slot():
    s = Scheduler(Settings(max_queue_size=1, batch_wait_ms=500), FakeWorker())
    await s.start()
    try:
        a = s.submit(params())
        with pytest.raises(InferenceError, match="queue_full"):
            s.submit(params())
        s.cancel(a)
        b = s.submit(params("new"))
        assert b.id != a.id
        s.cancel(b)
    finally:
        await s.close()


@pytest.mark.parametrize("kind", ["queue", "inference"])
async def test_timeouts(kind):
    cfg = Settings(
        batch_wait_ms=150 if kind == "queue" else 0,
        queue_timeout_s=0.03 if kind == "queue" else 5,
        request_timeout_s=0.03 if kind == "inference" else 5,
    )
    s = Scheduler(cfg, FakeWorker(0.025))
    await s.start()
    try:
        job = s.submit(params(max_new_tokens=100))
        with pytest.raises(InferenceError, match=kind + "_timeout"):
            await job.future
        assert job.stop.is_set()
    finally:
        await s.close()


async def test_model_error_does_not_kill_scheduler():
    s = Scheduler(Settings(batch_wait_ms=0), FakeWorker())
    await s.start()
    try:
        with pytest.raises(InferenceError, match="model_error"):
            await s.submit(params("RAISE")).future
        assert (await s.submit(params("ok")).future).text.startswith("ok")
        assert s.healthy
    finally:
        await s.close()


async def test_cache_replays_with_fresh_request_metadata():
    w = FakeWorker()
    s = Scheduler(Settings(batch_wait_ms=0), w)
    await s.start()
    try:
        a = await s.submit(params(temperature=0)).future
        b = await s.submit(params(temperature=0, stream=True)).future
        assert b.cached and not a.cached
        assert a.text == b.text and a.request_id != b.request_id
        assert len(w.batches) == 1
        assert b.queue_time_ms == 0 and b.ttft_ms is None
    finally:
        await s.close()


async def test_shutdown_finishes_active_rejects_queued():
    w = FakeWorker(0.03)
    s = Scheduler(Settings(batch_wait_ms=0, max_batch_size=1), w)
    await s.start()
    a = s.submit(params("active", max_new_tokens=3))
    while a.started is None:
        await asyncio.sleep(0.001)
    b = s.submit(params("queued"))
    closing = asyncio.create_task(s.close())
    assert (await a.future).text.startswith("active")
    with pytest.raises(InferenceError, match="server_shutdown"):
        await b.future
    await closing
    assert w.closed and not s.healthy
    assert not s.jobs


async def test_cancel_one_row_keeps_peer_and_suppresses_late_result():
    s = Scheduler(Settings(max_batch_size=2, batch_wait_ms=100), FakeWorker(0.01))
    await s.start()
    try:
        a = s.submit(params("a", stream=True, max_new_tokens=8))
        b = s.submit(params("b", max_new_tokens=4))
        kind, _ = await a.events.get()
        assert kind == "token" and not a.future.done()
        s.cancel(a)
        assert (await b.future).output_tokens == 4
        with pytest.raises(InferenceError, match="client_cancelled"):
            await a.future
        assert s.metrics.counts["completed"] == 1
    finally:
        await s.close()
