import asyncio
from collections import deque
from app.inference.batching import take_batch
from app.inference.request import Job
from app.api.schemas import GenerateRequest


async def test_fifo_skips_cancelled():
    loop = asyncio.get_running_loop()
    jobs = [
        Job(GenerateRequest(prompt=str(i)), loop.create_future(), asyncio.Queue())
        for i in range(5)
    ]
    jobs[1].future.cancel()
    q = deque(jobs)
    batch = take_batch(q, 2)
    assert [j.params.prompt for j in batch] == ["0", "2"]
    assert [j.params.prompt for j in q] == ["3", "4"]
