from collections import deque
from app.inference.request import Job


def take_batch(queue: deque[Job], limit: int) -> list[Job]:
    """FIFO static batches; all sampling options are implemented per row."""
    batch = []
    while queue and len(batch) < limit:
        job = queue.popleft()
        if not job.future.done():
            batch.append(job)
    return batch
