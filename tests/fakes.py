import time
from app.inference.request import Output


class FakeWorker:
    """No model/network for systems tests. Benchmarks use the real worker."""

    def __init__(self, delay=0.002):
        self.delay, self.loaded = delay, False
        self.identity = "fake@tests-only"
        self.batches = []
        self.closed = False

    def load(self):
        self.loaded = True

    def close(self):
        self.loaded = False
        self.closed = True

    def run_batch(self, jobs, emit):
        self.batches.append([j.params.prompt for j in jobs])
        if any(j.params.prompt == "RAISE" for j in jobs):
            raise RuntimeError("Intentional model failure")
        texts = [""] * len(jobs)
        for i in range(max(j.params.max_new_tokens for j in jobs)):
            time.sleep(self.delay)
            for row, job in enumerate(jobs):
                if job.stop.is_set() or i >= job.params.max_new_tokens:
                    continue
                delta = f"{job.params.prompt}:{i} "
                texts[row] += delta
                emit(job, "token", {"text": delta, "token_id": i + 1})
                if i + 1 == job.params.max_new_tokens:
                    emit(
                        job,
                        "done",
                        Output(
                            texts[row], len(job.params.prompt.split()), i + 1, "length"
                        ),
                    )
