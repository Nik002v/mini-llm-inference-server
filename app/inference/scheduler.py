import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from app.config import Settings
from app.api.schemas import GenerateRequest, GenerateResponse
from app.cache.lru import LRUCache
from app.metrics.metrics import Metrics
from app.inference.request import Job, Output, InferenceError
from app.inference.batching import take_batch

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, settings: Settings, worker, metrics: Metrics | None = None):
        self.settings, self.worker = settings, worker
        self.metrics = metrics or Metrics()
        self.queue: deque[Job] = deque()
        self.jobs: dict[str, Job] = {}
        self.changed = asyncio.Event()
        self.accepting = False
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model-worker"
        )
        self.cache = LRUCache(settings.cache_size, "not-loaded")
        self.active_count = 0

    async def start(self):
        self.loop = asyncio.get_running_loop()
        try:
            await self.loop.run_in_executor(self.executor, self.worker.load)
        except BaseException:
            self.executor.shutdown(wait=True)
            raise
        self.cache.model_identity = self.worker.identity
        self.accepting = True
        self.runner = asyncio.create_task(self._run(), name="batch-scheduler")
        self.monitor = asyncio.create_task(self._timeouts(), name="deadline-monitor")

    def submit(self, params: GenerateRequest) -> Job:
        self.metrics.inc("requests")
        if not self.healthy:
            self.metrics.error("unavailable")
            raise InferenceError("unavailable", 503)
        if (
            len(params.prompt) > self.settings.max_prompt_chars
            or params.max_new_tokens > self.settings.max_output_tokens
        ):
            self.metrics.error("request_limits_exceeded")
            raise InferenceError("request_limits_exceeded", 422)
        cached = self.cache.get(self.cache.key(params))
        self._prune()
        if cached is None and len(self.queue) >= self.settings.max_queue_size:
            self.metrics.error("queue_full")
            raise InferenceError("queue_full", 503)
        # One token event per generated token plus one terminal event: bounded per request.
        job = Job(
            params, self.loop.create_future(), asyncio.Queue(params.max_new_tokens + 1)
        )
        # Retrieve terminal exceptions even if a disconnected client no longer awaits the future.
        job.future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        self.jobs[job.id] = job
        if cached is not None:
            self.metrics.inc("cache_hits")
            self._finish(job, cached, cached=True)
        else:
            if self.cache.key(params) is not None:
                self.metrics.inc("cache_misses")
            self.queue.append(job)
            self._prune()
            self.changed.set()
        return job

    @property
    def healthy(self):
        return (
            self.accepting
            and self.worker.loaded
            and hasattr(self, "runner")
            and not self.runner.done()
        )

    def _prune(self):
        self.queue = deque(j for j in self.queue if not j.future.done())
        self.metrics.queue.set(len(self.queue))

    def fail(self, job: Job, error: InferenceError):
        if job.future.done():
            return
        job.stop.set()
        job.future.set_exception(error)
        # Discard pending chunks on errors, ensuring the terminal event is always deliverable.
        while not job.events.empty():
            job.events.get_nowait()
        job.events.put_nowait(("error", {"error": error.code, "request_id": job.id}))
        self.jobs.pop(job.id, None)
        self.metrics.error(error.code)
        self.changed.set()

    def cancel(self, job: Job):
        self.fail(job, InferenceError("client_cancelled", 499))
        self._prune()

    def _finish(self, job: Job, output: Output, cached=False):
        if job.future.done():
            return
        now = time.perf_counter()
        latency = now - job.created
        queue_time = (job.started or now) - job.created if not cached else 0
        ttft = job.first_token - job.created if job.first_token is not None else None
        response = GenerateResponse(
            request_id=job.id,
            text=output.text,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=latency * 1000,
            queue_time_ms=queue_time * 1000,
            ttft_ms=ttft * 1000 if ttft is not None else None,
            finish_reason=output.finish_reason,
            cached=cached,
        )
        self.cache.put(self.cache.key(job.params), output)
        job.future.set_result(response)
        job.events.put_nowait(("done", response.model_dump()))
        self.jobs.pop(job.id, None)
        self.metrics.inc("completed")
        self.metrics.latency.observe(latency)
        self.metrics.queue_time.observe(queue_time)
        if ttft is not None:
            self.metrics.ttft.observe(ttft)

    def _receive(self, job: Job, kind: str, payload):
        if kind == "token":
            self.metrics.inc("generated_tokens")
        if job.future.done():
            return
        if kind == "token":
            if job.first_token is None:
                job.first_token = time.perf_counter()
            if job.params.stream:
                job.events.put_nowait(("token", {"request_id": job.id, **payload}))
        elif kind == "done":
            self._finish(job, payload)
        elif kind == "error":
            self.fail(job, payload)

    def _emit(self, job, kind, payload):
        self.loop.call_soon_threadsafe(self._receive, job, kind, payload)

    async def _run(self):
        try:
            while self.accepting or self.queue:
                self._prune()
                if not self.queue:
                    self.changed.clear()
                    await self.changed.wait()
                    continue
                due = self.queue[0].created + self.settings.batch_wait_ms / 1000
                while self.accepting and len(self.queue) < self.settings.max_batch_size:
                    delay = due - time.perf_counter()
                    if delay <= 0:
                        break
                    self.changed.clear()
                    try:
                        await asyncio.wait_for(self.changed.wait(), delay)
                    except asyncio.TimeoutError:
                        break
                    self._prune()
                    if not self.queue:
                        break
                batch = take_batch(self.queue, self.settings.max_batch_size)
                self._prune()
                if not batch:
                    continue
                now = time.perf_counter()
                for job in batch:
                    job.started = now
                self.active_count = len(batch)
                self.metrics.active.set(len(batch))
                self.metrics.inc("batches")
                self.metrics.inc("batch_items", len(batch))
                try:
                    await self.loop.run_in_executor(
                        self.executor, self.worker.run_batch, batch, self._emit
                    )
                    # Drain callbacks scheduled by the worker before checking completion.
                    await asyncio.sleep(0)
                    for job in batch:
                        if not job.future.done():
                            self.fail(job, InferenceError("incomplete_generation"))
                except Exception:
                    log.exception("Model batch failed")
                    for job in batch:
                        self.fail(job, InferenceError("model_error"))
                finally:
                    self.active_count = 0
                    self.metrics.active.set(0)
        except Exception:
            self.accepting = False
            log.exception("Scheduler stopped unexpectedly")
            for job in list(self.jobs.values()):
                self.fail(job, InferenceError("scheduler_failed", 503))

    async def _timeouts(self):
        while True:
            now = time.perf_counter()
            for job in list(self.jobs.values()):
                age = now - job.created
                if age >= self.settings.request_timeout_s:
                    self.fail(job, InferenceError("inference_timeout", 504))
                elif job.started is None and age >= self.settings.queue_timeout_s:
                    self.fail(job, InferenceError("queue_timeout", 504))
            self._prune()
            await asyncio.sleep(0.01)

    async def close(self):
        self.accepting = False
        for job in list(self.queue):
            self.fail(job, InferenceError("server_shutdown", 503))
        self.queue.clear()
        self.changed.set()
        await (
            self.runner
        )  # Current forward/batch completes cooperatively; no unsafe thread kill.
        self.monitor.cancel()
        try:
            await self.monitor
        except asyncio.CancelledError:
            pass
        await self.loop.run_in_executor(self.executor, self.worker.close)
        self.executor.shutdown(wait=True)

    def stats(self):
        return {
            **self.metrics.counts,
            "queue_size": len(self.queue),
            "active_requests": self.active_count,
            "average_batch_size": self.metrics.counts["batch_items"]
            / max(1, self.metrics.counts["batches"]),
            "config": {
                "model": self.cache.model_identity,
                "max_batch_size": self.settings.max_batch_size,
                "batch_wait_ms": self.settings.batch_wait_ms,
                "cache_size": self.settings.cache_size,
            },
        }
