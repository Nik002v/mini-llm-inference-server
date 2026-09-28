from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()

        def counter(name, desc):
            return Counter("llm_" + name, desc, registry=self.registry)

        self.requests = counter("requests", "Validated requests submitted")
        self.completed = counter(
            "completed", "Successful requests, including cache hits"
        )
        self.failed = Counter(
            "llm_failed",
            "Failures by bounded reason",
            ["reason"],
            registry=self.registry,
        )
        self.cache_hits = counter("cache_hits", "Deterministic cache hits")
        self.cache_misses = counter(
            "cache_misses", "Eligible deterministic cache misses"
        )
        self.tokens = counter("generated_tokens", "Tokens computed (not cache replay)")
        self.batches = counter("batches", "Model batches executed")
        self.batch_items = counter("batch_items", "Requests dispatched in batches")
        self.queue = Gauge(
            "llm_queue_size", "Live waiting requests", registry=self.registry
        )
        self.active = Gauge(
            "llm_active_requests",
            "Requests in the running batch",
            registry=self.registry,
        )
        for attr in ["latency", "queue_time", "ttft"]:
            setattr(
                self,
                attr,
                Histogram(
                    "llm_" + attr + "_seconds",
                    attr,
                    registry=self.registry,
                    buckets=(
                        0.001,
                        0.005,
                        0.01,
                        0.02,
                        0.05,
                        0.1,
                        0.25,
                        0.5,
                        1,
                        2,
                        5,
                        10,
                        30,
                        60,
                        120,
                    ),
                ),
            )
        self.counts = dict(
            requests=0,
            completed=0,
            failed=0,
            cache_hits=0,
            cache_misses=0,
            generated_tokens=0,
            batches=0,
            batch_items=0,
        )

    def inc(self, name: str, amount: int = 1):
        self.counts[name] += amount
        getattr(self, {"generated_tokens": "tokens"}.get(name, name)).inc(amount)

    def error(self, code: str):
        self.counts["failed"] += 1
        self.failed.labels(reason=code).inc()

    def render(self):
        return generate_latest(self.registry)
