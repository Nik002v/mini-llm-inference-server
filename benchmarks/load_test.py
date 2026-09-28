"""Real HTTP load test. Examples: python -m benchmarks.load_test --help."""

import argparse
import asyncio
import json
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import httpx


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


async def run(
    url="http://127.0.0.1:8000",
    concurrency=8,
    requests=32,
    output_tokens=16,
    prompt_words=24,
    stream=False,
    repeat=False,
    warmup=1,
):
    limits = httpx.Limits(
        max_connections=max(2, concurrency),
        max_keepalive_connections=max(2, concurrency),
    )
    async with httpx.AsyncClient(
        base_url=url, timeout=300, limits=limits, trust_env=False
    ) as client:
        prompt = (
            "Explain how a computer manages memory. " * ((prompt_words + 6) // 7)
        ).split()
        prompt = " ".join(prompt[:prompt_words])

        def payload(i):
            return {
                "prompt": prompt if repeat else f"{prompt}\nExample {i}:",
                "max_new_tokens": output_tokens,
                "temperature": 0,
                "top_p": 1,
                "stream": stream,
            }

        for i in range(warmup):
            response = await client.post(
                "/generate", json={**payload(-1), "stream": False}
            )
            response.raise_for_status()
        before = (await client.get("/stats")).json()
        records = []
        work = asyncio.Queue()
        for i in range(requests):
            work.put_nowait(i)

        async def consumer():
            while True:
                try:
                    i = work.get_nowait()
                except asyncio.QueueEmpty:
                    return
                start = time.perf_counter()
                first = None
                data = None
                error = None
                try:
                    if stream:
                        async with client.stream(
                            "POST", "/generate", json=payload(i)
                        ) as response:
                            response.raise_for_status()
                            event = ""
                            async for line in response.aiter_lines():
                                if line.startswith("event: "):
                                    event = line[7:]
                                elif line.startswith("data: "):
                                    item = json.loads(line[6:])
                                    if event == "token" and first is None:
                                        first = (time.perf_counter() - start) * 1000
                                    elif event == "done":
                                        data = item
                                    elif event == "error":
                                        error = item["error"]
                            if data is None and error is None:
                                error = "missing_terminal_event"
                    else:
                        response = await client.post("/generate", json=payload(i))
                        if response.status_code != 200:
                            error = response.json().get(
                                "error", str(response.status_code)
                            )
                        else:
                            data = response.json()
                except Exception as exc:
                    error = type(exc).__name__
                records.append(
                    {
                        "index": i,
                        "latency_ms": (time.perf_counter() - start) * 1000,
                        "client_ttft_ms": first,
                        "error": error,
                        "result": data,
                    }
                )

        start = time.perf_counter()
        await asyncio.gather(*(consumer() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
        after = (await client.get("/stats")).json()
    ok = [r for r in records if r["error"] is None and r["result"] is not None]
    lat = [r["latency_ms"] for r in ok]
    ttft = [r["client_ttft_ms"] for r in ok if r["client_ttft_ms"] is not None]
    batches = after["batches"] - before["batches"]
    batch_items = after["batch_items"] - before["batch_items"]
    hits = after["cache_hits"] - before["cache_hits"]
    misses = after["cache_misses"] - before["cache_misses"]
    delta_tokens = after["generated_tokens"] - before["generated_tokens"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "config": after["config"],
        "concurrency": concurrency,
        "requests": requests,
        "stream": stream,
        "repeat": repeat,
        "prompt_words_requested": prompt_words,
        "max_output_tokens": output_tokens,
        "successful": len(ok),
        "failed": len(records) - len(ok),
        "errors": dict(Counter(r["error"] for r in records if r["error"])),
        "elapsed_s": elapsed,
        "requests_per_s": len(ok) / elapsed,
        "generated_tokens_per_s": delta_tokens / elapsed,
        "delivered_tokens_per_s": sum(r["result"]["output_tokens"] for r in ok)
        / elapsed,
        "latency_p50_ms": percentile(lat, 0.5),
        "latency_p95_ms": percentile(lat, 0.95),
        "latency_p99_ms": percentile(lat, 0.99),
        "ttft_p50_ms": percentile(ttft, 0.5),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "queue_mean_ms": statistics.mean(r["result"]["queue_time_ms"] for r in ok)
        if ok
        else None,
        "average_batch_size": batch_items / batches if batches else 0,
        "cache_hit_ratio": hits / (hits + misses) if hits + misses else None,
        "records": sorted(records, key=lambda r: r["index"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=24)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="results/load_test.json")
    args = parser.parse_args()
    if (
        min(args.concurrency, args.requests, args.output_tokens, args.prompt_words) < 1
        or args.warmup < 0
    ):
        parser.error("Counts must be positive; warmup must be nonnegative")
    options = vars(args).copy()
    path = Path(options.pop("output"))
    result = asyncio.run(run(**options))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
