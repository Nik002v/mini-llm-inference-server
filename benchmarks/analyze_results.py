import argparse
import csv
import json
from pathlib import Path


def analyze(directory):
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "requests_per_s" not in data:
            continue
        rows.append(
            {
                "experiment": path.stem,
                **data["config"],
                **{
                    k: v
                    for k, v in data.items()
                    if k not in ("records", "config", "errors")
                    and not isinstance(v, (dict, list))
                },
            }
        )
    if not rows:
        raise SystemExit("No benchmark JSON files found")
    with (directory / "benchmark_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = "| Experiment | Clients | Batch / wait ms | Avg batch | Req/s | Computed tok/s | p50 ms | p95 ms | p99 ms | TTFT p50 ms | Errors |"
    lines = [header, "|" + "---|" * 11]

    def num(n):
        return "—" if n is None else f"{n:.2f}"

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["experiment"],
                    str(r["concurrency"]),
                    f"{r['max_batch_size']} / {r['batch_wait_ms']}",
                    num(r["average_batch_size"]),
                    num(r["requests_per_s"]),
                    num(r["generated_tokens_per_s"]),
                    num(r["latency_p50_ms"]),
                    num(r["latency_p95_ms"]),
                    num(r["latency_p99_ms"]),
                    num(r["ttft_p50_ms"]),
                    str(r["failed"]),
                ]
            )
            + " |"
        )
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results")
    analyze(parser.parse_args().directory)
