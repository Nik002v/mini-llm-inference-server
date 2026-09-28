"""Restart a real HTTP server for each configuration; collect reproducible measured results."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import httpx
from benchmarks.load_test import run
from benchmarks.analyze_results import analyze


def scenarios(full=False):
    cases = [
        ("no_batch_c1", 1, 0, 1, 0, False, False),
        ("no_batch_c8", 1, 0, 8, 0, False, False),
        ("batch_c8_w0", 8, 0, 8, 0, False, False),
        ("batch_c8_w20", 8, 20, 8, 0, False, False),
        ("batch_c8_w50", 8, 50, 8, 0, False, False),
        ("stream_c8", 8, 20, 8, 0, True, False),
        ("cache_disabled", 8, 20, 8, 0, False, True),
        ("cache_enabled", 8, 20, 8, 128, False, True),
    ]
    if full:
        cases += [(f"window_{w}", 8, w, 8, 0, True, False) for w in [5, 10]]
        cases += [(f"batch_{b}", b, 20, 16, 0, True, False) for b in [1, 2, 4, 8, 16]]
        cases += [(f"clients_{c}", 8, 20, c, 0, True, False) for c in [1, 4, 8, 16, 32]]
    return cases


async def sweep(args):
    root = Path(__file__).resolve().parents[1]
    dest = (root / args.output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    # Never accidentally benchmark a pre-existing service on the selected port.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    for name, size, window, clients, cache, stream, repeat in scenarios(args.full):
        env = {
            **os.environ,
            "MAX_BATCH_SIZE": str(size),
            "BATCH_WAIT_MS": str(window),
            "CACHE_SIZE": str(cache),
            "REQUEST_TIMEOUT_S": "300",
            "QUEUE_TIMEOUT_S": "300",
            "HF_HUB_DISABLE_XET": "1",
        }
        if args.model:
            env["MODEL_NAME"] = args.model
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
        # CREATE_NO_WINDOW avoids opening a visible console for the benchmark helper.
        # On Windows a console-less helper cannot receive CTRL_BREAK; use an ASGI lifespan
        # control runner with a stop file instead (see server_runner).
        stopfile = dest / f"{name}.stop"
        if stopfile.exists():
            stopfile.unlink()
        logpath = dest / f"{name}.log"
        with logpath.open("w", encoding="utf-8") as logfile:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.server_runner",
                    str(args.port),
                    str(stopfile),
                ],
                cwd=root,
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            try:
                ready = False
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    for _ in range(900):
                        if process.poll() is not None:
                            raise RuntimeError(f"Server startup failed; see {logpath}")
                        try:
                            ready = (
                                await client.get(url + "/health")
                            ).status_code == 200
                        except httpx.HTTPError:
                            pass
                        if ready:
                            break
                        await asyncio.sleep(1)
                if not ready:
                    raise RuntimeError("Server startup timed out")
                result = await run(
                    url,
                    clients,
                    args.requests,
                    args.output_tokens,
                    args.prompt_words,
                    stream,
                    repeat,
                )
                (dest / f"{name}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                print(
                    name,
                    "req/s",
                    round(result["requests_per_s"], 2),
                    "failed",
                    result["failed"],
                    flush=True,
                )
                if result["failed"]:
                    raise RuntimeError(f"Benchmark {name} has failures")
            finally:
                stopfile.write_text("stop", encoding="utf-8")
                try:
                    await asyncio.to_thread(process.wait, timeout=180)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        "Graceful benchmark server shutdown exceeded 180 seconds"
                    )
                if stopfile.exists():
                    stopfile.unlink()
    analyze(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=24)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if min(args.requests, args.output_tokens, args.prompt_words) < 1:
        parser.error("Counts must be positive")
    asyncio.run(sweep(args))
