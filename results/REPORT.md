# Measured local results

These are real HTTP requests to the real Hugging Face model, not the fake test worker.

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Device: CPU, float32, 4 PyTorch threads, 8 logical CPU cores.
- Host: Windows-10-10.0.19045-SP0; AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD.
- Python 3.12.7, PyTorch 2.5.1, Transformers 4.57.6.
- Workload: 16 requests per scenario, maximum 12 output tokens, 24-word base prompt. Unique requests add an example index; cache experiments use one repeated prompt. One warm-up precedes measurement. Model loading and download are excluded.
- 128 successful measured requests, 0 failures across 8 scenarios.
- Command: `python -m benchmarks.sweep --requests 16 --output-tokens 12 --output results/cpu_smollm2`.

## Observations

At concurrency 8, disabling batching produced **0.41 req/s** and **4.87 computed tokens/s**. A maximum batch of 8 with a 50 ms window produced **1.21 req/s** and **14.54 computed tokens/s**, or **2.98x** the request throughput in this run. Average dispatch batch size grew from 1.0 to 8.0.

Queueing dominates the unbatched concurrent case: client p95 was **20.67 s**, compared with **7.23 s** for the 50 ms batch window. Batching can therefore reduce end-to-end latency under load even though it deliberately waits before dispatch. At low load, the same window adds avoidable waiting; the mechanism does not promise better latency for every arrival pattern.

The 0 ms and 20 ms windows achieved 0.94 and 0.89 req/s, both with average batch size 4.0. In this short closed-loop workload, 50 ms collected full batches, while 20 ms did not. That is an observation about this arrival pattern, not evidence that 50 ms is universally optimal. Thermal state, OS scheduling, HTTP timing and the small sample make fine-grained rankings noisy. Do not infer a monotonic window/latency curve from these eight runs.

Streaming client TTFT p50 was **4.42 s**, versus total latency p50 **7.91 s**. TTFT includes queueing and padded prefill and measures the first token event; displayed text can lag while a word is buffered.

After warming the repeated deterministic prompt, the cache-hit ratio was **100%**, with **171.60 req/s** and p50 **38.12 ms**. Computed model tokens/s was correctly **0**: cache replay is not new inference. This is steady-state cache latency, not a mixed hit/miss workload.

## Full measured table

| Experiment | Clients | Batch / wait ms | Avg batch | Req/s | Computed tok/s | p50 ms | p95 ms | p99 ms | TTFT p50 ms | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| batch_c8_w0 | 8 | 8 / 0.0 | 4.00 | 0.94 | 11.32 | 8445.58 | 9037.00 | 10323.74 | — | 0 |
| batch_c8_w20 | 8 | 8 / 20.0 | 4.00 | 0.89 | 10.66 | 8318.15 | 11681.28 | 11681.69 | — | 0 |
| batch_c8_w50 | 8 | 8 / 50.0 | 8.00 | 1.21 | 14.54 | 6594.46 | 7229.18 | 7232.00 | — | 0 |
| cache_disabled | 8 | 8 / 20.0 | 4.00 | 0.97 | 11.69 | 8199.83 | 10140.59 | 10141.81 | — | 0 |
| cache_enabled | 8 | 8 / 20.0 | 0.00 | 171.60 | 0.00 | 38.12 | 64.08 | 65.19 | — | 0 |
| no_batch_c1 | 1 | 1 / 0.0 | 1.00 | 0.37 | 4.43 | 2482.18 | 3620.49 | 5279.00 | — | 0 |
| no_batch_c8 | 8 | 1 / 0.0 | 1.00 | 0.41 | 4.87 | 18928.63 | 20666.65 | 20667.13 | — | 0 |
| stream_c8 | 8 | 8 / 20.0 | 4.00 | 0.99 | 11.89 | 7909.72 | 8537.34 | 9278.37 | 4416.90 | 0 |


## Artifacts and limitations

- [CSV](cpu_smollm2/benchmark_results.csv) and [table](cpu_smollm2/SUMMARY.md); each scenario's JSON contains per-request timings, IDs, outputs and failures.
- [Final local dependency environment](environment.json). The prepared local venv reuses the machine's installed CPU PyTorch; fresh installs use requirements.txt. Prometheus client and HTTP transport dependencies were normalized during implementation; these short exploratory runs are not a fully frozen scientific benchmark campaign.
- CPU only; no GPU measurements. 16 samples per scenario are insufficient for stable p99 estimates. Percentiles use linear interpolation and successful responses only; failures are reported separately.
- A closed-loop test issues the next request after an earlier one finishes. It does not model an independent Poisson arrival process or sustained overload.
- Full batch-size and concurrency axis sweeps are implemented with `--full`, but this saved run covers batch sizes 1/8, concurrency 1/8, and windows 0/20/50 ms. Unit tests separately exercise 1/4/8/16/32 clients.
- Raw outputs demonstrate transport and scheduling, not model answer quality. The small model receives literal completion prompts, not a chat template.
