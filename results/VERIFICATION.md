# Verification

- `RUN_MODEL_TESTS=1 python -m pytest -q`: **28 passed in 48.64 seconds**.
- `ruff check app benchmarks tests`: passes.
- `ruff format --check app benchmarks tests`: passes.
- `python -m pip check`: no broken requirements in the prepared local environment.
- `docker --config .cache/docker compose config --quiet`: valid.
- Real CPU model: 8 benchmark scenarios / 128 successful measured HTTP requests / no failures.
- `docker --config .cache/docker build -t mini-llm-server:local .`: successful (image ID `1cc5c39f6a98`, about 1.58 GB).
- Docker runtime smoke test with the real offline SmolLM2 model: healthy container; normal generation, four concurrent requests, cache reuse, SSE reconstruction, validation, metrics, and graceful stop all passed. Seven requests completed with zero failures; see `docker_smoke.json`.

See REPORT.md and cpu_smollm2/*.json for performance evidence, and docker_smoke.json for the container runtime result. Default tests skip the one real-model test unless RUN_MODEL_TESTS=1 is set.
