"""Opt-in real-model correctness tests; downloads the configured model if necessary."""

import asyncio
import os
import pytest
from app.api.schemas import GenerateRequest
from app.config import Settings
from app.inference.request import Job
from app.inference.worker import ModelWorker

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 for real HF model tests",
)


async def test_batched_greedy_matches_hf_generate_and_stream_text():
    settings = Settings()
    worker = ModelWorker(settings)
    worker.load()
    try:
        loop = asyncio.get_running_loop()
        prompts = [
            "The capital of France is",
            "A virtual memory page is a fixed size block of",
        ]
        jobs = [
            Job(
                GenerateRequest(prompt=p, max_new_tokens=3, temperature=0),
                loop.create_future(),
                asyncio.Queue(),
            )
            for p in prompts
        ]
        events = {j.id: [] for j in jobs}
        worker.run_batch(jobs, lambda j, k, p: events[j.id].append((k, p)))
        for job in jobs:
            kinds = events[job.id]
            result = [p for k, p in kinds if k == "done"][0]
            tokens = [p for k, p in kinds if k == "token"]
            assert "".join(t["text"] for t in tokens) == result.text
            encoded = worker.tokenizer(job.params.prompt, return_tensors="pt").to(
                worker.device
            )
            with worker.torch.inference_mode():
                baseline = worker.model.generate(
                    **encoded,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=worker.tokenizer.pad_token_id,
                )
            ids = baseline[0, encoded["input_ids"].shape[1] :]
            expected = worker.tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert result.text == expected
            assert result.output_tokens == len(ids)
    finally:
        worker.close()
