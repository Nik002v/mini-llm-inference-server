"""Only this component touches the tokenizer/model; executed on one dedicated thread."""

import logging
from typing import Callable
from app.config import Settings
from app.inference.request import Job, Output, InferenceError

log = logging.getLogger(__name__)
Emit = Callable[[Job, str, object], None]


class ModelWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(self.settings.torch_threads)
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        kwargs = dict(
            revision=self.settings.model_revision,
            cache_dir=self.settings.hf_cache_dir,
            trust_remote_code=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_name, **kwargs
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError("The tokenizer must provide an EOS or PAD token")
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                dtype=torch.float32 if device == "cpu" else torch.float16,
                attn_implementation="eager",
                use_safetensors=True,
                **kwargs,
            )
            .to(device)
            .eval()
        )
        self.context_size = getattr(self.model.config, "max_position_embeddings", 2048)
        self.identity = (
            self.settings.model_name
            + "@"
            + str(
                getattr(self.model.config, "_commit_hash", None)
                or self.settings.model_revision
            )
        )
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.loaded = True

    def close(self):
        self.loaded = False
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "torch") and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def run_batch(self, jobs: list[Job], emit: Emit):
        torch = self.torch
        live, encodings = [], []
        for job in jobs:
            if job.stop.is_set():
                continue
            ids = self.tokenizer.encode(job.params.prompt, add_special_tokens=True)
            if (
                not ids
                or len(ids) > self.settings.max_input_tokens
                or len(ids) + job.params.max_new_tokens > self.context_size
            ):
                emit(job, "error", InferenceError("context_length_exceeded", 422))
                continue
            live.append(job)
            encodings.append(ids)
        if not live:
            return
        # Budget padded context conservatively, to avoid batch-dependent position overflows.
        width = max(map(len, encodings))
        allowed = [
            i
            for i, j in enumerate(live)
            if width + j.params.max_new_tokens <= self.context_size
        ]
        # Each request was individually valid. If padding would overflow, execute it alone.
        # This rare path preserves validity; scheduler metrics count dispatch batches, not forwards.
        if len(allowed) != len(live):
            for job in live:
                self.run_batch([job], emit)
            return
        batch = self.tokenizer.pad(
            {"input_ids": encodings}, padding=True, return_tensors="pt"
        )
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        generated = [[] for _ in live]
        emitted = ["" for _ in live]
        done = [False] * len(live)
        past = None
        with torch.inference_mode():
            for _ in range(max(j.params.max_new_tokens for j in live)):
                if all(d or j.stop.is_set() for d, j in zip(done, live)):
                    break
                positions = mask.long().cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 0)
                if past is not None:
                    positions = positions[:, -1:]
                result = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=past,
                    use_cache=True,
                )
                past = result.past_key_values
                next_ids = []
                for i, job in enumerate(live):
                    if done[i] or job.stop.is_set():
                        next_ids.append(self.tokenizer.pad_token_id)
                        continue
                    logits = result.logits[i, -1, :].float()
                    params = job.params
                    if params.temperature == 0:
                        token = logits.argmax().item()
                    else:
                        logits = logits / params.temperature
                        sorted_logits, indices = torch.sort(logits, descending=True)
                        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
                        remove = cumulative > params.top_p
                        remove[1:] = remove[:-1].clone()
                        remove[0] = False
                        sorted_logits[remove] = -float("inf")
                        sample = torch.multinomial(
                            torch.softmax(sorted_logits, dim=-1), 1
                        )
                        token = indices[sample].item()
                    next_ids.append(token)
                    generated[i].append(token)
                    text = self.tokenizer.decode(
                        generated[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    end = (
                        token in self.eos or len(generated[i]) >= params.max_new_tokens
                    )
                    # Flush complete words, retaining partial UTF-8 / unfinished words.
                    stable = text if end else text[: text.rfind(" ") + 1]
                    if not stable.startswith(emitted[i]):
                        raise RuntimeError("Tokenizer changed previously streamed text")
                    delta = stable[len(emitted[i]) :]
                    emitted[i] = stable
                    emit(job, "token", {"text": delta, "token_id": token})
                    if end:
                        done[i] = True
                        emit(
                            job,
                            "done",
                            Output(
                                text,
                                len(encodings[i]),
                                len(generated[i]),
                                "eos" if token in self.eos else "length",
                            ),
                        )
                input_ids = torch.tensor(next_ids, device=self.device).unsqueeze(1)
                mask = torch.cat(
                    [
                        mask,
                        torch.ones(
                            (len(live), 1), device=self.device, dtype=mask.dtype
                        ),
                    ],
                    dim=1,
                )
