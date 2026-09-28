import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from app.api.schemas import GenerateRequest


class InferenceError(Exception):
    def __init__(self, code: str, status: int = 500):
        self.code, self.status = code, status
        super().__init__(code)


@dataclass(frozen=True)
class Output:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass
class Job:
    params: GenerateRequest
    future: asyncio.Future
    events: asyncio.Queue
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.perf_counter)
    started: float | None = None
    first_token: float | None = None
    stop: threading.Event = field(default_factory=threading.Event)
