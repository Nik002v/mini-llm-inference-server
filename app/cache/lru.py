import hashlib
import json
from collections import OrderedDict
from app.api.schemas import GenerateRequest
from app.inference.request import Output


class LRUCache:
    """Event-loop owned. No entries for stochastic generation."""

    def __init__(self, capacity: int, model_identity: str):
        self.capacity = capacity
        self.model_identity = model_identity
        self.items: OrderedDict[str, Output] = OrderedDict()

    def key(self, params: GenerateRequest) -> str | None:
        if not self.capacity or params.temperature != 0:
            return None
        data = params.model_dump(exclude={"stream"})
        data["model"] = self.model_identity
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get(self, key: str | None) -> Output | None:
        if key is None or key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str | None, output: Output):
        if key is None or not self.capacity:
            return
        self.items[key] = output
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
