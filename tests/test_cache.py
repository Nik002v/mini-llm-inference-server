from app.cache.lru import LRUCache
from app.api.schemas import GenerateRequest
from app.inference.request import Output


def test_lru_eviction_and_stream_key():
    cache = LRUCache(2, "model@sha")
    a = GenerateRequest(prompt="a", temperature=0)
    b = a.model_copy(update={"prompt": "b"})
    c = a.model_copy(update={"prompt": "c"})
    out = Output("answer", 1, 1, "eos")
    cache.put(cache.key(a), out)
    cache.put(cache.key(b), out)
    assert cache.get(cache.key(a)) == out
    cache.put(cache.key(c), out)
    assert cache.get(cache.key(b)) is None
    assert cache.key(a) == cache.key(a.model_copy(update={"stream": True}))
    assert cache.key(a) != LRUCache(2, "model@other").key(a)
    assert cache.key(a.model_copy(update={"max_new_tokens": 1})) != cache.key(a)


def test_stochastic_and_disabled_cache():
    assert LRUCache(2, "x").key(GenerateRequest(prompt="a", temperature=0.1)) is None
    assert LRUCache(0, "x").key(GenerateRequest(prompt="a", temperature=0)) is None
