from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=100000)
    max_new_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, gt=0, le=1)
    stream: bool = False


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    queue_time_ms: float
    ttft_ms: float | None
    finish_reason: str
    cached: bool = False
