from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_revision: str = "main"
    device: str = "cpu"
    torch_threads: int = Field(4, ge=1)
    max_batch_size: int = Field(8, ge=1, le=64)
    batch_wait_ms: float = Field(20, ge=0, le=5000)
    max_queue_size: int = Field(100, ge=1, le=10000)
    queue_timeout_s: float = Field(30, gt=0)
    request_timeout_s: float = Field(120, gt=0)
    max_input_tokens: int = Field(1024, ge=1)
    max_output_tokens: int = Field(512, ge=1, le=4096)
    cache_size: int = Field(128, ge=0)
    max_prompt_chars: int = Field(16000, ge=1)
    max_body_bytes: int = Field(131072, ge=1024)
    hf_cache_dir: str = ".cache/huggingface"
