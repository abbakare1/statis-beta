"""Model Provider — abstract interface for LLM adapters.

Agents never depend on Ollama directly. They depend on ModelProvider.
This allows swapping between MiMo, Qwen, Llama, or cloud providers
without changing any agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from loguru import logger


@dataclass
class ModelMetadata:
    """Tracks which model produced a response."""
    provider: str = ""
    model_name: str = ""
    model_version: str = ""
    inference_timestamp: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_validated: bool = False
    validation_error: str | None = None


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""
    content: str = ""
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    success: bool = True
    error: str | None = None
    retries_used: int = 0


class ModelProvider(ABC):
    """Abstract base for all LLM providers.

    All agents MUST use this interface. Never call Ollama directly.
    """

    provider_name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        format_json: bool = False,
        num_ctx: int | None = None,
    ) -> ModelResponse:
        """Send a chat request. Returns standardized ModelResponse."""
        ...

    @abstractmethod
    async def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Chat requesting JSON output. Parses and returns dict."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat — yields content chunks."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check whether the model server is reachable."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return list of locally available model names."""
        ...

    def get_metadata(
        self,
        model: str | None = None,
        prompt_version: str = "1.0",
        schema_version: str = "1.0",
    ) -> ModelMetadata:
        """Create a metadata template for tracking."""
        return ModelMetadata(
            provider=self.provider_name,
            model_name=model or "",
            inference_timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
