"""Ollama Adapter — concrete ModelProvider for Ollama local inference.

Supports MiMo, Qwen, Llama, Gemma, and any other Ollama-hosted model.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx
from loguru import logger

from app.core.model_provider import ModelProvider, ModelResponse, ModelMetadata
from config.settings import get_settings


class OllamaAdapter(ModelProvider):
    """Ollama REST API adapter implementing ModelProvider."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
        max_retries: int = 2,
    ):
        s = get_settings()
        self.base_url = base_url or s.OLLAMA_BASE_URL
        self.default_model = default_model or s.OLLAMA_MODEL
        self.timeout = timeout or s.OLLAMA_TIMEOUT
        self.num_ctx = num_ctx or s.OLLAMA_NUM_CTX
        self.max_retries = max_retries

    async def _post(self, endpoint: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}{endpoint}", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        format_json: bool = False,
        num_ctx: int | None = None,
    ) -> ModelResponse:
        """Send a chat request with retry logic and metadata tracking."""
        model = model or self.default_model
        metadata = self.get_metadata(model=model)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": num_ctx or self.num_ctx,
                    },
                }
                if format_json:
                    payload["format"] = "json"

                data = await self._post("/api/chat", payload)
                content = data.get("message", {}).get("content", "")

                # Extract token counts if available
                metadata.prompt_tokens = data.get("prompt_eval_count", 0)
                metadata.completion_tokens = data.get("eval_count", 0)
                metadata.total_tokens = metadata.prompt_tokens + metadata.completion_tokens
                metadata.response_validated = True

                return ModelResponse(
                    content=content,
                    metadata=metadata,
                    success=True,
                    retries_used=attempt,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Ollama chat attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))

        metadata.response_validated = False
        metadata.validation_error = last_error
        return ModelResponse(
            content="",
            metadata=metadata,
            success=False,
            error=last_error,
            retries_used=self.max_retries,
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Chat requesting JSON output; parses and returns dict with validation."""
        response = await self.chat(
            messages, model=model, temperature=temperature, format_json=True
        )

        if not response.success:
            raise RuntimeError(f"Model call failed: {response.error}")

        cleaned = response.content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from model response: {e}")
            logger.debug(f"Raw response (first 500 chars): {cleaned[:500]}")
            raise ValueError(f"Model returned invalid JSON: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat — yields content chunks."""
        model = model or self.default_model
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta

    async def is_available(self) -> bool:
        """Check whether Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return list of locally available model names."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_provider: ModelProvider | None = None


def get_model_provider(**kwargs) -> ModelProvider:
    """Return the configured model provider singleton.

    Accepts optional kwargs to override defaults on first call.
    """
    global _provider
    if _provider is None:
        _provider = OllamaAdapter(**kwargs)
    return _provider


def reset_model_provider() -> None:
    """Reset the singleton (for testing)."""
    global _provider
    _provider = None
