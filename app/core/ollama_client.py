"""Ollama API client — deterministic interface to the local LLM."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from config.settings import get_settings


class OllamaClient:
    """Thin async wrapper around the Ollama REST API."""

    def __init__(self):
        s = get_settings()
        self.base_url = s.OLLAMA_BASE_URL
        self.model = s.OLLAMA_MODEL
        self.timeout = s.OLLAMA_TIMEOUT
        self.num_ctx = s.OLLAMA_NUM_CTX

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
    ) -> str:
        """Send a chat request and return the assistant content."""
        payload: dict[str, Any] = {
            "model": model or self.model,
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
        return data.get("message", {}).get("content", "")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Chat requesting JSON output; parses and returns dict."""
        raw = await self.chat(
            messages, model=model, temperature=temperature, format_json=True
        )
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat — yields content chunks."""
        payload = {
            "model": model or self.model,
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


_client: OllamaClient | None = None


def get_ollama() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
