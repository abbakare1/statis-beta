"""Tests for the Model Provider adapter pattern."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from app.core.model_provider import ModelProvider, ModelResponse, ModelMetadata
from app.core.ollama_adapter import OllamaAdapter, get_model_provider, reset_model_provider


class TestModelMetadata:
    def test_default_metadata(self):
        m = ModelMetadata()
        assert m.provider == ""
        assert m.model_name == ""
        assert m.response_validated is False
        assert isinstance(m.inference_timestamp, str)

    def test_metadata_with_values(self):
        m = ModelMetadata(
            provider="ollama",
            model_name="qwen2.5-coder:7b",
            prompt_version="2.0",
            total_tokens=1500,
            response_validated=True,
        )
        assert m.provider == "ollama"
        assert m.total_tokens == 1500
        assert m.response_validated is True


class TestModelResponse:
    def test_success_response(self):
        r = ModelResponse(content="hello", success=True)
        assert r.content == "hello"
        assert r.success is True
        assert r.error is None
        assert r.retries_used == 0

    def test_failure_response(self):
        r = ModelResponse(content="", success=False, error="timeout", retries_used=2)
        assert r.success is False
        assert r.error == "timeout"
        assert r.retries_used == 2

    def test_default_metadata(self):
        r = ModelResponse()
        assert isinstance(r.metadata, ModelMetadata)


class TestOllamaAdapter:
    def test_init_defaults(self):
        adapter = OllamaAdapter()
        assert adapter.provider_name == "ollama"
        assert adapter.base_url == "http://localhost:11434"
        assert adapter.default_model == "qwen2.5-coder:7b"
        assert adapter.max_retries == 2

    def test_init_custom(self):
        adapter = OllamaAdapter(
            base_url="http://custom:9999",
            default_model="mimo-v2-flash",
            timeout=30,
            num_ctx=4096,
            max_retries=5,
        )
        assert adapter.base_url == "http://custom:9999"
        assert adapter.default_model == "mimo-v2-flash"
        assert adapter.timeout == 30
        assert adapter.num_ctx == 4096
        assert adapter.max_retries == 5

    def test_get_metadata(self):
        adapter = OllamaAdapter()
        meta = adapter.get_metadata(model="test-model", prompt_version="3.0")
        assert meta.provider == "ollama"
        assert meta.model_name == "test-model"
        assert meta.prompt_version == "3.0"
        assert meta.schema_version == "1.0"
        assert meta.inference_timestamp != ""

    @pytest.mark.asyncio
    async def test_is_available_false_when_no_server(self):
        adapter = OllamaAdapter(base_url="http://localhost:19999", timeout=2)
        result = await adapter.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_chat_failure_returns_error(self):
        adapter = OllamaAdapter(base_url="http://localhost:19999", timeout=2, max_retries=1)
        result = await adapter.chat([{"role": "user", "content": "test"}])
        assert result.success is False
        assert result.error is not None
        assert result.retries_used == 1


class TestModelProviderSingleton:
    def test_get_provider_returns_ollama(self):
        reset_model_provider()
        p = get_model_provider()
        assert isinstance(p, OllamaAdapter)
        assert p.provider_name == "ollama"

    def test_singleton_returns_same_instance(self):
        reset_model_provider()
        p1 = get_model_provider()
        p2 = get_model_provider()
        assert p1 is p2

    def test_reset_creates_new_instance(self):
        reset_model_provider()
        p1 = get_model_provider()
        reset_model_provider()
        p2 = get_model_provider()
        assert p1 is not p2


class TestAgentProviderInjection:
    """Test that agents can receive a provider via constructor or context."""

    def test_agent_with_constructor_provider(self):
        from app.agents.data_ingestion import DataIngestionAgent
        mock_provider = OllamaAdapter(base_url="http://mock:9999")
        agent = DataIngestionAgent(model_provider=mock_provider)
        # Non-LLM agents still work, but have the provider available
        assert agent._model_provider is mock_provider

    def test_agent_get_provider_from_context(self):
        from app.agents.data_ingestion import DataIngestionAgent
        mock_provider = OllamaAdapter(base_url="http://mock:9999")
        agent = DataIngestionAgent()
        ctx = {"model_provider": mock_provider}
        p = agent.get_provider(ctx)
        assert p is mock_provider

    def test_agent_get_provider_from_instance(self):
        from app.agents.data_ingestion import DataIngestionAgent
        mock_provider = OllamaAdapter(base_url="http://mock:9999")
        agent = DataIngestionAgent(model_provider=mock_provider)
        p = agent.get_provider()
        assert p is mock_provider

    def test_agent_get_provider_fallback_to_singleton(self):
        from app.agents.data_ingestion import DataIngestionAgent
        reset_model_provider()
        agent = DataIngestionAgent()
        p = agent.get_provider()
        assert isinstance(p, OllamaAdapter)
