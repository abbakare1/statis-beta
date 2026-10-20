"""Agent base class — common interface for all STATIS agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from app.memory.audit import log_audit


class AgentResult(BaseModel):
    """Standard wrapper for agent outputs."""
    success: bool
    data: dict[str, Any] = {}
    warnings: list[str] = []
    confidence: str = "insufficient"
    agent_name: str = ""
    execution_time_ms: int = 0


class BaseAgent(ABC):
    """Abstract base for all STATIS agents.

    Agents that require an LLM receive a ModelProvider via context['model_provider']
    or via the model_provider attribute set by the orchestrator.
    """

    name: str = "base_agent"
    requires_llm: bool = False
    description: str = ""

    def __init__(self, model_provider=None):
        """Optional model provider injection for LLM agents."""
        self._model_provider = model_provider

    def get_provider(self, context: dict[str, Any] = None):
        """Get the model provider, checking context first, then instance, then singleton."""
        if context and "model_provider" in context:
            return context["model_provider"]
        if self._model_provider is not None:
            return self._model_provider
        from app.core.ollama_adapter import get_model_provider
        return get_model_provider()

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """Execute the agent and return structured result with audit trail."""
        start = datetime.now(timezone.utc)
        try:
            result = await self._execute(context)
            elapsed = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result.execution_time_ms = elapsed
            result.agent_name = self.name

            await log_audit(
                action=f"{self.name}_completed",
                agent=self.name,
                decision="success" if result.success else "partial_failure",
                input_summary=self._summarize_input(context),
                output_summary=self._summarize_output(result),
                confidence=result.confidence,
                warnings=result.warnings if result.warnings else None,
                project_id=context.get("project_id"),
                session_id=context.get("session_id"),
            )
            return result

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            error_result = AgentResult(
                success=False,
                data={"error": str(exc)},
                warnings=[f"Agent {self.name} failed: {exc}"],
                confidence="insufficient",
                agent_name=self.name,
                execution_time_ms=elapsed,
            )
            await log_audit(
                action=f"{self.name}_failed",
                agent=self.name,
                decision="failure",
                input_summary=self._summarize_input(context),
                output_summary=str(exc),
                confidence="insufficient",
                warnings=[str(exc)],
                project_id=context.get("project_id"),
                session_id=context.get("session_id"),
            )
            return error_result

    @abstractmethod
    async def _execute(self, context: dict[str, Any]) -> AgentResult:
        """Agent-specific logic. Subclasses implement this."""
        ...

    def _summarize_input(self, context: dict[str, Any]) -> str:
        """Produce a truncated string summary of the input context."""
        keys = list(context.keys())[:10]
        parts = []
        for k in keys:
            v = context[k]
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            parts.append(f"{k}={s}")
        return "; ".join(parts)

    def _summarize_output(self, result: AgentResult) -> str:
        """Produce a truncated string summary of the output."""
        keys = list(result.data.keys())[:10]
        parts = []
        for k in keys:
            v = result.data[k]
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            parts.append(f"{k}={s}")
        return "; ".join(parts) if parts else "(no output)"
