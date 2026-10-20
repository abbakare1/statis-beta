"""Research Question Agent — parse and structure the research question using LLM."""

from __future__ import annotations

from app.agents.base import BaseAgent, AgentResult
from app.memory.store import store_memory


SYSTEM_PROMPT = """You are a research methodology expert. Your task is to parse a researcher's
natural-language research question into a structured Research Question Specification.

You must identify:
- Population: Who is being studied?
- Exposure / Intervention: What is being tested or observed?
- Comparator: What is the reference group?
- Outcome: What is being measured?
- Time horizon: Over what period?
- Hypothesis: What is the expected direction of effect?
- Primary objective: What is the main question?
- Secondary objectives: Any additional questions?
- Ambiguities: What is unclear or could be interpreted multiple ways?
- Clarifications needed: What additional information would strengthen the study?

Return a JSON object with exactly these keys:
{
  "population": "...",
  "exposure": "...",
  "comparator": "...",
  "outcome": "...",
  "time_horizon": "...",
  "hypothesis": "...",
  "primary_objective": "...",
  "secondary_objectives": ["..."],
  "ambiguities": ["..."],
  "clarifications_needed": ["..."]
}

If a field is not determinable from the input, use null.
Be precise but do not fabricate information that is not present.
Flag genuine ambiguities — do not assume what the researcher meant."""


class ResearchQuestionAgent(BaseAgent):
    name = "research_question"
    requires_llm = True
    description = "Interpret the research question and produce a structured specification."

    async def _execute(self, context: dict) -> AgentResult:
        raw_question = context.get("research_question", "")
        if not raw_question.strip():
            return AgentResult(
                success=False,
                data={"error": "No research question provided."},
                warnings=["Research question is empty."],
            )

        provider = self.get_provider(context)

        # Check if provider is available
        if not await provider.is_available():
            return AgentResult(
                success=False,
                data={"error": f"Model provider '{provider.provider_name}' is not reachable."},
                warnings=[f"Cannot reach {provider.provider_name} at configured URL."],
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this research question:\n\n{raw_question}"},
        ]

        try:
            result_dict = await provider.chat_json(messages, temperature=0.0)
        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": f"LLM call failed: {e}"},
                warnings=[f"LLM error: {e}"],
            )

        # Store key findings in project memory
        project_id = context.get("project_id")
        if project_id:
            for key in ["population", "exposure", "outcome", "hypothesis", "primary_objective"]:
                value = result_dict.get(key)
                if value:
                    await store_memory(
                        memory_type="project",
                        category="research_question",
                        key=key,
                        value=str(value),
                        source="ai_interpretation",
                        confidence="moderate",
                        project_id=project_id,
                    )

        # Compute confidence based on completeness
        fields_present = sum(1 for v in result_dict.values() if v and v != [] and v != "")
        total_fields = len(result_dict)
        completeness = fields_present / total_fields if total_fields > 0 else 0

        if completeness > 0.8:
            confidence = "high"
        elif completeness > 0.5:
            confidence = "moderate"
        else:
            confidence = "low"

        ambiguities = result_dict.get("ambiguities", [])
        clarifications = result_dict.get("clarifications_needed", [])
        warnings = []
        if ambiguities:
            warnings.extend([f"Ambiguity: {a}" for a in ambiguities])
        if clarifications:
            warnings.extend([f"Clarification needed: {c}" for c in clarifications])

        return AgentResult(
            success=True,
            data=result_dict,
            warnings=warnings,
            confidence=confidence,
        )
