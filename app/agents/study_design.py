"""Study Design Agent — classify and evaluate the study design using LLM.

Phase 2D.2: Preserves UNKNOWN state when no design info is provided.
Does NOT fabricate "observational" when design is unknown.
"""

from __future__ import annotations

from app.agents.base import BaseAgent, AgentResult
from app.memory.store import store_memory
from app.statistics.study_design_state import classify_design_state, DesignState


SYSTEM_PROMPT = """You are an epidemiologist and study design expert. Given a research question
and any study design information provided by the researcher, determine the most likely study design
and evaluate its appropriateness.

Supported study designs:
- Cross-sectional
- Case-control
- Cohort (retrospective, prospective)
- Randomized controlled trial (RCT)
- Diagnostic accuracy study
- Prognostic study
- Survival analysis study
- Repeated-measures / longitudinal
- Ecological study
- Survey study
- Experimental laboratory study

You must:
1. Identify the most likely study design from the description
2. Describe the design in plain language
3. Evaluate whether the design is appropriate for the stated research question
4. Flag any methodological concerns
5. Provide recommendations

Return a JSON object:
{
  "design_type": "cohort",
  "design_description": "This is a retrospective cohort study that...",
  "appropriate_for_question": true,
  "concerns": ["..."],
  "recommendations": ["..."],
  "confidence": "high"
}

confidence must be one of: "high", "moderate", "low", "insufficient"
Do NOT change the user's stated study design. If they say it is an RCT, evaluate it as an RCT.
Only suggest an alternative if the stated design is clearly misapplied.

IMPORTANT: If no study design information is provided by the researcher, set design_type to "unknown"
and confidence to "insufficient". Do NOT infer or fabricate a study design."""


class StudyDesignAgent(BaseAgent):
    name = "study_design"
    requires_llm = True
    description = "Classify and evaluate the study design."

    async def _execute(self, context: dict) -> AgentResult:
        research_question = context.get("research_question", "")
        study_design_description = context.get("study_design", "")

        # If no design info provided, return UNKNOWN without calling LLM
        if not study_design_description or not study_design_description.strip():
            return AgentResult(
                success=True,
                data={
                    "design_type": "unknown",
                    "design_description": "No study design information provided by researcher.",
                    "appropriate_for_question": None,
                    "concerns": [],
                    "recommendations": ["Researcher should specify the study design."],
                    "confidence": "insufficient",
                    "design_state": DesignState.UNKNOWN.value,
                },
                warnings=["Study design is UNKNOWN. Downstream analyses should preserve this uncertainty."],
                confidence="insufficient",
            )

        provider = self.get_provider(context)

        if not await provider.is_available():
            # LLM unavailable — preserve UNKNOWN state
            return AgentResult(
                success=True,
                data={
                    "design_type": "unknown",
                    "design_description": "LLM unavailable to classify study design.",
                    "appropriate_for_question": None,
                    "concerns": [],
                    "recommendations": [],
                    "confidence": "insufficient",
                    "design_state": DesignState.UNKNOWN.value,
                },
                warnings=[f"Cannot reach {provider.provider_name}. Design state set to UNKNOWN."],
                confidence="insufficient",
            )

        user_prompt = f"""Research Question: {research_question}

Researcher's study design description: {study_design_description}

Evaluate the study design."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result_dict = await provider.chat_json(messages, temperature=0.0)
        except Exception as e:
            return AgentResult(
                success=True,
                data={
                    "design_type": "unknown",
                    "design_description": f"LLM classification failed: {e}",
                    "appropriate_for_question": None,
                    "concerns": [],
                    "recommendations": [],
                    "confidence": "insufficient",
                    "design_state": DesignState.UNKNOWN.value,
                },
                warnings=[f"LLM error: {e}. Design state set to UNKNOWN."],
                confidence="insufficient",
            )

        # Classify the design state
        design_state, _ = classify_design_state(
            study_design_from_researcher=study_design_description,
            study_design_from_llm=result_dict.get("design_type"),
        )

        result_dict["design_state"] = design_state.value

        # Store in memory
        project_id = context.get("project_id")
        if project_id:
            await store_memory(
                memory_type="project",
                category="study_design",
                key="design_type",
                value=result_dict.get("design_type", ""),
                source="ai_interpretation",
                confidence=result_dict.get("confidence", "low"),
                project_id=project_id,
            )

        warnings = []
        if not result_dict.get("appropriate_for_question", True):
            warnings.append("The stated study design may not be appropriate for this research question.")
        concerns = result_dict.get("concerns", [])
        warnings.extend([f"Methodological concern: {c}" for c in concerns])

        return AgentResult(
            success=True,
            data=result_dict,
            warnings=warnings,
            confidence=result_dict.get("confidence", "moderate"),
        )
