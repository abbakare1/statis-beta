"""Causal Language Guard — classifies statistical interpretations by causal strength.

Classifies statements as:
- SUPPORTED: Statement is appropriate given the study design and analysis
- ASSOCIATIONAL_ONLY: Statement uses causal language but only association is supported
- REVIEW_REQUIRED: Statement may require explicit justification
- BLOCKED: Statement makes unsupported causal claims

This does NOT silently rewrite the researcher's conclusions.
Instead it flags concerns and records the researcher's decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CausalVerdict(str, Enum):
    SUPPORTED = "supported"
    ASSOCIATIONAL_ONLY = "associational_only"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"


# Causal verbs and phrases that imply causation
_CAUSAL_VERBS = [
    r"\bcauses?\b",
    r"\bcaused\b",
    r"\bcausing\b",
    r"\bprevents?\b",
    r"\bprevented\b",
    r"\breduces?\s+risk\b",
    r"\bincreases?\s+risk\b",
    r"\bleads?\s+to\b",
    r"\bresults?\s+in\b",
    r"\bprotects?\s+against\b",
    r"\bexposure\s+to\b.*\bresulted\s+in\b",
    r"\btreatment\s+of\b.*\bleads\s+to\b",
    r"\beffect\s+of\b.*\bon\b",
    r"\bimpact\s+of\b.*\bon\b",
    r"\bcausal\b",
    r"\bconsequence\s+of\b",
    r"\bimproves?\b",
    r"\bworsens?\b",
    r"\bdrives?\b",
    r"\bmediates?\b",
]

# Associational phrases that are safe for observational data
_ASSOCIATIONAL_PHRASES = [
    r"\bassociated\s+with\b",
    r"\bcorrelated\s+with\b",
    r"\brelated\s+to\b",
    r"\blinked\s+to\b",
    r"\bpredicts?\b",
    r"\bpredicted\b",
    r"\binversely\s+associated\b",
    r"\bpositively\s+associated\b",
    r"\bnegatively\s+associated\b",
    r"\bhigher\s+.*\bassociated\s+with\b",
    r"\blower\s+.*\bassociated\s+with\b",
    r"\bincreased\s+.*\bassociated\s+with\b",
    r"\bobesity\s+was\s+associated\b",
    r"\bobesity\s+is\s+associated\b",
]

# RCT-specific terms that support causal language when design is experimental
_RCT_INDICATORS = [
    r"\brandomized\s+controlled\b",
    r"\brandomised\s+controlled\b",
    r"\brct\b",
    r"\brandom\s+assignment\b",
    r"\brandom\s+allocation\b",
    r"\bplacebo\b",
    r"\bdouble[-\s]blind\b",
    r"\bsingle[-\s]blind\b",
    r"\bintervention\s+group\b",
    r"\bcontrol\s+group\b",
]


@dataclass
class CausalClassification:
    """Result of classifying a statement."""
    verdict: CausalVerdict
    statement: str
    causal_phrases_found: list[str] = field(default_factory=list)
    associational_phrases_found: list[str] = field(default_factory=list)
    study_design: str | None = None
    is_rct: bool = False
    reasoning: str = ""


class CausalLanguageGuard:
    """Classify statistical interpretations by causal strength.

    Usage:
        guard = CausalLanguageGuard()
        result = guard.classify(
            statement="Obesity caused higher readmission rates.",
            study_design="retrospective_cohort",
        )
        # result.verdict == CausalVerdict.REVIEW_REQUIRED
    """

    def classify(
        self,
        statement: str,
        study_design: str | None = None,
        analysis_method: str | None = None,
        causal_inference_documented: bool = False,
    ) -> CausalClassification:
        """Classify a statement's causal language.

        Args:
            statement: The generated or written interpretation to classify.
            study_design: The study design (e.g., 'retrospective_cohort', 'rct').
            analysis_method: The statistical method used (e.g., 'logistic_regression').
            causal_inference_documented: Whether the researcher has explicitly
                documented causal inference assumptions and methodology.
        """
        is_rct = self._is_rct_design(study_design)
        causal_phrases = self._find_causal_phrases(statement)
        associational_phrases = self._find_associational_phrases(statement)

        # Decision logic
        if not causal_phrases:
            # No causal language detected
            return CausalClassification(
                verdict=CausalVerdict.SUPPORTED,
                statement=statement,
                causal_phrases_found=[],
                associational_phrases_found=associational_phrases,
                study_design=study_design,
                is_rct=is_rct,
                reasoning="No causal language detected in statement.",
            )

        # Causal language IS present
        if is_rct or causal_inference_documented:
            # RCT or documented causal inference methodology
            return CausalClassification(
                verdict=CausalVerdict.SUPPORTED,
                statement=statement,
                causal_phrases_found=causal_phrases,
                associational_phrases_found=associational_phrases,
                study_design=study_design,
                is_rct=is_rct,
                reasoning=(
                    f"Causal language detected ({', '.join(causal_phrases)}), "
                    f"but study design '{study_design}' supports causal interpretation"
                    f"{' (RCT)' if is_rct else ' (documented causal inference)'}"
                ),
            )

        # Observational study with causal language
        if self._is_strong_causal(causal_phrases):
            return CausalClassification(
                verdict=CausalVerdict.REVIEW_REQUIRED,
                statement=statement,
                causal_phrases_found=causal_phrases,
                associational_phrases_found=associational_phrases,
                study_design=study_design,
                is_rct=False,
                reasoning=(
                    f"Strong causal language detected ({', '.join(causal_phrases)}) "
                    f"in observational study '{study_design}'. "
                    f"Researcher must explicitly justify causal interpretation or "
                    f"rewrite as associational."
                ),
            )

        # Moderate causal language in observational study
        return CausalClassification(
            verdict=CausalVerdict.ASSOCIATIONAL_ONLY,
            statement=statement,
            causal_phrases_found=causal_phrases,
            associational_phrases_found=associational_phrases,
            study_design=study_design,
            is_rct=False,
            reasoning=(
                f"Causal language detected ({', '.join(causal_phrases)}) "
                f"but study design '{study_design}' only supports associational interpretation. "
                f"Consider rephrasing as: 'X was associated with Y' instead of 'X caused Y'."
            ),
        )

    def suggest_revision(self, classification: CausalClassification) -> str | None:
        """Suggest an associational revision if the verdict is not SUPPORTED.

        Does NOT modify the original statement.
        Returns a suggested alternative or None.
        """
        if classification.verdict == CausalVerdict.SUPPORTED:
            return None

        statement = classification.statement

        # Simple pattern-based suggestions
        revisions = [
            (r"\bcaused\b", "was associated with"),
            (r"\bcauses\b", "is associated with"),
            (r"\bprevented\b", "was inversely associated with"),
            (r"\bprevents\b", "is inversely associated with"),
            (r"\breduces?\s+risk\b", "is associated with lower risk of"),
            (r"\bincreases?\s+risk\b", "is associated with higher risk of"),
            (r"\bleads?\s+to\b", "is associated with"),
            (r"\bresults?\s+in\b", "is associated with"),
            (r"\bimproves\b", "is associated with improvement in"),
            (r"\bworsens\b", "is associated with worsening of"),
        ]

        revised = statement
        for pattern, replacement in revisions:
            new = re.sub(pattern, replacement, revised, flags=re.IGNORECASE)
            if new != revised:
                revised = new
                break  # Only replace the first causal phrase

        if revised != statement:
            return revised
        return None

    def _is_rct_design(self, study_design: str | None) -> bool:
        """Check if the study design indicates an RCT."""
        if not study_design:
            return False
        design_lower = study_design.lower().replace(" ", "_")
        for pattern in _RCT_INDICATORS:
            if re.search(pattern, design_lower, re.IGNORECASE):
                return True
        return design_lower in ("rct", "randomized_controlled_trial", "experimental")

    def _find_causal_phrases(self, statement: str) -> list[str]:
        """Find causal phrases in the statement."""
        found = []
        for pattern in _CAUSAL_VERBS:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                found.append(match.group())
        return found

    def _find_associational_phrases(self, statement: str) -> list[str]:
        """Find associational phrases in the statement."""
        found = []
        for pattern in _ASSOCIATIONAL_PHRASES:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                found.append(match.group())
        return found

    def _is_strong_causal(self, phrases: list[str]) -> bool:
        """Check if any phrase is a strong causal claim."""
        strong = {
            "caused", "causes", "cause",
            "prevented", "prevents", "prevent",
            "leads to", "lead to", "leading to",
            "results in", "resulted in",
            "reduces risk", "reduce risk",
            "increases risk", "increase risk",
            "improves", "worsens",
        }
        return any(p.lower() in strong for p in phrases)


@dataclass
class CausalGuardResult:
    """Result from the guard with researcher decision tracking."""
    classification: CausalClassification
    original_statement: str
    suggested_revision: str | None = None
    researcher_decision: str | None = None  # accept, modify, override
    researcher_modified_statement: str | None = None
    researcher_reason: str | None = None
    timestamp: str | None = None
