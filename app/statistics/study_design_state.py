"""Study Design State — explicit states for study design information.

Prevents fabricating "observational" when design is unknown.
"""

from __future__ import annotations

from enum import Enum


class DesignState(str, Enum):
    """Explicit states for study design information."""
    KNOWN = "known"              # Researcher provided, confirmed
    UNKNOWN = "unknown"          # No design information provided
    AMBIGUOUS = "ambiguous"      # Partial or contradictory information
    RESEARCHER_CONFIRMED = "researcher_confirmed"  # Researcher confirmed the classification


class DesignType(str, Enum):
    """Supported study design types."""
    CROSS_SECTIONAL = "cross_sectional"
    CASE_CONTROL = "case_control"
    COHORT_RETROSPECTIVE = "cohort_retrospective"
    COHORT_PROSPECTIVE = "cohort_prospective"
    RCT = "rct"
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    PROGNOSTIC = "prognostic"
    SURVIVAL = "survival"
    LONGITUDINAL = "longitudinal"
    ECOLOGICAL = "ecological"
    SURVEY = "survey"
    EXPERIMENTAL_LAB = "experimental_lab"
    UNKNOWN = "unknown"


def classify_design_state(
    study_design_from_researcher: str | None = None,
    study_design_from_llm: str | None = None,
) -> tuple[DesignState, DesignType | None]:
    """Classify the study design state.

    Returns (state, design_type).
    If no design info is provided, state = UNKNOWN and design_type = None.
    """
    if study_design_from_researcher and study_design_from_researcher.strip():
        # Researcher provided explicit design info
        return DesignState.KNOWN, _normalize_design_type(study_design_from_researcher)

    if study_design_from_llm and study_design_from_llm.strip():
        # LLM inferred a design, but researcher didn't confirm
        return DesignState.AMBIGUOUS, _normalize_design_type(study_design_from_llm)

    return DesignState.UNKNOWN, None


def _normalize_design_type(design_str: str) -> DesignType:
    """Map a design string to a DesignType enum."""
    lower = design_str.lower().strip()

    mapping = {
        "cross_sectional": DesignType.CROSS_SECTIONAL,
        "cross-sectional": DesignType.CROSS_SECTIONAL,
        "case_control": DesignType.CASE_CONTROL,
        "case-control": DesignType.CASE_CONTROL,
        "cohort": DesignType.COHORT_PROSPECTIVE,
        "retrospective": DesignType.COHORT_RETROSPECTIVE,
        "prospective": DesignType.COHORT_PROSPECTIVE,
        "rct": DesignType.RCT,
        "randomized": DesignType.RCT,
        "randomised": DesignType.RCT,
        "diagnostic": DesignType.DIAGNOSTIC_ACCURACY,
        "prognostic": DesignType.PROGNOSTIC,
        "survival": DesignType.SURVIVAL,
        "longitudinal": DesignType.LONGITUDINAL,
        "ecological": DesignType.ECOLOGICAL,
        "survey": DesignType.SURVEY,
        "experimental": DesignType.EXPERIMENTAL_LAB,
        "laboratory": DesignType.EXPERIMENTAL_LAB,
    }

    for key, dtype in mapping.items():
        if key in lower:
            return dtype

    return DesignType.UNKNOWN
