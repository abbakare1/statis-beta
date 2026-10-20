"""Anti-P-Hacking Guardrails — prevent optimization of analysis for significance.

The system may compare strategies for robustness but must never optimize
the analysis for statistical significance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class AntiPHackingVerdict(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


class AntiPHackingCheck(BaseModel):
    """A single anti-p-hacking check result."""
    check_name: str
    verdict: AntiPHackingVerdict
    description: str
    details: str | None = None


class AntiPHackingGuard:
    """Guard against p-hacking and significance-seeking behavior."""

    def check_strategy_selection(
        self,
        strategies: list[dict[str, Any]],
        selected_strategy: dict[str, Any],
        p_value: float | None = None,
    ) -> list[AntiPHackingCheck]:
        """Check if strategy selection is potentially p-hacking.

        Flags:
        - Selecting based on smallest p-value
        - Removing outliers until significance
        - Choosing imputation that improves significance
        - Trying analyses until one is significant
        """
        checks = []

        # Check 1: p-value as selection criterion
        if p_value is not None and "p_value_reason" in str(selected_strategy):
            checks.append(AntiPHackingCheck(
                check_name="p_value_selection",
                verdict=AntiPHackingVerdict.BLOCKED,
                description="Strategy selected based on p-value. This is p-hacking.",
                details="Do not select transformations or outlier handling based on resulting p-values.",
            ))

        # Check 2: Multiple strategies with significance comparison
        if len(strategies) > 1:
            has_significance_comparison = any(
                "significance" in str(s.get("comparison_reason", "")).lower()
                or "p_value" in str(s.get("comparison_reason", "")).lower()
                for s in strategies
            )
            if has_significance_comparison:
                checks.append(AntiPHackingCheck(
                    check_name="significance_comparison",
                    verdict=AntiPHackingVerdict.SUSPICIOUS,
                    description="Strategies compared based on statistical significance.",
                    details="Compare strategies for robustness, not for achieving significance.",
                ))

        # Check 3: Iterative exclusion pattern
        if "iteration_count" in str(selected_strategy):
            try:
                iteration = int(selected_strategy.get("iteration_count", 0))
                if iteration > 1:
                    checks.append(AntiPHackingCheck(
                        check_name="iterative_exclusion",
                        verdict=AntiPHackingVerdict.SUSPICIOUS,
                        description=f"Iterative outlier exclusion ({iteration} iterations).",
                        details="Repeatedly removing outliers to achieve significance is p-hacking.",
                    ))
            except (ValueError, TypeError):
                pass

        if not checks:
            checks.append(AntiPHackingCheck(
                check_name="default",
                verdict=AntiPHackingVerdict.SAFE,
                description="No p-hacking concerns detected.",
            ))

        return checks

    def check_transformation_selection(
        self,
        transformations: list[dict[str, Any]],
        selected: dict[str, Any],
        result_p_value: float | None = None,
    ) -> list[AntiPHackingCheck]:
        """Check if transformation selection is potentially p-hacking."""
        checks = []

        if result_p_value is not None and selected.get("selection_reason") == "smallest_p_value":
            checks.append(AntiPHackingCheck(
                check_name="transformation_p_value_selection",
                verdict=AntiPHackingVerdict.BLOCKED,
                description="Transformation selected to minimize p-value.",
                details="Transformations must be selected based on distributional properties and domain requirements, not significance.",
            ))

        if not checks:
            checks.append(AntiPHackingCheck(
                check_name="transformation_default",
                verdict=AntiPHackingVerdict.SAFE,
                description="No transformation p-hacking concerns detected.",
            ))

        return checks

    def check_outlier_exclusion(
        self,
        exclusion_justification: str,
        affects_significance: bool = False,
    ) -> list[AntiPHackingCheck]:
        """Check if outlier exclusion is justified on methodological grounds."""
        checks = []

        if affects_significance:
            checks.append(AntiPHackingCheck(
                check_name="outlier_significance_justification",
                verdict=AntiPHackingVerdict.BLOCKED,
                description="Outlier exclusion justified by achieving significance.",
                details="Outliers must be excluded based on documented data errors or methodological grounds, not significance.",
            ))

        methodological_terms = [
            "data_entry_error", "measurement_error", "equipment_failure",
            "protocol_violation", "duplicate_record", "known_error",
        ]
        has_methodological_justification = any(
            term in exclusion_justification.lower()
            for term in methodological_terms
        )

        if not has_methodological_justification and not affects_significance:
            checks.append(AntiPHackingCheck(
                check_name="outlier_justification",
                verdict=AntiPHackingVerdict.SUSPICIOUS,
                description="Outlier exclusion lacks documented methodological justification.",
                details="Document the specific reason for exclusion (e.g., data entry error, equipment failure).",
            ))

        if not checks:
            checks.append(AntiPHackingCheck(
                check_name="outlier_default",
                verdict=AntiPHackingVerdict.SAFE,
                description="Outlier exclusion has methodological justification.",
            ))

        return checks

    def generate_report(
        self,
        all_checks: list[AntiPHackingCheck],
    ) -> dict[str, Any]:
        """Generate an anti-p-hacking report from all checks."""
        n_safe = sum(1 for c in all_checks if c.verdict == AntiPHackingVerdict.SAFE)
        n_suspicious = sum(1 for c in all_checks if c.verdict == AntiPHackingVerdict.SUSPICIOUS)
        n_blocked = sum(1 for c in all_checks if c.verdict == AntiPHackingVerdict.BLOCKED)

        overall = "safe"
        if n_blocked > 0:
            overall = "blocked"
        elif n_suspicious > 0:
            overall = "suspicious"

        return {
            "overall_verdict": overall,
            "n_safe": n_safe,
            "n_suspicious": n_suspicious,
            "n_blocked": n_blocked,
            "checks": [c.model_dump() for c in all_checks],
        }
