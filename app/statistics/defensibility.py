"""Defensibility Assessment — deterministic evaluation of methodological defensibility.

Produces a DefensibilityStatus based on:
- Execution status
- Numerical sanity status
- Statistical diagnostic status
- Survival-specific checks (PH assumption, sparse events, censoring balance)

This does NOT mean:
- statistically significant
- clinically important
- causally effective
- publishable

It only describes whether the executed analysis satisfies the
deterministic methodological checks implemented by STATIS.
"""

from __future__ import annotations

from app.statistics.result import ExecutionStatus, SanityStatus
from app.statistics.diagnostics import (
    DefensibilityStatus,
    DiagnosticDiagnosticStatus,
    DiagnosticSummary,
)


def assess_defensibility(
    execution_status: ExecutionStatus,
    sanity_status: SanityStatus,
    diagnostic_status: DiagnosticDiagnosticStatus,
    *,
    execution_blocked: bool = False,
    survival_metadata: dict | None = None,
) -> DefensibilityStatus:
    """Compute deterministic defensibility status.

    Rules (applied in order):
    1. If execution failed or was blocked → NOT_DEFENSIBLE
    2. If numerical sanity failed → NOT_DEFENSIBLE
    3. If diagnostics failed → NOT_DEFENSIBLE
    4. Survival-specific review triggers → REVIEW_REQUIRED
    5. If diagnostics require review → REVIEW_REQUIRED
    6. If diagnostics warned → DEFENSIBLE_WITH_CAVEATS
    7. Otherwise → DEFENSIBLE

    survival_metadata (optional) dict keys:
        method_id: str — "kaplan_meier", "log_rank", "cox_proportional_hazards"
        n_events: int — total events observed
        n_censored: int — total censored observations
        n_total: int — total observations
        ph_assumption_met: bool | None — for Cox PH only
        sparse_events: bool — True if events < 10
        all_events: bool — True if n_events == n_total
        no_events: bool — True if n_events == 0
    """
    # Rule 1: execution failure
    if execution_status != ExecutionStatus.SUCCESS or execution_blocked:
        return DefensibilityStatus.NOT_DEFENSIBLE

    # Rule 2: numerical sanity failure
    if sanity_status == SanityStatus.FAIL:
        return DefensibilityStatus.NOT_DEFENSIBLE

    # Rule 3: diagnostic failure
    if diagnostic_status == DiagnosticDiagnosticStatus.FAIL:
        return DefensibilityStatus.NOT_DEFENSIBLE

    # Rule 4: survival-specific review triggers
    if survival_metadata:
        survival_review = _check_survival_defensibility(survival_metadata)
        if survival_review is not None:
            return survival_review

    # Rule 5: diagnostic review required
    if diagnostic_status == DiagnosticDiagnosticStatus.REVIEW_REQUIRED:
        return DefensibilityStatus.REVIEW_REQUIRED

    # Rule 6: diagnostic warnings
    if diagnostic_status == DiagnosticDiagnosticStatus.WARNING:
        return DefensibilityStatus.DEFENSIBLE_WITH_CAVEATS

    # Rule 7: default
    return DefensibilityStatus.DEFENSIBLE


def _check_survival_defensibility(metadata: dict) -> DefensibilityStatus | None:
    """Check survival-specific defensibility conditions.

    Returns a DefensibilityStatus if review is needed, None if OK.
    """
    n_events = metadata.get("n_events", 0)
    n_censored = metadata.get("n_censored", 0)
    n_total = metadata.get("n_total", 0)
    method_id = metadata.get("method_id", "")
    ph_met = metadata.get("ph_assumption_met")
    sparse_events = metadata.get("sparse_events", False)
    no_events = metadata.get("no_events", False)
    all_events = metadata.get("all_events", False)

    # Zero events: survival analysis is meaningless
    if no_events or n_events == 0:
        return DefensibilityStatus.NOT_DEFENSIBLE

    # Cox PH: PH assumption violation requires review
    if method_id == "cox_proportional_hazards" and ph_met is False:
        return DefensibilityStatus.REVIEW_REQUIRED

    # Cox PH: perfect convergence issues (all events or no events)
    if method_id == "cox_proportional_hazards" and (all_events or no_events):
        return DefensibilityStatus.REVIEW_REQUIRED

    # Sparse events: reviewer should inspect
    if sparse_events or (n_events > 0 and n_events < 10):
        return DefensibilityStatus.REVIEW_REQUIRED

    # Extreme censoring (>90%): limited information
    if n_total > 0 and n_censored > 0.9 * n_total:
        return DefensibilityStatus.REVIEW_REQUIRED

    return None
