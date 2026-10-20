"""Statistical Diagnostics Framework — deterministic assumption and diagnostic checks.

Every diagnostic is computed deterministically from the dataset and the
statistical result. No LLM reasoning is involved.

Diagnostics produce structured DiagnosticResult objects that record:
- What was checked
- What was observed
- Whether it passed, warned, or failed
- Why (methodological rationale)

Diagnostics NEVER:
- Select methods based on p-values
- Automatically change statistical tests
- Remove observations
- Transform variables
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DiagnosticSeverity(str, Enum):
    """Severity of a diagnostic finding."""
    INFO = "info"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"
    ERROR = "error"


class DiagnosticStatus(str, Enum):
    """Outcome of a diagnostic check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


class DefensibilityStatus(str, Enum):
    """Overall methodological defensibility of the analysis.

    This does NOT mean:
    - statistically significant
    - clinically important
    - causally effective
    - publishable

    It only describes whether the executed analysis satisfies the
    deterministic methodological checks implemented by STATIS.
    """
    DEFENSIBLE = "defensible"
    DEFENSIBLE_WITH_CAVEATS = "defensible_with_caveats"
    REVIEW_REQUIRED = "review_required"
    NOT_DEFENSIBLE = "not_defensible"


class DiagnosticDiagnosticStatus(str, Enum):
    """Overall status of all diagnostics for an analysis."""
    PASS = "pass"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class DiagnosticResult(BaseModel):
    """A single deterministic diagnostic check result."""
    diagnostic_id: str
    diagnostic_name: str
    severity: DiagnosticSeverity = DiagnosticSeverity.INFO
    status: DiagnosticStatus = DiagnosticStatus.NOT_CHECKED
    variable: str | None = None
    variables: list[str] = []
    method: str = ""
    observed_value: Any = None
    threshold: Any = None
    message: str = ""
    rationale: str = ""
    deterministic: bool = True
    provenance: dict[str, Any] = {}


class DiagnosticSummary(BaseModel):
    """Aggregated summary of all diagnostics for an analysis."""
    overall_status: DiagnosticDiagnosticStatus = DiagnosticDiagnosticStatus.PASS
    diagnostics: list[DiagnosticResult] = []
    warnings: list[str] = []
    review_required: list[str] = []
    errors: list[str] = []
    assumptions_checked: list[str] = []
    assumptions_failed: list[str] = []
    assumptions_uncertain: list[str] = []

    def add_diagnostic(self, diag: DiagnosticResult):
        """Add a diagnostic and update summary counts."""
        self.diagnostics.append(diag)
        if diag.status == DiagnosticStatus.WARNING:
            self.warnings.append(diag.message)
        elif diag.status == DiagnosticStatus.REVIEW_REQUIRED:
            self.review_required.append(diag.message)
        elif diag.status == DiagnosticStatus.FAIL:
            self.errors.append(diag.message)

    def finalize(self):
        """Compute overall status from individual diagnostics."""
        if self.errors:
            self.overall_status = DiagnosticDiagnosticStatus.FAIL
        elif self.review_required:
            self.overall_status = DiagnosticDiagnosticStatus.REVIEW_REQUIRED
        elif self.warnings:
            self.overall_status = DiagnosticDiagnosticStatus.WARNING
        else:
            self.overall_status = DiagnosticDiagnosticStatus.PASS


def build_diagnostic_summary(
    diagnostics: list[DiagnosticResult],
    assumptions_checked: list[str] | None = None,
) -> DiagnosticSummary:
    """Build a DiagnosticSummary from a list of DiagnosticResults."""
    summary = DiagnosticSummary(
        assumptions_checked=assumptions_checked or [],
    )
    for d in diagnostics:
        summary.add_diagnostic(d)
        if d.status == DiagnosticStatus.FAIL:
            summary.assumptions_failed.append(d.diagnostic_name)
        elif d.status in (DiagnosticStatus.WARNING, DiagnosticStatus.REVIEW_REQUIRED):
            summary.assumptions_uncertain.append(d.diagnostic_name)
    summary.finalize()
    return summary
