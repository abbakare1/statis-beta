"""StatisticalResult — standardized contract for all statistical computation outputs.

Every deterministic statistical method MUST return a StatisticalResult.
This is the single source of truth for what was computed, its validity,
and its provenance. No interpretation or scientific summary is included.
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Whether execution completed successfully."""
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"  # guardrails prevented execution


class SanityStatus(str, Enum):
    """Result of numerical sanity validation."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"


class NumericalSanity(BaseModel):
    """Deterministic sanity checks on computed values."""
    status: SanityStatus = SanityStatus.NOT_CHECKED
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    blockers: list[str] = []


class DataVerification(BaseModel):
    """Dataset identity verification before execution."""
    verified: bool = False
    identity_verified: bool = False  # True only when expected hashes match
    identity_mismatch_reason: str | None = None  # reason for identity failure
    content_hash: str | None = None
    schema_hash: str | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    columns: list[str] = []
    variables_used: list[str] = []
    n_complete_cases: int | None = None
    verification_warnings: list[str] = []


class StatisticalResult(BaseModel):
    """Standardized output contract for all statistical computations.

    Every method returns this structure. Fields that don't apply to a
    specific method are left as None/empty.

    This model does NOT contain scientific interpretation.
    It contains only computed values, metadata, and validation status.
    """
    # Identity
    method_id: str
    method_name: str
    execution_status: ExecutionStatus = ExecutionStatus.SUCCESS

    # Input verification
    data_verification: DataVerification = Field(default_factory=DataVerification)

    # Core results (method-specific, nullable)
    test_statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None  # e.g., "Cohen's d", "r", "OR"
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float = 0.95
    degrees_of_freedom: float | None = None

    # Sample information
    n_observed: int | None = None
    n_groups: int | None = None
    group_sizes: dict[str, int] | None = None

    # Descriptive statistics (method-specific)
    descriptive: dict[str, Any] = {}

    # Full result from underlying engine
    engine_result: dict[str, Any] = {}

    # Sanity validation
    numerical_sanity: NumericalSanity = Field(default_factory=NumericalSanity)

    # Guardrails
    guardrails_passed: bool = True
    guardrail_warnings: list[str] = []
    guardrail_blockers: list[str] = []

    # Statistical diagnostics
    diagnostic_status: str = "not_checked"  # pass, warning, review_required, fail, not_checked
    diagnostic_summary: dict[str, Any] = {}
    methodological_warnings: list[str] = []
    review_required: list[str] = []
    alternative_methods_considered: list[str] = []
    diagnostic_version: str = "2e.1"

    # Defensibility
    defensibility_status: str = "not_assessed"  # defensible, defensible_with_caveats, review_required, not_defensible, not_assessed

    # Warnings from computation
    computation_warnings: list[str] = []

    # Provenance
    engine_version: str = "2e.1"
    engine_name: str = "scipy+statsmodels"
    timestamp: float = Field(default_factory=time.time)

    # Reproducibility
    result_fingerprint: str | None = None

    def compute_fingerprint(self) -> str:
        """Compute deterministic fingerprint of this result for reproducibility."""
        canonical = {
            "method_id": self.method_id,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "n_observed": self.n_observed,
            "descriptive": _canonicalize(self.descriptive),
        }
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        self.result_fingerprint = hashlib.sha256(serialized).hexdigest()
        return self.result_fingerprint

    def to_summary(self) -> dict[str, Any]:
        """Produce a concise summary for display/storage (no interpretation)."""
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "execution_status": self.execution_status.value,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "confidence_interval": self.confidence_interval,
            "n_observed": self.n_observed,
            "n_groups": self.n_groups,
            "group_sizes": self.group_sizes,
            "guardrails_passed": self.guardrails_passed,
            "n_guardrail_warnings": len(self.guardrail_warnings),
            "n_guardrail_blockers": len(self.guardrail_blockers),
            "numerical_sanity_status": self.numerical_sanity.status.value,
            "diagnostic_status": self.diagnostic_status,
            "defensibility_status": self.defensibility_status,
            "n_methodological_warnings": len(self.methodological_warnings),
            "n_review_required": len(self.review_required),
            "result_fingerprint": self.result_fingerprint,
        }


def _canonicalize(obj: Any) -> Any:
    """Recursive canonicalization for deterministic fingerprinting."""
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    elif isinstance(obj, float):
        return round(obj, 10)
    elif isinstance(obj, bool):
        return obj
    return obj
