"""Analysis Revalidation — revalidate analyses after dataset transformations.

When a derived dataset is created, existing analysis specifications must be
revalidated against the new dataset to ensure they remain valid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.statistics.specification import StatisticalAnalysisSpecification
from app.statistics.validator import StatisticalValidator, ValidationResult


def revalidate_analyses(
    derived_file_path: Path | str,
    specifications: list[dict],
) -> list[dict]:
    """Revalidate a list of analysis specifications against a derived dataset.

    Returns a list of revalidation results, one per specification.
    Each result contains: analysis_id, status, validation_result, original_status.
    """
    validator = StatisticalValidator()
    results = []

    for spec_dict in specifications:
        analysis_id = spec_dict.get("analysis_id", "unknown")
        original_status = spec_dict.get("validation_status", "pending")

        try:
            spec = StatisticalAnalysisSpecification(**spec_dict)
            validation = validator.validate(spec, file_path=Path(derived_file_path))
            results.append({
                "analysis_id": analysis_id,
                "status": validation.status.value,
                "validation_result": {
                    "warnings": validation.warnings,
                    "blockers": validation.blockers,
                    "checks": [
                        {"check": c.check_name, "status": c.status, "description": c.description}
                        for c in validation.checks
                    ],
                },
                "original_status": original_status,
                "changed": validation.status.value != original_status,
                "blocked_by_transformation": validation.status.value == "block",
            })
        except Exception as e:
            results.append({
                "analysis_id": analysis_id,
                "status": "error",
                "validation_result": {"error": str(e)},
                "original_status": original_status,
                "changed": True,
                "blocked_by_transformation": True,
            })

    return results


def check_execution_allowed(
    derived_file_path: Path | str,
    spec_dict: dict,
) -> tuple[bool, str]:
    """Check if an analysis is still allowed to execute on a derived dataset.

    Returns (allowed, reason).
    """
    validator = StatisticalValidator()

    try:
        spec = StatisticalAnalysisSpecification(**spec_dict)
        validation = validator.validate(spec, file_path=Path(derived_file_path))

        if validation.status.value == "block":
            blockers = "; ".join(validation.blockers)
            return False, f"Revalidation failed: {blockers}"

        return True, "Revalidation passed"

    except Exception as e:
        return False, f"Revalidation error: {e}"
