"""Reproducibility Engine — verify that analysis runs can be reproduced identically.

Given the same inputs (specification, dataset identity, parameters), the same
deterministic engine must produce the same result. This module verifies that
invariant.
"""

from __future__ import annotations

from typing import Any

from app.data.run_store import (
    compute_input_fingerprint,
    compute_result_fingerprint,
    find_runs_by_fingerprint,
    get_analysis_run,
)


class ReproducibilityStatus:
    VERIFIED = "verified"         # Same inputs → same result
    MISMATCH = "mismatch"         # Same inputs → different result (BUG)
    NEW_RUN = "new_run"           # No prior run with these inputs
    INCOMPLETE = "incomplete"     # Prior run did not complete
    ENVIRONMENT_DIFF = "environment_diff"  # Different Python/engine version


async def verify_reproducibility(
    specification: dict,
    dataset_content_hash: str,
    dataset_schema_hash: str,
    current_result: dict | None = None,
) -> dict:
    """Verify that an analysis is reproducible.

    If current_result is provided, compares it against the most recent run
    with the same inputs. If not provided, just checks if prior runs exist.

    Returns a dict with:
        - status: one of ReproducibilityStatus values
        - matching_runs: list of prior runs with same inputs
        - prior_result: the most recent matching result (if any)
        - differences: list of specific differences (if mismatch)
    """
    input_fp = compute_input_fingerprint(
        specification, dataset_content_hash, dataset_schema_hash
    )

    prior_runs = await find_runs_by_fingerprint(input_fp)

    # Filter to completed runs
    completed_runs = [r for r in prior_runs if r["completed_at"] is not None]

    if not completed_runs:
        return {
            "status": ReproducibilityStatus.NEW_RUN,
            "input_fingerprint": input_fp,
            "matching_runs": [],
            "prior_result": None,
            "differences": [],
        }

    most_recent = completed_runs[0]
    prior_result = most_recent.get("result")

    # Check environment compatibility
    env_diffs = []
    if most_recent.get("python_version"):
        import sys
        current_py = sys.version.split()[0]
        if current_py != most_recent["python_version"]:
            env_diffs.append(f"Python: {most_recent['python_version']} → {current_py}")
    if most_recent.get("engine_version"):
        from app.data.run_store import capture_environment
        current_env = capture_environment()
        if current_env["engine_version"] != most_recent["engine_version"]:
            env_diffs.append(
                f"Engine: {most_recent['engine_version']} → {current_env['engine_version']}"
            )

    if env_diffs:
        return {
            "status": ReproducibilityStatus.ENVIRONMENT_DIFF,
            "input_fingerprint": input_fp,
            "matching_runs": completed_runs[:5],
            "prior_result": prior_result,
            "differences": env_diffs,
        }

    # If no current result to compare, just report prior runs exist
    if current_result is None:
        return {
            "status": ReproducibilityStatus.VERIFIED,
            "input_fingerprint": input_fp,
            "matching_runs": completed_runs[:5],
            "prior_result": prior_result,
            "differences": [],
        }

    # Compare results
    current_fp = compute_result_fingerprint(current_result)
    prior_fp = most_recent.get("result_fingerprint")

    if prior_fp and current_fp == prior_fp:
        return {
            "status": ReproducibilityStatus.VERIFIED,
            "input_fingerprint": input_fp,
            "result_fingerprint": current_fp,
            "matching_runs": completed_runs[:5],
            "prior_result": prior_result,
            "differences": [],
        }

    # Fingerprints don't match — find specific differences
    differences = _find_result_differences(current_result, prior_result)

    return {
        "status": ReproducibilityStatus.MISMATCH,
        "input_fingerprint": input_fp,
        "result_fingerprint": current_fp,
        "matching_runs": completed_runs[:5],
        "prior_result": prior_result,
        "differences": differences,
    }


def _find_result_differences(current: Any, prior: Any, path: str = "") -> list[str]:
    """Recursively find differences between two result structures."""
    differences = []

    if type(current) != type(prior):
        differences.append(f"{path}: type mismatch ({type(current).__name__} vs {type(prior).__name__})")
        return differences

    if isinstance(current, dict):
        all_keys = set(current.keys()) | set(prior.keys())
        for key in sorted(all_keys):
            if key not in current:
                differences.append(f"{path}.{key}: missing in current")
            elif key not in prior:
                differences.append(f"{path}.{key}: missing in prior")
            else:
                differences.extend(_find_result_differences(current[key], prior[key], f"{path}.{key}"))
    elif isinstance(current, list):
        if len(current) != len(prior):
            differences.append(f"{path}: length mismatch ({len(current)} vs {len(prior)})")
        for i, (c, p) in enumerate(zip(current, prior)):
            differences.extend(_find_result_differences(c, p, f"{path}[{i}]"))
    elif isinstance(current, float):
        if abs(current - prior) > 1e-6:
            differences.append(f"{path}: {current} vs {prior}")
    elif current != prior:
        differences.append(f"{path}: {current} vs {prior}")

    return differences
