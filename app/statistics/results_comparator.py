"""Results Comparator — compare analysis results across different specifications.

When researchers want to understand how different analytical choices affect
conclusions, this module provides structured comparison of results from
different runs.
"""

from __future__ import annotations

from typing import Any

from app.data.run_store import get_analysis_run, compute_result_fingerprint


class ComparisonResult:
    """Structured comparison of two analysis results."""

    def __init__(self, run_a: dict, run_b: dict):
        self.run_a = run_a
        self.run_b = run_b
        self.differences: list[dict] = []
        self.similarities: list[dict] = []
        self._compare()

    def _compare(self):
        """Compare the two results field by field."""
        result_a = self.run_a.get("result") or {}
        result_b = self.run_b.get("result") or {}

        # Compare top-level numeric fields
        all_keys = set(result_a.keys()) | set(result_b.keys())
        for key in sorted(all_keys):
            val_a = result_a.get(key)
            val_b = result_b.get(key)

            if key in result_a and key not in result_b:
                self.differences.append({
                    "field": key,
                    "type": "missing_in_b",
                    "value_a": _safe_serialize(val_a),
                })
            elif key not in result_a and key in result_b:
                self.differences.append({
                    "field": key,
                    "type": "missing_in_a",
                    "value_b": _safe_serialize(val_b),
                })
            elif self._values_differ(val_a, val_b):
                self.differences.append({
                    "field": key,
                    "type": "value_differs",
                    "value_a": _safe_serialize(val_a),
                    "value_b": _safe_serialize(val_b),
                })
            else:
                self.similarities.append({
                    "field": key,
                    "value": _safe_serialize(val_a),
                })

    def _values_differ(self, a: Any, b: Any) -> bool:
        """Check if two values are meaningfully different."""
        if type(a) != type(b):
            return True
        if isinstance(a, float):
            return abs(a - b) > 1e-6
        return a != b

    def to_dict(self) -> dict:
        return {
            "run_a": {
                "run_id": self.run_a.get("run_id"),
                "analysis_type": self.run_a.get("analysis_type"),
                "input_fingerprint": self.run_a.get("input_fingerprint"),
                "result_fingerprint": self.run_a.get("result_fingerprint"),
            },
            "run_b": {
                "run_id": self.run_b.get("run_id"),
                "analysis_type": self.run_b.get("analysis_type"),
                "input_fingerprint": self.run_b.get("input_fingerprint"),
                "result_fingerprint": self.run_b.get("result_fingerprint"),
            },
            "n_differences": len(self.differences),
            "n_similarities": len(self.similarities),
            "conclusions_same": self._conclusions_same(),
            "differences": self.differences,
            "similarities": self.similarities,
        }

    def _conclusions_same(self) -> bool | None:
        """Heuristic: do the p-values lead to the same conclusion at α=0.05?"""
        p_a = self._extract_p_value(self.run_a.get("result") or {})
        p_b = self._extract_p_value(self.run_b.get("result") or {})

        if p_a is None or p_b is None:
            return None

        return (p_a < 0.05) == (p_b < 0.05)

    def _extract_p_value(self, result: dict) -> float | None:
        """Try to extract a p-value from a result dict."""
        if "p_value" in result:
            return result["p_value"]
        if "pvalue" in result:
            return result["pvalue"]
        # Handle nested results (e.g., from regression)
        for key, val in result.items():
            if isinstance(val, dict):
                p = self._extract_p_value(val)
                if p is not None:
                    return p
        return None


def _safe_serialize(val: Any) -> Any:
    """Safely serialize a value for JSON output."""
    if isinstance(val, float):
        return round(val, 10)
    if isinstance(val, dict):
        return {k: _safe_serialize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_safe_serialize(item) for item in val]
    return val


async def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    """Compare two analysis runs by their IDs.

    Returns a structured ComparisonResult as a dict.
    """
    run_a = await get_analysis_run(run_id_a)
    run_b = await get_analysis_run(run_id_b)

    if not run_a:
        raise ValueError(f"Run {run_id_a} not found")
    if not run_b:
        raise ValueError(f"Run {run_id_b} not found")

    comparison = ComparisonResult(run_a, run_b)
    return comparison.to_dict()


async def compare_by_fingerprint(
    specification: dict,
    dataset_content_hash: str,
    dataset_schema_hash: str,
) -> dict:
    """Compare all runs that used the exact same inputs.

    If multiple runs exist with the same fingerprint, they should be identical
    (reproducibility check). If they differ, there's a non-determinism bug.
    """
    from app.data.run_store import find_runs_by_fingerprint, compute_input_fingerprint

    fp = compute_input_fingerprint(specification, dataset_content_hash, dataset_schema_hash)
    runs = await find_runs_by_fingerprint(fp)

    if len(runs) < 2:
        return {
            "input_fingerprint": fp,
            "n_runs": len(runs),
            "all_identical": True,
            "comparisons": [],
        }

    comparisons = []
    all_identical = True
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            comp = ComparisonResult(runs[i], runs[j])
            comp_dict = comp.to_dict()
            comparisons.append(comp_dict)
            if comp_dict["n_differences"] > 0:
                all_identical = False

    return {
        "input_fingerprint": fp,
        "n_runs": len(runs),
        "all_identical": all_identical,
        "comparisons": comparisons,
    }
