"""Data Dictionary Agent — auto-construct a data dictionary for the dataset.

Phase 2D.2: Uses the variable role resolver for deterministic role assignment.
Stores role resolutions with proper provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent, AgentResult
from app.memory.store import store_memory
from app.statistics.variable_roles import (
    VariableRole,
    RoleSource,
    resolve_variable_roles,
    is_identifier,
)


class DataDictionaryAgent(BaseAgent):
    name = "data_dictionary"
    requires_llm = False
    description = "Build a data dictionary: types, distributions, candidate roles."

    async def _execute(self, context: dict) -> AgentResult:
        file_path = Path(context["file_path"])
        project_id = context.get("project_id")
        research_question = context.get("research_question")

        df = self._load(file_path)
        if df is None:
            return AgentResult(success=False, data={"error": "Cannot load dataset"})

        variables = []
        for col in df.columns:
            series = df[col]
            info = self._analyze_variable(col, series)
            variables.append(info)

        # Resolve roles using the new resolver
        role_resolutions = resolve_variable_roles(variables, research_question)

        # Update variable roles with resolved values
        for resolution in role_resolutions.resolutions:
            var_info = next((v for v in variables if v["name"] == resolution.variable_name), None)
            if var_info:
                var_info["role"] = resolution.proposed_role.value
                var_info["role_source"] = resolution.role_source.value
                var_info["role_confidence"] = resolution.confidence
                var_info["role_reasoning"] = resolution.reasoning_summary
                if resolution.linked_research_question_term:
                    var_info["linked_rq_term"] = resolution.linked_research_question_term

        # Store variable classifications in memory with proper provenance
        for var in variables:
            if var["role"]:
                await store_memory(
                    memory_type="project",
                    category="variable_role",
                    key=f"{var['name']}_role",
                    value=var["role"],
                    source=var.get("role_source", "deterministic_rule"),
                    confidence=var.get("role_confidence", "moderate"),
                    project_id=project_id,
                )

        return AgentResult(
            success=True,
            data={
                "n_variables": len(variables),
                "variables": variables,
                "outcome_candidates": [v["name"] for v in variables if v["role"] == "outcome"],
                "predictor_candidates": [v["name"] for v in variables if v["role"] in ("predictor", "exposure", "grouping_variable")],
                "identifier_candidates": [v["name"] for v in variables if v["role"] == "identifier"],
                "confounder_candidates": [v["name"] for v in variables if v["role"] == "confounder_candidate"],
                "problem_type": role_resolutions.problem_type.value,
                "has_role_conflicts": role_resolutions.has_conflicts,
                "unresolved_variables": role_resolutions.unresolved_variables,
            },
            confidence="high",
        )

    def _load(self, path: Path) -> pd.DataFrame | None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path)
            elif suffix == ".tsv":
                return pd.read_csv(path, sep="\t")
            elif suffix in (".xlsx", ".xls"):
                return pd.read_excel(path)
            elif suffix == ".parquet":
                return pd.read_parquet(path)
        except Exception:
            return None

    def _analyze_variable(self, name: str, series: pd.Series) -> dict:
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique())
        missing_pct = round(n_missing / n_total * 100, 2) if n_total > 0 else 0

        # Detect type
        if pd.api.types.is_datetime64_any_dtype(series):
            var_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            var_type = "numeric"
        elif pd.api.types.is_bool_dtype(series):
            var_type = "boolean"
        elif series.dtype == object:
            if n_unique <= 20:
                var_type = "categorical"
            else:
                var_type = "text"
        else:
            var_type = str(series.dtype)

        # Distribution summary
        dist = {}
        if var_type == "numeric":
            valid = series.dropna()
            if len(valid) > 0:
                dist = {
                    "mean": float(valid.mean()),
                    "std": float(valid.std()) if len(valid) > 1 else 0,
                    "min": float(valid.min()),
                    "q25": float(valid.quantile(0.25)),
                    "median": float(valid.median()),
                    "q75": float(valid.quantile(0.75)),
                    "max": float(valid.max()),
                    "skewness": float(valid.skew()) if len(valid) > 2 else None,
                    "kurtosis": float(valid.kurtosis()) if len(valid) > 3 else None,
                }
        elif var_type in ("categorical", "boolean"):
            vc = series.value_counts().head(10)
            dist = {str(k): int(v) for k, v in vc.items()}

        # Sample values
        sample = series.dropna().head(5).tolist()
        sample = [_to_jsonable(v) for v in sample]

        return {
            "name": name,
            "dtype": var_type,
            "role": None,  # Will be resolved by the resolver
            "role_source": None,
            "n_unique": n_unique,
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "is_identifier": n_unique == n_total and n_missing == 0,
            "sample_values": sample,
            "distribution_summary": dist,
        }


def _to_jsonable(v):
    """Convert a value to JSON-serializable form."""
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return str(v)
