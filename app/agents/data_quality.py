"""Data Quality Agent — comprehensive data quality assessment.

Uses deterministic engines for:
- Missingness characterization
- Outlier detection (IQR, Z-score, MAD)
- Outlier interpretation and handling strategies
- Transformation suggestion and domain validation
- Sensitivity analysis variants
- Anti-p-hacking guardrails

Does NOT impute, transform, or remove data. Only detects and characterizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent, AgentResult
from app.statistics.data_quality import DataQualityProfile, MissingnessStatus
from app.statistics.missing_data import analyze_missingness, generate_missing_data_strategies
from app.statistics.outlier_engine import detect_outliers, generate_outlier_strategies, compute_sensitivity_variants
from app.statistics.transformations import suggest_transformations


class DataQualityAgent(BaseAgent):
    name = "data_quality"
    requires_llm = False
    description = "Produce a comprehensive data-quality report with handling strategies."

    async def _execute(self, context: dict) -> AgentResult:
        file_path = Path(context["file_path"])
        outcome = context.get("outcome")
        predictors = context.get("predictors", [])

        df = self._load(file_path)
        if df is None:
            return AgentResult(success=False, data={"error": "Cannot load dataset"})

        issues = []

        # 1. Missing data analysis (deterministic engine)
        missingness = analyze_missingness(df, outcome=outcome, predictors=predictors)
        missing_strategies = generate_missing_data_strategies(
            missingness, outcome=outcome, predictors=predictors,
        )

        # Convert missingness to legacy issue format
        for var in missingness.variables:
            if var.status == MissingnessStatus.NONE:
                continue
            severity_map = {
                MissingnessStatus.LOW: "info",
                MissingnessStatus.MODERATE: "warning",
                MissingnessStatus.HIGH: "warning",
                MissingnessStatus.EXTREME: "critical",
                MissingnessStatus.ALL_MISSING: "critical",
            }
            issues.append({
                "issue_type": "missing",
                "severity": severity_map.get(var.status, "info"),
                "column": var.variable,
                "description": f"Column '{var.variable}' has {var.pct_missing}% missing values ({var.status.value})",
                "n_affected": var.n_missing,
                "recommendation": var.recommendation,
            })

        # 2. Duplicate detection
        dup_report = self._detect_duplicates(df)
        issues.extend(dup_report["issues"])

        # 3. Invalid / impossible values
        validity_report = self._check_validity(df)
        issues.extend(validity_report["issues"])

        # 4. Outlier detection (IQR + Z-score + MAD)
        outlier_results = detect_outliers(df, methods=["iqr", "zscore", "mad"])
        outlier_strategies = generate_outlier_strategies(outlier_results)
        sensitivity_variants = compute_sensitivity_variants(outlier_results)

        for result in outlier_results:
            if result.n_flagged == 0:
                continue
            severity = "warning" if result.pct_flagged > 10 else "info"
            issues.append({
                "issue_type": "outlier",
                "severity": severity,
                "column": result.variable,
                "description": (
                    f"{result.n_flagged} potential outliers ({result.pct_flagged}%) "
                    f"in '{result.variable}' using {result.method} method "
                    f"[{result.interpretation.value}]"
                ),
                "n_affected": result.n_flagged,
                "recommendation": (
                    f"Interpretation: {result.interpretation.value}. "
                    f"Handling: {', '.join(s.value for s in result.handling_strategies)}."
                ),
            })

        # 5. Constant / near-constant variables
        const_report = self._check_constant_variables(df)
        issues.extend(const_report["issues"])

        # 6. Transformation suggestions (for numeric variables with skewness)
        transformation_candidates = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                candidates = suggest_transformations(df, col)
                transformation_candidates.extend(candidates)

        # 7. Build DataQualityProfile
        profile = DataQualityProfile(
            dataset_id=context.get("dataset_id"),
            row_count=df.shape[0],
            column_count=df.shape[1],
            duplicate_rows=dup_report["n_duplicates"],
            constant_variables=[
                col for col in df.columns if df[col].nunique(dropna=True) <= 1
            ],
            missingness=missingness,
            outlier_summary=outlier_results,
            missing_data_strategies=missing_strategies,
            transformation_candidates=transformation_candidates,
        )

        # Summary statistics
        total_missing_cells = int(df.isna().sum().sum())
        total_cells = df.shape[0] * df.shape[1]

        report = {
            "dataset_id": context.get("dataset_id"),
            "n_rows": df.shape[0],
            "n_columns": df.shape[1],
            "total_missing_cells": total_missing_cells,
            "missing_pct": round(total_missing_cells / total_cells * 100, 2) if total_cells > 0 else 0,
            "duplicate_rows": dup_report["n_duplicates"],
            "duplicate_pct": round(dup_report["n_duplicates"] / len(df) * 100, 2) if len(df) > 0 else 0,
            "n_issues": len(issues),
            "n_critical": sum(1 for i in issues if i["severity"] == "critical"),
            "n_warning": sum(1 for i in issues if i["severity"] == "warning"),
            "n_info": sum(1 for i in issues if i["severity"] == "info"),
            "issues": issues,
            "missing_by_column": {
                v.variable: {"n_missing": v.n_missing, "pct": v.pct_missing, "status": v.status.value}
                for v in missingness.variables
            },
            "summary": self._build_summary(df, issues),
            # Phase 2D additions
            "missingness": missingness.model_dump(),
            "missing_data_strategies": [s.model_dump() for s in missing_strategies],
            "outlier_summary": [o.model_dump() for o in outlier_results],
            "outlier_strategies": [s.model_dump() for s in outlier_strategies],
            "sensitivity_variants": sensitivity_variants,
            "transformation_candidates": [t.model_dump() for t in transformation_candidates],
            "profile": profile.model_dump(),
        }

        severity = "high"
        if report["n_critical"] > 0:
            severity = "critical"
        elif report["n_warning"] > 5:
            severity = "moderate"

        return AgentResult(
            success=True,
            data=report,
            warnings=[i["description"] for i in issues if i["severity"] == "critical"],
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

    def _detect_duplicates(self, df: pd.DataFrame) -> dict:
        n_dup = int(df.duplicated().sum())
        issues = []
        if n_dup > 0:
            pct = round(n_dup / len(df) * 100, 2)
            severity = "warning" if pct > 1 else "info"
            issues.append({
                "issue_type": "duplicate",
                "severity": severity,
                "column": None,
                "description": f"{n_dup} duplicate rows detected ({pct}%)",
                "n_affected": n_dup,
                "recommendation": "Investigate whether duplicates are data-entry errors or legitimate repeated measures.",
            })
        return {"n_duplicates": n_dup, "issues": issues}

    def _check_validity(self, df: pd.DataFrame) -> dict:
        issues = []
        for col in df.columns:
            lower = col.lower()
            series = df[col].dropna()

            if not pd.api.types.is_numeric_dtype(series):
                continue

            if "age" in lower:
                if (series < 0).any():
                    issues.append({
                        "issue_type": "invalid",
                        "severity": "critical",
                        "column": col,
                        "description": f"Negative values found in '{col}'",
                        "n_affected": int((series < 0).sum()),
                        "recommendation": "Negative ages are impossible. Check data entry.",
                    })
                if (series > 150).any():
                    issues.append({
                        "issue_type": "invalid",
                        "severity": "warning",
                        "column": col,
                        "description": f"Implausibly high values in '{col}' (>{150})",
                        "n_affected": int((series > 150).sum()),
                        "recommendation": "Values >150 may be data entry errors.",
                    })

            if any(p in lower for p in ["pct", "percent", "rate", "proportion"]):
                if (series < 0).any():
                    issues.append({
                        "issue_type": "invalid",
                        "severity": "warning",
                        "column": col,
                        "description": f"Negative values in percentage column '{col}'",
                        "n_affected": int((series < 0).sum()),
                    })
                if (series > 100).any():
                    issues.append({
                        "issue_type": "invalid",
                        "severity": "warning",
                        "column": col,
                        "description": f"Values >100 in percentage column '{col}'",
                        "n_affected": int((series > 100).sum()),
                    })

        return {"issues": issues}

    def _check_constant_variables(self, df: pd.DataFrame) -> dict:
        issues = []
        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                issues.append({
                    "issue_type": "constant",
                    "severity": "warning",
                    "column": col,
                    "description": f"Variable '{col}' has only {n_unique} unique value(s) — constant or near-constant",
                    "recommendation": "Consider dropping. Provides no information for analysis.",
                })
        return {"issues": issues}

    def _build_summary(self, df: pd.DataFrame, issues: list) -> str:
        parts = []
        parts.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        n_critical = sum(1 for i in issues if i["severity"] == "critical")
        n_warning = sum(1 for i in issues if i["severity"] == "warning")
        if n_critical > 0:
            parts.append(f"{n_critical} critical issues require attention.")
        if n_warning > 0:
            parts.append(f"{n_warning} warnings should be reviewed.")
        if not issues:
            parts.append("No significant data quality issues detected.")
        return " ".join(parts)
