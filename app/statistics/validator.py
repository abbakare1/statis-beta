"""Statistical Validator — validates analysis specifications before execution.

The validator checks whether a proposed analysis is defensible given the data,
method, and study design. It does NOT trust the AI's recommendation blindly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    ValidationStatus,
    AnalysisAssumption,
    AnalysisDiagnosis,
)
from app.statistics.registry import get_method, MethodEntry


class ValidationResult:
    """Structured output from the validator."""

    def __init__(self):
        self.status = ValidationStatus.PENDING
        self.warnings: list[str] = []
        self.blockers: list[str] = []
        self.checks: list[AnalysisDiagnosis] = []
        self.recommendations: list[str] = []

    def add_pass(self, name: str, detail: str, value: Any = None):
        self.checks.append(AnalysisDiagnosis(
            check_name=name, status="pass", description=detail, value=value,
        ))

    def add_warning(self, name: str, detail: str, value: Any = None, threshold: Any = None):
        self.warnings.append(f"{name}: {detail}")
        self.checks.append(AnalysisDiagnosis(
            check_name=name, status="warning", description=detail,
            value=value, threshold=threshold,
        ))

    def add_block(self, name: str, detail: str):
        self.blockers.append(f"{name}: {detail}")
        self.checks.append(AnalysisDiagnosis(
            check_name=name, status="fail", description=detail,
        ))

    def add_recommendation(self, text: str):
        self.recommendations.append(text)

    def finalize(self) -> ValidationStatus:
        if self.blockers:
            self.status = ValidationStatus.BLOCK
        elif self.warnings:
            self.status = ValidationStatus.WARNING
        else:
            self.status = ValidationStatus.PASS
        return self.status


class StatisticalValidator:
    """Validates a StatisticalAnalysisSpecification against data and method requirements."""

    def validate(
        self,
        spec: StatisticalAnalysisSpecification,
        file_path: Path | str | None = None,
        data_frame: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """Run all validation checks and return a ValidationResult."""
        result = ValidationResult()

        # Load data if needed
        df = data_frame
        if df is None and file_path:
            df = self._load(file_path)
        if df is None:
            result.add_block("data_missing", "No dataset provided for validation.")
            result.finalize()
            return result

        # 1. Variable existence checks
        self._check_variables_exist(spec, df, result)

        # 2. Variable type compatibility
        if not result.blockers:
            self._check_variable_types(spec, df, result)

        # 3. Sample size adequacy
        if not result.blockers:
            self._check_sample_size(spec, df, result)

        # 4. Method compatibility
        if not result.blockers:
            self._check_method_compatibility(spec, df, result)

        # 5. Missing data check
        if not result.blockers:
            self._check_missing_data(spec, df, result)

        # 6. Separation / perfect prediction (logistic regression)
        if not result.blockers:
            self._check_separation(spec, df, result)

        # 7. Multicollinearity (regression models)
        if not result.blockers:
            self._check_multicollinearity(spec, df, result)

        # 8. Constant variables
        if not result.blockers:
            self._check_constant_variables(spec, df, result)

        result.finalize()

        # Update spec
        spec.validation_status = result.status
        spec.validation_warnings = result.warnings
        spec.validation_blockers = result.blockers
        spec.diagnostics = result.checks

        return result

    def _load(self, path: Path | str) -> pd.DataFrame | None:
        try:
            p = Path(path)
            if p.suffix == ".csv":
                return pd.read_csv(p)
            elif p.suffix == ".tsv":
                return pd.read_csv(p, sep="\t")
            elif p.suffix in (".xlsx", ".xls"):
                return pd.read_excel(p)
            elif p.suffix == ".parquet":
                return pd.read_parquet(p)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
        return None

    def _check_variables_exist(self, spec, df, result):
        """Check that all specified variables exist in the dataset."""
        all_vars = [spec.primary_outcome]
        all_vars.extend(spec.predictors)
        all_vars.extend(spec.covariates)
        all_vars.extend(spec.subgroup_variables)

        df_cols = set(df.columns)
        for var in all_vars:
            if var not in df_cols:
                result.add_block("variable_not_found", f"Variable '{var}' not found in dataset.")

        if not result.blockers:
            result.add_pass("variables_exist", f"All {len(all_vars)} specified variables found.")

    def _check_variable_types(self, spec, df, result):
        """Check that outcome/predictor types are compatible with the method."""
        method_entry = get_method(spec.analysis_method)
        if method_entry is None:
            result.add_warning("unknown_method", f"Method '{spec.analysis_method}' not in registry.")
            return

        # Check outcome type
        outcome_col = df[spec.primary_outcome]
        if pd.api.types.is_numeric_dtype(outcome_col):
            if outcome_col.nunique() <= 2:
                outcome_type = "binary"
            else:
                outcome_type = "continuous"
        elif outcome_col.dtype == object:
            if outcome_col.nunique() <= 10:
                outcome_type = "categorical"
            else:
                outcome_type = "continuous"  # treat high-cardinality text as error
        else:
            outcome_type = "continuous"

        if "any" not in method_entry.outcome_types and outcome_type not in method_entry.outcome_types:
            result.add_warning(
                "outcome_type_mismatch",
                f"Outcome '{spec.primary_outcome}' appears {outcome_type}, "
                f"but {method_entry.name} expects {method_entry.outcome_types}.",
            )
        else:
            result.add_pass("outcome_type", f"Outcome type '{outcome_type}' is compatible.")

        # Check predictor types
        for pred in spec.predictors:
            if pred not in df.columns:
                continue
            pred_col = df[pred]
            if pd.api.types.is_numeric_dtype(pred_col):
                pred_type = "continuous"
            elif pred_col.nunique() <= 2:
                pred_type = "binary"
            else:
                pred_type = "categorical"

            if "any" not in method_entry.predictor_types and pred_type not in method_entry.predictor_types:
                result.add_warning(
                    "predictor_type_mismatch",
                    f"Predictor '{pred}' appears {pred_type}, "
                    f"but {method_entry.name} expects {method_entry.predictor_types}.",
                )

    def _check_sample_size(self, spec, df, result):
        """Check minimum sample size for the method."""
        method_entry = get_method(spec.analysis_method)
        if method_entry is None:
            return

        n = len(df)
        min_n = method_entry.min_observations

        if n < min_n:
            result.add_block(
                "insufficient_sample_size",
                f"Dataset has {n} observations but {method_entry.name} requires at least {min_n}.",
            )
        elif n < min_n * 2:
            result.add_warning(
                "small_sample_size",
                f"Dataset has {n} observations. {method_entry.name} may be underpowered. "
                f"Recommended minimum: {min_n * 2}.",
            )
        else:
            result.add_pass("sample_size", f"Sample size ({n}) is adequate for {method_entry.name}.")

    def _check_method_compatibility(self, spec, df, result):
        """Check that the method supports the study design."""
        method_entry = get_method(spec.analysis_method)
        if method_entry is None:
            return

        if spec.study_design and "any" not in method_entry.study_designs:
            if spec.study_design.lower().replace(" ", "_") not in method_entry.study_designs:
                result.add_warning(
                    "design_compatibility",
                    f"Study design '{spec.study_design}' may not be ideal for {method_entry.name}. "
                    f"Compatible designs: {method_entry.study_designs}.",
                )
            else:
                result.add_pass("design_compatible", f"Design '{spec.study_design}' is compatible.")

    def _check_missing_data(self, spec, df, result):
        """Check missing data in key variables."""
        all_vars = [spec.primary_outcome] + spec.predictors + spec.covariates
        total_n = len(df)

        for var in all_vars:
            if var not in df.columns:
                continue
            n_missing = int(df[var].isna().sum())
            pct_missing = n_missing / total_n * 100

            if pct_missing > 50:
                result.add_block(
                    "extreme_missing",
                    f"Variable '{var}' has {pct_missing:.1f}% missing values (>50%).",
                )
            elif pct_missing > 20:
                result.add_warning(
                    "high_missingness",
                    f"Variable '{var}' has {pct_missing:.1f}% missing values.",
                )
            elif n_missing > 0:
                result.add_warning(
                    "some_missingness",
                    f"Variable '{var}' has {n_missing} missing values ({pct_missing:.1f}%).",
                )

        # Complete-case observations
        complete_vars = [v for v in all_vars if v in df.columns]
        n_complete = df[complete_vars].dropna().shape[0] if complete_vars else total_n
        complete_pct = n_complete / total_n * 100 if total_n > 0 else 0

        if complete_pct < 70:
            result.add_warning(
                "low_complete_cases",
                f"Only {complete_pct:.1f}% of observations have complete data for all variables.",
            )
        else:
            result.add_pass("missing_data", f"{complete_pct:.1f}% complete cases available.")

    def _check_separation(self, spec, df, result):
        """Check for complete separation in logistic regression."""
        if spec.analysis_method != "logistic_regression":
            return

        outcome = spec.primary_outcome
        if outcome not in df.columns:
            return

        for pred in spec.predictors:
            if pred not in df.columns:
                continue
            if df[pred].nunique() > 2:
                continue  # skip continuous predictors for this check

            # Check if predictor perfectly predicts outcome
            ct = pd.crosstab(df[pred], df[outcome])
            if ct.shape[0] <= 1 or ct.shape[1] <= 1:
                continue
            if (ct == 0).any().any():
                result.add_warning(
                    "separation_detected",
                    f"Potential complete separation between '{pred}' and outcome '{outcome}'. "
                    "Logistic regression may not converge.",
                )

    def _check_multicollinearity(self, spec, df, result):
        """Check for multicollinearity among numeric predictors."""
        if spec.analysis_method not in ("linear_regression", "logistic_regression"):
            return

        numeric_vars = [v for v in spec.predictors + spec.covariates if v in df.columns]
        numeric_vars = [v for v in numeric_vars if pd.api.types.is_numeric_dtype(df[v])]

        if len(numeric_vars) < 2:
            return

        # Compute correlation matrix
        corr_matrix = df[numeric_vars].corr()
        high_corr_pairs = []

        for i in range(len(numeric_vars)):
            for j in range(i + 1, len(numeric_vars)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.9:
                    high_corr_pairs.append((numeric_vars[i], numeric_vars[j], corr))

        if high_corr_pairs:
            for v1, v2, c in high_corr_pairs:
                result.add_warning(
                    "high_correlation",
                    f"Predictors '{v1}' and '{v2}' are highly correlated (r={c:.3f}). "
                    "Consider removing one to reduce multicollinearity.",
                )
        else:
            result.add_pass("multicollinearity", "No high correlations among predictors detected.")

    def _check_constant_variables(self, spec, df, result):
        """Check for constant or near-constant variables."""
        all_vars = spec.predictors + spec.covariates

        for var in all_vars:
            if var not in df.columns:
                continue
            n_unique = df[var].nunique(dropna=True)
            if n_unique <= 1:
                result.add_block(
                    "constant_variable",
                    f"Variable '{var}' has only {n_unique} unique value(s) — no variation to model.",
                )
