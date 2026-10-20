"""Table Renderer — deterministic publication table engine.

Phase 2G: Converts validated StatisticalResult objects into publication-ready
tables. Never recomputes statistics; consumes persisted results with provenance.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.outputs.specification import TableSpecification, TableType, OutputFormat


class TableRenderer:
    """Deterministic table renderer."""

    def render(
        self,
        spec: TableSpecification,
        *,
        df: pd.DataFrame | None = None,
        results: list[dict[str, Any]] | None = None,
        output_dir: Path,
    ) -> Path:
        """Render a table based on specification.

        Returns the path to the generated artifact.
        Raises ValueError on invalid specifications or missing data.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dispatch to specific renderer
        table_data = self._render_table(spec, df=df, results=results)

        if table_data is None:
            raise ValueError(
                f"Failed to render table type '{spec.table_type.value}'"
            )

        # Save
        output_path = output_dir / f"{spec.output_id}.{spec.output_format.value}"
        self._save_table(table_data, spec, output_path)

        return output_path

    def _render_table(
        self,
        spec: TableSpecification,
        *,
        df: pd.DataFrame | None,
        results: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Dispatch to specific table renderer."""
        renderers = {
            TableType.DESCRIPTIVE_BASELINE: self._render_descriptive_table,
            TableType.INFERENTIAL_RESULTS: self._render_inferential_table,
            TableType.REGRESSION_TABLE: self._render_regression_table,
            TableType.SURVIVAL_TABLE: self._render_survival_table,
        }

        renderer = renderers.get(spec.table_type)
        if renderer is None:
            return None

        return renderer(spec, df=df, results=results)

    def _render_descriptive_table(
        self,
        spec: TableSpecification,
        *,
        df: pd.DataFrame | None,
        results: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Render Table 1: Baseline/descriptive characteristics."""
        if df is None or not spec.variables:
            return None

        rows = []
        for var in spec.variables:
            if var not in df.columns:
                continue

            series = df[var].dropna()
            n_total = len(df[var])
            n_missing = int(df[var].isna().sum())

            if pd.api.types.is_numeric_dtype(df[var]):
                # Continuous variable
                mean = series.mean()
                sd = series.std(ddof=1) if len(series) > 1 else 0
                median = series.median()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)

                rows.append({
                    "variable": var,
                    "n": n_total,
                    "n_missing": n_missing,
                    "mean_sd": f"{mean:.{spec.decimal_places}f} ± {sd:.{spec.decimal_places}f}",
                    "median_iqr": f"{median:.{spec.decimal_places}f} [{q1:.{spec.decimal_places}f}, {q3:.{spec.decimal_places}f}]",
                    "n_pct": "",
                })
            else:
                # Categorical variable
                value_counts = series.value_counts()
                n_pct_parts = []
                for val, count in value_counts.items():
                    pct = count / n_total * 100
                    n_pct_parts.append(f"{val}: {count} ({pct:.1f}%)")

                rows.append({
                    "variable": var,
                    "n": n_total,
                    "n_missing": n_missing,
                    "mean_sd": "",
                    "median_iqr": "",
                    "n_pct": "; ".join(n_pct_parts),
                })

        return {
            "columns": ["Variable", "N", "Missing N", "Mean ± SD", "Median [IQR]", "n (%)"],
            "rows": rows,
            "title": spec.title or "Table 1. Baseline Characteristics",
        }

    def _render_inferential_table(
        self,
        spec: TableSpecification,
        *,
        df: pd.DataFrame | None,
        results: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Render inferential results table."""
        if not results:
            return None

        rows = []
        for result in results:
            method_id = result.get("method_id", "")
            eng = result.get("engine_result", {})

            row = {
                "method": method_id,
                "test_statistic": self._format_number(result.get("test_statistic"), spec.decimal_places),
                "df": self._format_number(result.get("degrees_of_freedom"), spec.decimal_places),
                "p_value": self._format_p_value(result.get("p_value")),
                "effect_size": self._format_number(result.get("effect_size"), spec.decimal_places),
                "effect_measure": result.get("effect_size_name", ""),
                "n": result.get("n_observed", ""),
            }

            # Add CI if available
            if spec.include_ci and result.get("confidence_interval"):
                ci = result["confidence_interval"]
                row["ci_95"] = f"[{ci[0]:.{spec.decimal_places}f}, {ci[1]:.{spec.decimal_places}f}]"
            else:
                row["ci_95"] = ""

            rows.append(row)

        columns = ["Method", "Test Statistic", "df", "P value", "Effect Size",
                   "Effect Measure", "95% CI", "N"]

        return {
            "columns": columns,
            "rows": rows,
            "title": spec.title or "Inferential Results",
        }

    def _render_regression_table(
        self,
        spec: TableSpecification,
        *,
        df: pd.DataFrame | None,
        results: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Render regression results table."""
        if not results:
            return None

        rows = []
        for result in results:
            method_id = result.get("method_id", "")
            eng = result.get("engine_result", {})

            coefficients = eng.get("coefficients", {})
            conf_int = eng.get("confidence_intervals", {})
            p_values = eng.get("p_values", {})
            standard_errors = eng.get("standard_errors", {})

            for covariate, coef in coefficients.items():
                if covariate == "Intercept" and not spec.include_intercept:
                    continue

                ci = conf_int.get(covariate, (None, None))
                se = standard_errors.get(covariate)
                p = p_values.get(covariate)

                # Determine effect measure label
                if method_id == "logistic_regression":
                    effect_label = "OR"
                    effect_value = eng.get("odds_ratios", {}).get(covariate)
                elif method_id == "cox_proportional_hazards":
                    effect_label = "HR"
                    effect_value = eng.get("hazard_ratios", {}).get(covariate)
                else:
                    effect_label = "β"
                    effect_value = coef

                rows.append({
                    "covariate": covariate,
                    "estimate": self._format_number(effect_value, spec.decimal_places),
                    "effect_label": effect_label,
                    "std_error": self._format_number(se, spec.decimal_places),
                    "p_value": self._format_p_value(p),
                    "ci_95": f"[{ci[0]:.{spec.decimal_places}f}, {ci[1]:.{spec.decimal_places}f}]" if ci[0] is not None else "",
                    "n": result.get("n_observed", ""),
                })

        columns = ["Variable", "Estimate", "Measure", "Std. Error", "P value", "95% CI", "N"]

        return {
            "columns": columns,
            "rows": rows,
            "title": spec.title or "Regression Results",
        }

    def _render_survival_table(
        self,
        spec: TableSpecification,
        *,
        df: pd.DataFrame | None,
        results: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Render survival analysis results table."""
        if not results:
            return None

        rows = []
        for result in results:
            method_id = result.get("method_id", "")
            eng = result.get("engine_result", {})

            if method_id == "kaplan_meier":
                groups = eng.get("groups", {})
                overall = groups.get("overall", groups)

                rows.append({
                    "method": "Kaplan-Meier",
                    "n": eng.get("n_total", result.get("n_observed", "")),
                    "events": eng.get("n_events", ""),
                    "censored": eng.get("n_censored", ""),
                    "median_survival": self._format_number(overall.get("median_survival"), spec.decimal_places),
                    "effect_measure": "",
                    "estimate": "",
                    "ci_95": "",
                    "p_value": "",
                })

            elif method_id == "log_rank":
                rows.append({
                    "method": "Log-rank",
                    "n": result.get("n_observed", ""),
                    "events": eng.get("n_events", ""),
                    "censored": eng.get("n_censored", ""),
                    "median_survival": "",
                    "effect_measure": "χ²",
                    "estimate": self._format_number(result.get("test_statistic"), spec.decimal_places),
                    "ci_95": "",
                    "p_value": self._format_p_value(result.get("p_value")),
                })

            elif method_id == "cox_proportional_hazards":
                coefficients = eng.get("coefficients", {})
                hazard_ratios = eng.get("hazard_ratios", {})
                conf_int = eng.get("confidence_intervals", {})
                p_values = eng.get("p_values", {})

                for covariate, coef in coefficients.items():
                    if covariate == "Intercept":
                        continue
                    hr = hazard_ratios.get(covariate)
                    ci = conf_int.get(covariate, (None, None))
                    p = p_values.get(covariate)

                    rows.append({
                        "method": f"Cox PH: {covariate}",
                        "n": eng.get("n_obs", result.get("n_observed", "")),
                        "events": eng.get("n_events", ""),
                        "censored": "",
                        "median_survival": "",
                        "effect_measure": "HR",
                        "estimate": self._format_number(hr, spec.decimal_places),
                        "ci_95": f"[{ci[0]:.{spec.decimal_places}f}, {ci[1]:.{spec.decimal_places}f}]" if ci[0] is not None else "",
                        "p_value": self._format_p_value(p),
                    })

        columns = ["Method", "N", "Events", "Censored", "Median Survival",
                   "Measure", "Estimate", "95% CI", "P value"]

        return {
            "columns": columns,
            "rows": rows,
            "title": spec.title or "Survival Analysis Results",
        }

    def _format_number(self, value: Any, decimals: int) -> str:
        """Format a number to specified decimal places."""
        if value is None:
            return ""
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    def _format_p_value(self, p_value: Any) -> str:
        """Format a p-value."""
        if p_value is None:
            return ""
        try:
            p = float(p_value)
            if p < 0.001:
                return "<0.001"
            return f"{p:.3f}"
        except (ValueError, TypeError):
            return str(p_value)

    def _save_table(
        self,
        table_data: dict[str, Any],
        spec: TableSpecification,
        output_path: Path,
    ) -> None:
        """Save table data to file."""
        if spec.output_format == OutputFormat.CSV:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(table_data["columns"])
                for row in table_data["rows"]:
                    writer.writerow([row.get(col.lower().replace(" ", "_"), "") for col in table_data["columns"]])

        elif spec.output_format == OutputFormat.JSON:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(table_data, f, indent=2)

        elif spec.output_format == OutputFormat.MARKDOWN:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# {table_data['title']}\n\n")
                # Write header
                header = "| " + " | ".join(table_data["columns"]) + " |"
                separator = "| " + " | ".join(["---"] * len(table_data["columns"])) + " |"
                f.write(header + "\n")
                f.write(separator + "\n")
                # Write rows
                for row in table_data["rows"]:
                    values = []
                    for col in table_data["columns"]:
                        key = col.lower().replace(" ", "_")
                        values.append(str(row.get(key, "")))
                    f.write("| " + " | ".join(values) + " | \n")

        elif spec.output_format == OutputFormat.HTML:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"<h2>{table_data['title']}</h2>\n")
                f.write("<table border='1' cellpadding='5' cellspacing='0'>\n")
                f.write("<tr>")
                for col in table_data["columns"]:
                    f.write(f"<th>{col}</th>")
                f.write("</tr>\n")
                for row in table_data["rows"]:
                    f.write("<tr>")
                    for col in table_data["columns"]:
                        key = col.lower().replace(" ", "_")
                        f.write(f"<td>{row.get(key, '')}</td>")
                    f.write("</tr>\n")
                f.write("</table>\n")

        else:
            # Default: JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(table_data, f, indent=2)
