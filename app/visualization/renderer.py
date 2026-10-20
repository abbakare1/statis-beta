"""Figure Renderer — deterministic visualization engine.

Phase 2G: Converts validated StatisticalResult objects into publication-quality
figures using matplotlib. Never recomputes statistics; consumes persisted results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from app.outputs.specification import FigureSpecification, FigureType, OutputFormat


# ---------------------------------------------------------------------------
# Format mapping
# ---------------------------------------------------------------------------

_FORMAT_MAP = {
    OutputFormat.PNG: "png",
    OutputFormat.SVG: "svg",
    OutputFormat.PDF: "pdf",
}


def _get_matplotlib_format(fmt: OutputFormat) -> str:
    return _FORMAT_MAP.get(fmt, "png")


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class FigureRenderer:
    """Deterministic figure renderer using matplotlib."""

    def render(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None = None,
        result: dict[str, Any] | None = None,
        output_dir: Path,
    ) -> Path:
        """Render a figure based on specification.

        Returns the path to the generated artifact.
        Raises ValueError on invalid specifications or missing data.
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set deterministic matplotlib state
        self._setup_matplotlib()

        # Dispatch to specific renderer
        fig = self._render_figure(spec, df=df, result=result)

        if fig is None:
            raise ValueError(
                f"Failed to render figure type '{spec.figure_type.value}'"
            )

        # Apply common styling
        self._apply_common_style(fig, spec)

        # Save
        ext = _get_matplotlib_format(spec.output_format)
        output_path = output_dir / f"{spec.output_id}.{ext}"
        fig.savefig(
            output_path,
            dpi=spec.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        return output_path

    def _setup_matplotlib(self) -> None:
        """Set deterministic matplotlib configuration."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })

    def _apply_common_style(self, fig: plt.Figure, spec: FigureSpecification) -> None:
        """Apply common styling to figure."""
        if spec.title:
            fig.suptitle(spec.title, fontsize=12, fontweight="bold", y=1.02)

        if spec.x_label:
            for ax in fig.axes:
                if ax.get_xlabel() == "" or ax.get_xlabel() is None:
                    ax.set_xlabel(spec.x_label)

        if spec.y_label:
            for ax in fig.axes:
                if ax.get_ylabel() == "" or ax.get_ylabel() is None:
                    ax.set_ylabel(spec.y_label)

        if spec.show_grid:
            for ax in fig.axes:
                ax.grid(True, alpha=0.3)

    def _render_figure(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        """Dispatch to specific figure renderer."""
        renderers = {
            FigureType.HISTOGRAM: self._render_histogram,
            FigureType.BOX_PLOT: self._render_box_plot,
            FigureType.SCATTER_PLOT: self._render_scatter_plot,
            FigureType.BAR_CHART: self._render_bar_chart,
            FigureType.GROUPED_BOX_PLOT: self._render_grouped_box_plot,
            FigureType.MEAN_CI_PLOT: self._render_mean_ci_plot,
            FigureType.COEFFICIENT_PLOT: self._render_coefficient_plot,
            FigureType.KAPLAN_MEIER: self._render_kaplan_meier,
            FigureType.GROUPED_KAPLAN_MEIER: self._render_grouped_kaplan_meier,
            FigureType.RESIDUAL_VS_FITTED: self._render_residual_vs_fitted,
            FigureType.QQ_PLOT: self._render_qq_plot,
        }

        renderer = renderers.get(spec.figure_type)
        if renderer is None:
            return None

        return renderer(spec, df=df, result=result)

    # -----------------------------------------------------------------------
    # Descriptive figures
    # -----------------------------------------------------------------------

    def _render_histogram(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if df is None or spec.variable is None or spec.variable not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))
        data = df[spec.variable].dropna()
        ax.hist(data, bins=30, edgecolor="white", alpha=0.8, color="#4C72B0")
        ax.set_xlabel(spec.x_label or spec.variable)
        ax.set_ylabel(spec.y_label or "Frequency")
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_box_plot(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if df is None or spec.variable is None or spec.variable not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))
        data = df[spec.variable].dropna()
        bp = ax.boxplot(
            [data.values],
            tick_labels=[spec.variable],
            patch_artist=True,
            boxprops=dict(facecolor="#4C72B0", alpha=0.7),
            medianprops=dict(color="darkred", linewidth=2),
        )
        ax.set_ylabel(spec.y_label or spec.variable)
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_scatter_plot(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if df is None:
            return None

        x_var = spec.x_variable or spec.variable
        y_var = spec.y_variable

        if x_var is None or y_var is None:
            return None
        if x_var not in df.columns or y_var not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))
        x_data = df[x_var].dropna()
        y_data = df[y_var].dropna()

        # Align data
        mask = df[x_var].notna() & df[y_var].notna()
        x_aligned = df.loc[mask, x_var].values
        y_aligned = df.loc[mask, y_var].values

        ax.scatter(x_aligned, y_aligned, alpha=0.6, s=30, color="#4C72B0")

        # Add regression line if we have a result
        if result and "engine_result" in result:
            eng = result["engine_result"]
            if "coefficients" in eng and "Intercept" in eng["coefficients"]:
                intercept = eng["coefficients"]["Intercept"]
                slope = eng["coefficients"].get(x_var)
                if slope is not None:
                    x_line = np.linspace(x_aligned.min(), x_aligned.max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, "r-", linewidth=2, label="Regression line")
                    ax.legend()

        ax.set_xlabel(spec.x_label or x_var)
        ax.set_ylabel(spec.y_label or y_var)
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_bar_chart(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if df is None or spec.variable is None or spec.variable not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))
        counts = df[spec.variable].value_counts()
        if spec.category_order:
            counts = counts.reindex(spec.category_order).dropna()

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
        ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)], alpha=0.8)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_xlabel(spec.x_label or spec.variable)
        ax.set_ylabel(spec.y_label or "Count")
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_grouped_box_plot(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if df is None or spec.variable is None or spec.grouping_variable is None:
            return None
        if spec.variable not in df.columns or spec.grouping_variable not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))

        groups = df[spec.grouping_variable].dropna().unique()
        if spec.category_order:
            groups = [g for g in spec.category_order if g in groups]

        data_by_group = [
            df.loc[df[spec.grouping_variable] == g, spec.variable].dropna().values
            for g in groups
        ]

        bp = ax.boxplot(
            data_by_group,
            tick_labels=[str(g) for g in groups],
            patch_artist=True,
            medianprops=dict(color="darkred", linewidth=2),
        )

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel(spec.x_label or spec.grouping_variable)
        ax.set_ylabel(spec.y_label or spec.variable)
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_mean_ci_plot(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if df is None or spec.grouping_variable is None or spec.variable is None:
            return None
        if spec.grouping_variable not in df.columns or spec.variable not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))

        groups = df[spec.grouping_variable].dropna().unique()
        if spec.category_order:
            groups = [g for g in spec.category_order if g in groups]

        means = []
        ci_lower = []
        ci_upper = []

        for g in groups:
            data = df.loc[df[spec.grouping_variable] == g, spec.variable].dropna()
            if len(data) >= 2:
                m = data.mean()
                se = data.std(ddof=1) / np.sqrt(len(data))
                # 95% CI using normal approximation
                means.append(m)
                ci_lower.append(m - 1.96 * se)
                ci_upper.append(m + 1.96 * se)
            else:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        x_pos = range(len(groups))
        ax.errorbar(
            x_pos, means,
            yerr=[np.array(means) - np.array(ci_lower), np.array(ci_upper) - np.array(means)],
            fmt="o-", capsize=5, capthick=2, linewidth=2, color="#4C72B0",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlabel(spec.x_label or spec.grouping_variable)
        ax.set_ylabel(spec.y_label or spec.variable)
        if spec.title:
            ax.set_title(spec.title)

        return fig

    # -----------------------------------------------------------------------
    # Regression figures
    # -----------------------------------------------------------------------

    def _render_coefficient_plot(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if result is None or "engine_result" not in result:
            return None

        eng = result["engine_result"]
        coefficients = eng.get("coefficients", {})
        conf_int = eng.get("confidence_intervals", {})

        if not coefficients:
            return None

        # Filter out intercept
        covariates = [k for k in coefficients if k != "Intercept"]
        if not covariates:
            covariates = list(coefficients.keys())

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))

        y_pos = range(len(covariates))
        coefs = [coefficients.get(c, 0) for c in covariates]
        ci_lowers = [conf_int.get(c, (0, 0))[0] for c in covariates]
        ci_uppers = [conf_int.get(c, (0, 0))[1] for c in covariates]

        errors = [
            [c - l for c, l in zip(coefs, ci_lowers)],
            [u - c for c, u in zip(coefs, ci_uppers)],
        ]

        ax.errorbar(
            coefs, y_pos,
            xerr=errors,
            fmt="o", capsize=5, capthick=2, linewidth=2, color="#4C72B0",
        )
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(covariates)
        ax.set_xlabel(spec.x_label or "Coefficient")
        if spec.title:
            ax.set_title(spec.title)

        return fig

    # -----------------------------------------------------------------------
    # Survival figures
    # -----------------------------------------------------------------------

    def _render_kaplan_meier(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if result is None or "engine_result" not in result:
            return None

        eng = result["engine_result"]
        km_data = eng.get("groups", {}).get("overall", eng.get("groups", {}))

        if "survival_times" not in km_data or "survival_probabilities" not in km_data:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))

        times = km_data["survival_times"]
        surv = km_data["survival_probabilities"]

        # Step plot for KM curve
        ax.step(times, surv, where="post", linewidth=2, color="#4C72B0")

        # Confidence bands if available
        if "confidence_lower" in km_data and "confidence_upper" in km_data:
            ax.fill_between(
                times,
                km_data["confidence_lower"],
                km_data["confidence_upper"],
                step="post", alpha=0.2, color="#4C72B0",
            )

        # Censoring marks
        if "censoring_times" in km_data:
            censor_times = km_data["censoring_times"]
            for ct in censor_times:
                # Find survival probability at censoring time
                idx = np.searchsorted(times, ct, side="right") - 1
                if 0 <= idx < len(surv):
                    ax.plot(ct, surv[idx], "k|", markersize=10, markeredgewidth=2)

        ax.set_xlabel(spec.x_label or "Time")
        ax.set_ylabel(spec.y_label or "Survival Probability")
        ax.set_ylim(0, 1.05)
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_grouped_kaplan_meier(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if result is None or "engine_result" not in result:
            return None

        eng = result["engine_result"]
        groups = eng.get("groups", {})

        if not groups:
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

        for i, (group_name, group_data) in enumerate(groups.items()):
            if "survival_times" not in group_data or "survival_probabilities" not in group_data:
                continue
            times = group_data["survival_times"]
            surv = group_data["survival_probabilities"]
            ax.step(times, surv, where="post", linewidth=2,
                    color=colors[i % len(colors)], label=str(group_name))

        ax.set_xlabel(spec.x_label or "Time")
        ax.set_ylabel(spec.y_label or "Survival Probability")
        ax.set_ylim(0, 1.05)
        if spec.show_legend:
            ax.legend()
        if spec.title:
            ax.set_title(spec.title)

        return fig

    # -----------------------------------------------------------------------
    # Diagnostic figures
    # -----------------------------------------------------------------------

    def _render_residual_vs_fitted(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if result is None or "engine_result" not in result:
            return None

        eng = result["engine_result"]
        residuals = eng.get("residuals", [])
        fitted = eng.get("fitted_values", [])

        if not residuals or not fitted or len(residuals) != len(fitted):
            return None

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))
        ax.scatter(fitted, residuals, alpha=0.6, s=30, color="#4C72B0")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(spec.x_label or "Fitted Values")
        ax.set_ylabel(spec.y_label or "Residuals")
        if spec.title:
            ax.set_title(spec.title)

        return fig

    def _render_qq_plot(
        self,
        spec: FigureSpecification,
        *,
        df: pd.DataFrame | None,
        result: dict[str, Any] | None,
    ) -> plt.Figure | None:
        if result is None or "engine_result" not in result:
            return None

        eng = result["engine_result"]
        residuals = eng.get("residuals", [])

        if not residuals:
            return None

        from scipy import stats as sp_stats

        fig, ax = plt.subplots(figsize=(spec.width_inches, spec.height_inches))
        (osm, osr), (slope, intercept, r) = sp_stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, alpha=0.6, s=30, color="#4C72B0")
        ax.plot(osm, slope * osm + intercept, "r-", linewidth=2, label="Theoretical")
        ax.set_xlabel(spec.x_label or "Theoretical Quantiles")
        ax.set_ylabel(spec.y_label or "Sample Quantiles")
        if spec.title:
            ax.set_title(spec.title)
        ax.legend()

        return fig
