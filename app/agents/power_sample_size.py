"""Power and Sample-Size Agent — evaluate dataset adequacy for proposed analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from app.agents.base import BaseAgent, AgentResult
from app.statistics.engine import power_analysis_t_test, _load_dataframe


class PowerSampleSizeAgent(BaseAgent):
    name = "power_sample_size"
    requires_llm = False
    description = "Evaluate whether the dataset is adequate for the proposed analysis."

    async def _execute(self, context: dict) -> AgentResult:
        file_path = Path(context["file_path"])
        analysis_type = context.get("analysis_type", "t_test")
        outcome = context.get("outcome")
        group = context.get("group")
        alpha = context.get("alpha", 0.05)
        target_power = context.get("target_power", 0.80)

        df = _load_dataframe(file_path)
        n_total = len(df)

        report = {
            "n_total": n_total,
            "analysis_type": analysis_type,
            "alpha": alpha,
            "target_power": target_power,
        }

        if analysis_type in ("t_test", "independent_t_test") and outcome and group:
            groups = df[group].dropna().unique()
            if len(groups) >= 2:
                g1 = df.loc[df[group] == groups[0], outcome].dropna()
                g2 = df.loc[df[group] == groups[1], outcome].dropna()

                n1, n2 = len(g1), len(g2)
                report["n_group1"] = n1
                report["n_group2"] = n2

                # Observed effect size (Cohen's d)
                if n1 > 1 and n2 > 1:
                    pooled_std = np.sqrt(
                        ((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2)
                        / (n1 + n2 - 2)
                    )
                    if pooled_std > 0:
                        observed_d = abs(g1.mean() - g2.mean()) / pooled_std
                        report["observed_cohens_d"] = round(float(observed_d), 4)
                        report["effect_size_interpretation"] = _interpret_effect_size(observed_d)

                        # Post-hoc power
                        posthoc = power_analysis_t_test(
                            effect_size=observed_d,
                            n_per_group=min(n1, n2),
                            alpha=alpha,
                        )
                        report["posthoc_power"] = posthoc.get("power")

                        # Minimum detectable effect at current N
                        mde = power_analysis_t_test(
                            n_per_group=min(n1, n2),
                            alpha=alpha,
                            power=target_power,
                        )
                        report["min_detectable_effect"] = mde.get("min_detectable_effect_size")

                        # Required N for observed effect
                        req_n = power_analysis_t_test(
                            effect_size=observed_d,
                            alpha=alpha,
                            power=target_power,
                        )
                        report["required_n_per_group_for_observed_effect"] = req_n.get("required_n_per_group")
                        report["required_total_n_for_observed_effect"] = req_n.get("total_required_n")

                        # Classification
                        if posthoc.get("power", 0) >= target_power:
                            report["adequacy"] = "adequate"
                            report["adequacy_message"] = f"Dataset appears adequate for detecting the observed effect (power={posthoc.get('power', 0):.3f})."
                        else:
                            report["adequacy"] = "potentially_inadequate"
                            report["adequacy_message"] = (
                                f"Dataset may be underpowered for the observed effect size "
                                f"(power={posthoc.get('power', 0):.3f} vs target {target_power}). "
                                f"Minimum detectable effect at current N: {mde.get('min_detectable_effect_size', 'N/A')}."
                            )

                        report["caveat"] = (
                            "Post-hoc power is a descriptive statistic of this dataset, not a definitive "
                            "measure of whether the study 'should have' detected an effect. Interpret cautiously."
                        )

        elif analysis_type == "linear_regression" and outcome:
            n_predictors = len(context.get("predictors", []))
            report["n_predictors"] = n_predictors
            if n_predictors > 0:
                # Rule of thumb: 10-20 observations per predictor
                report["rule_of_thumb_min_n"] = n_predictors * 15
                report["rule_of_thumb_adequate"] = n_total >= report["rule_of_thumb_min_n"]
                report["adequacy"] = "adequate" if report["rule_of_thumb_adequate"] else "potentially_inadequate"
                report["adequacy_message"] = (
                    f"Rule of thumb: {n_predictors} predictors need ~{report['rule_of_thumb_min_n']} observations. "
                    f"Available: {n_total}."
                )

        elif analysis_type == "logistic_regression" and outcome:
            n_predictors = len(context.get("predictors", []))
            event_rate = df[outcome].mean() if pd.api.types.is_numeric_dtype(df[outcome]) else None
            report["n_predictors"] = n_predictors
            if event_rate is not None:
                n_events = int(df[outcome].sum())
                report["event_rate"] = round(float(event_rate), 4)
                report["n_events"] = n_events
                # EPV >= 10 rule of thumb
                report["events_per_variable"] = round(n_events / n_predictors, 1) if n_predictors > 0 else None
                if report["events_per_variable"] and report["events_per_variable"] >= 10:
                    report["adequacy"] = "adequate"
                    report["adequacy_message"] = f"Events per variable = {report['events_per_variable']:.1f} (>= 10 recommended)."
                elif report["events_per_variable"]:
                    report["adequacy"] = "potentially_inadequate"
                    report["adequacy_message"] = f"Events per variable = {report['events_per_variable']:.1f} (< 10 recommended). Model may be unstable."

        else:
            report["adequacy"] = "insufficient_information"
            report["adequacy_message"] = "Insufficient information to perform power assessment."

        return AgentResult(
            success=True,
            data=report,
            confidence="high" if report.get("adequacy") != "insufficient_information" else "low",
        )


def _interpret_effect_size(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"
