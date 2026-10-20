"""Statistical Method Registry — structured metadata for every supported method.

The registry defines what each method requires, what it assumes, and when it is appropriate.
The Analysis Planner and Validator query this registry rather than relying on LLM knowledge alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodEntry:
    """Structured metadata for a single statistical method."""
    method_id: str
    name: str
    description: str

    # Compatibility
    outcome_types: list[str]  # continuous, binary, categorical, count, survival, time
    predictor_types: list[str]  # continuous, categorical, binary, mixed
    study_designs: list[str]  # any, rct, cohort, case_control, cross_sectional, etc.
    min_groups: int = 1
    min_observations: int = 2

    # Assumptions
    assumptions: list[str] = field(default_factory=list)
    assumption_tests: list[str] = field(default_factory=list)

    # Effect measures
    effect_measure: str = ""
    confidence_intervals: bool = True

    # Alternatives
    alternative_methods: list[str] = field(default_factory=list)
    sensitivity_methods: list[str] = field(default_factory=list)

    # Warnings
    common_misuse: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    # Implementation
    engine_function: str = ""
    software: str = "scipy/statsmodels"

    # References
    references: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in registry entries
# ---------------------------------------------------------------------------

REGISTRY: dict[str, MethodEntry] = {}


def register_method(entry: MethodEntry):
    REGISTRY[entry.method_id] = entry


def get_method(method_id: str) -> MethodEntry | None:
    return REGISTRY.get(method_id)


def list_methods() -> list[str]:
    return list(REGISTRY.keys())


def find_methods_for_outcome(outcome_type: str) -> list[MethodEntry]:
    return [m for m in REGISTRY.values() if outcome_type in m.outcome_types or "any" in m.outcome_types]


def find_methods_for_design(design: str) -> list[MethodEntry]:
    return [m for m in REGISTRY.values() if design in m.study_designs or "any" in m.study_designs]


# ---------------------------------------------------------------------------
# Register all Phase 1 methods
# ---------------------------------------------------------------------------

register_method(MethodEntry(
    method_id="independent_t_test",
    name="Independent samples t-test",
    description="Compare means between two independent groups.",
    outcome_types=["continuous"],
    predictor_types=["binary"],
    study_designs=["any"],
    min_groups=2,
    min_observations=10,
    assumptions=[
        "Independence of observations",
        "Normality within each group (or large n for CLT)",
        "Approximate homogeneity of variance (Welch corrects for unequal variances)",
    ],
    assumption_tests=["Shapiro-Wilk", "Levene's test"],
    effect_measure="Cohen's d",
    confidence_intervals=True,
    alternative_methods=["Mann-Whitney U", "Permutation test"],
    sensitivity_methods=["Equal variance t-test", "Bootstrap CI"],
    common_misuse=[
        "Used when groups are not independent",
        "Used with small samples without checking normality",
        "Interpreting significance as practical importance",
    ],
    engine_function="independent_t_test",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="paired_t_test",
    name="Paired samples t-test",
    description="Compare means of two related measurements.",
    outcome_types=["continuous"],
    predictor_types=["binary"],
    study_designs=["any"],
    min_groups=2,
    min_observations=5,
    assumptions=[
        "Paired observations are related",
        "Differences are approximately normally distributed",
    ],
    assumption_tests=["Shapiro-Wilk on differences"],
    effect_measure="Mean difference",
    confidence_intervals=True,
    alternative_methods=["Wilcoxon signed-rank"],
    engine_function="paired_t_test",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="chi_square",
    name="Chi-square test of independence",
    description="Test association between two categorical variables.",
    outcome_types=["categorical", "binary"],
    predictor_types=["categorical", "binary"],
    study_designs=["any"],
    min_groups=2,
    min_observations=20,
    assumptions=[
        "Independent observations",
        "Expected cell counts >= 5 (at least 80% of cells)",
        "Adequate total sample size",
    ],
    assumption_tests=["Expected cell count inspection"],
    effect_measure="Cramér's V",
    confidence_intervals=False,
    alternative_methods=["Fisher's exact test", "G-test"],
    common_misuse=[
        "Used when expected cell counts are too low",
        "Used with paired/correlated data",
    ],
    engine_function="chi_square_test",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="fisher_exact",
    name="Fisher's exact test",
    description="Exact test for 2x2 contingency tables.",
    outcome_types=["binary"],
    predictor_types=["binary"],
    study_designs=["any"],
    min_groups=2,
    min_observations=4,
    assumptions=[
        "Independent observations",
        "Fixed marginal totals",
        "2x2 table",
    ],
    effect_measure="Odds ratio",
    confidence_intervals=True,
    alternative_methods=["Chi-square", "G-test"],
    engine_function="fisher_exact_test",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="pearson_correlation",
    name="Pearson correlation",
    description="Measure linear association between two continuous variables.",
    outcome_types=["continuous"],
    predictor_types=["continuous"],
    study_designs=["any"],
    min_observations=10,
    assumptions=[
        "Linear relationship between variables",
        "Bivariate normal distribution",
        "No significant outliers",
        "Homoscedasticity",
    ],
    assumption_tests=["Scatterplot inspection", "Shapiro-Wilk"],
    effect_measure="r (Pearson's r)",
    confidence_intervals=True,
    alternative_methods=["Spearman correlation", "Kendall's tau"],
    engine_function="correlation_test",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="spearman_correlation",
    name="Spearman rank correlation",
    description="Measure monotonic association between two variables.",
    outcome_types=["continuous", "ordinal"],
    predictor_types=["continuous", "ordinal"],
    study_designs=["any"],
    min_observations=10,
    assumptions=[
        "Monotonic relationship",
        "Ordinal or continuous variables",
    ],
    effect_measure="rho (Spearman's rho)",
    confidence_intervals=True,
    alternative_methods=["Pearson correlation", "Kendall's tau"],
    engine_function="correlation_test",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="one_way_anova",
    name="One-way ANOVA",
    description="Compare means across three or more independent groups.",
    outcome_types=["continuous"],
    predictor_types=["categorical"],
    study_designs=["any"],
    min_groups=2,
    min_observations=15,
    assumptions=[
        "Independence of observations",
        "Normality within each group",
        "Homogeneity of variances (homoscedasticity)",
    ],
    assumption_tests=["Shapiro-Wilk", "Levene's test", "Brown-Forsythe"],
    effect_measure="Eta-squared, partial eta-squared",
    confidence_intervals=True,
    alternative_methods=["Kruskal-Wallis", "Welch's ANOVA"],
    sensitivity_methods=["Welch's ANOVA (unequal variances)"],
    common_misuse=[
        "Used when assumptions are severely violated",
        "Interpreting F-test without post-hoc pairwise comparisons",
    ],
    engine_function="one_way_anova",
    software="scipy.stats + statsmodels",
))

register_method(MethodEntry(
    method_id="linear_regression",
    name="OLS Linear regression",
    description="Model linear relationship between continuous outcome and predictors.",
    outcome_types=["continuous"],
    predictor_types=["continuous", "categorical", "binary", "mixed"],
    study_designs=["any"],
    min_observations=20,
    assumptions=[
        "Linear relationship between outcome and predictors",
        "Independence of residuals",
        "Homoscedasticity of residuals",
        "Normality of residuals",
        "No perfect multicollinearity",
    ],
    assumption_tests=["Residual plots", "Durbin-Watson", "Breusch-Pagan", "Shapiro-Wilk on residuals", "VIF"],
    effect_measure="R-squared, adjusted R-squared, coefficients",
    confidence_intervals=True,
    alternative_methods=["Robust regression", "Generalized linear model"],
    engine_function="linear_regression",
    software="statsmodels",
))

register_method(MethodEntry(
    method_id="logistic_regression",
    name="Logistic regression",
    description="Model binary outcome as function of predictors.",
    outcome_types=["binary"],
    predictor_types=["continuous", "categorical", "binary", "mixed"],
    study_designs=["any"],
    min_observations=50,
    assumptions=[
        "Binary outcome",
        "Independent observations",
        "Linearity of logit with continuous predictors",
        "No perfect multicollinearity",
        "Events per variable >= 10 (recommended)",
    ],
    assumption_tests=["Hosmer-Lemeshow", "VIF", "Events per variable check"],
    effect_measure="Odds ratio",
    confidence_intervals=True,
    alternative_methods=["Exact logistic regression", "Firth's penalized logistic regression"],
    common_misuse=[
        "Used when events per variable < 5",
        "Including too many predictors relative to events",
        "Ignoring separation",
    ],
    engine_function="logistic_regression",
    software="statsmodels",
))

register_method(MethodEntry(
    method_id="mann_whitney_u",
    name="Mann-Whitney U test",
    description="Nonparametric comparison of two independent groups.",
    outcome_types=["continuous", "ordinal"],
    predictor_types=["binary"],
    study_designs=["any"],
    min_groups=2,
    min_observations=6,
    assumptions=[
        "Independent observations",
        "Ordinal or continuous outcome",
        "Similar distribution shape (for location shift interpretation)",
    ],
    effect_measure="Rank-biserial r",
    confidence_intervals=True,
    alternative_methods=["Independent t-test", "Permutation test"],
    engine_function="mann_whitney_u",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="wilcoxon_signed_rank",
    name="Wilcoxon signed-rank test",
    description="Nonparametric comparison of two paired measurements.",
    outcome_types=["continuous", "ordinal"],
    predictor_types=["binary"],
    study_designs=["any"],
    min_observations=5,
    assumptions=[
        "Paired observations",
        "Ordinal or continuous outcome",
        "Symmetric distribution of differences",
    ],
    effect_measure="Effect size r",
    confidence_intervals=True,
    alternative_methods=["Paired t-test", "Sign test"],
    engine_function="wilcoxon_signed_rank",
    software="scipy.stats",
))

register_method(MethodEntry(
    method_id="descriptive",
    name="Descriptive statistics",
    description="Summarize data characteristics.",
    outcome_types=["any"],
    predictor_types=["any"],
    study_designs=["any"],
    min_observations=1,
    effect_measure="Various summary statistics",
    engine_function="descriptive_statistics",
    software="numpy/pandas/scipy",
))


# ---------------------------------------------------------------------------
# Survival Analysis Methods
# ---------------------------------------------------------------------------

register_method(MethodEntry(
    method_id="kaplan_meier",
    name="Kaplan-Meier estimator",
    description="Non-parametric survival function estimation.",
    outcome_types=["survival"],
    predictor_types=["categorical", "binary"],
    study_designs=["any"],
    min_groups=1,
    min_observations=5,
    assumptions=[
        "Censoring is non-informative (independent of outcome)",
        "Survival probabilities are the same for subjects recruited early and late",
        "Events are independent",
    ],
    effect_measure="Survival probability",
    confidence_intervals=True,
    alternative_methods=["Nelson-Aalen estimator"],
    common_misuse=[
        "Interpreting KM curves as causal without experimental design",
        "Extrapolating beyond observed data",
        "Ignoring informative censoring",
    ],
    limitations=[
        "Does not handle time-varying covariates",
        "Assumes non-informative censoring",
    ],
    engine_function="kaplan_meier",
    software="lifelines",
))

register_method(MethodEntry(
    method_id="log_rank",
    name="Log-rank test",
    description="Compare survival distributions between two groups.",
    outcome_types=["survival"],
    predictor_types=["binary"],
    study_designs=["any"],
    min_groups=2,
    min_observations=10,
    assumptions=[
        "Censoring is non-informative",
        "Survival distributions are proportional (hazard ratio is constant over time)",
        "Two independent groups",
    ],
    effect_measure="Test statistic (chi-square)",
    confidence_intervals=False,
    alternative_methods=["Wilcoxon (Breslow) test", "Tarone-Ware test"],
    common_misuse=[
        "Using when groups are not independent",
        "Interpreting as a test of median survival",
    ],
    engine_function="log_rank",
    software="lifelines",
))

register_method(MethodEntry(
    method_id="cox_proportional_hazards",
    name="Cox proportional hazards regression",
    description="Model hazard as function of covariates with proportional hazards assumption.",
    outcome_types=["survival"],
    predictor_types=["continuous", "categorical", "binary", "mixed"],
    study_designs=["any"],
    min_observations=20,
    assumptions=[
        "Proportional hazards: hazard ratio is constant over time",
        "Independent observations",
        "Log-linear relationship between continuous predictors and log-hazard",
        "No perfect multicollinearity",
    ],
    assumption_tests=["Schoenfeld residuals", "PH test"],
    effect_measure="Hazard ratio",
    confidence_intervals=True,
    alternative_methods=["Accelerated failure time model", "Stratified Cox model"],
    common_misuse=[
        "Violating proportional hazards without checking",
        "Interpreting hazard ratio as causal in observational studies",
        "Overfitting with too many covariates relative to events",
    ],
    limitations=[
        "Assumes proportional hazards",
        "Does not handle time-varying covariates in basic form",
    ],
    engine_function="cox_proportional_hazards",
    software="lifelines",
))
