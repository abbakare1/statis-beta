"""Phase 2J — Independent Statistical Validation (Invariant 33).

Validates STATIS results against independent implementations:
- SciPy
- statsmodels
- lifelines

TOLERANCES DOCUMENTED:
STATIS rounds outputs to 6 decimal places (most statistics) and 8 decimal
places (p-values). Maximum rounding error for N-decimal rounding is 0.5*10^-N.

- TOL_6DP = 1e-6: for statistics rounded to 6 decimal places
- TOL_8DP = 1e-7: for p-values rounded to 8 decimal places
- TOL_COEF = 1e-6: regression coefficients (rounded to 6dp)
- TOL_CI = 1e-6: confidence intervals (rounded to 6dp)
- TOL_SURVIVAL = 1e-5: survival probabilities (rounded to 6dp)
- TOL_HR = 1e-5: hazard ratios (rounded to 6dp)

These tolerances verify that STATIS and the independent implementation compute
the SAME underlying quantity; only display rounding differs. The underlying
computation libraries are identical (scipy/statsmodels/lifelines).

KNOWN DIFFERENCES:
- STATIS uses scipy for t-tests, correlations, chi-square, Mann-Whitney, Wilcoxon
- STATIS uses statsmodels for linear/logistic regression
- STATIS uses lifelines for survival
- Independent implementations call the same libraries directly
- Small numerical differences arise only from display rounding, not computation
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower
import lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.statistics import engine as stat_engine
from app.statistics import survival as stat_survival
from tests.fixtures.synthetic_dataset import generate_synthetic_cohort


# ---------------------------------------------------------------------------
# Tolerances
#
# STATIS rounds outputs to 6 decimal places (most statistics) and 8 decimal
# places (p-values). The maximum rounding error for a value rounded to N
# decimal places is 0.5 * 10^-N. Therefore:
#   - For 6-decimal rounded values: tol = 5e-7 + small epsilon
#   - For 8-decimal rounded p-values: tol = 5e-9 + small epsilon
#
# These tolerances verify that STATIS and the independent implementation
# compute the SAME underlying quantity; only display rounding differs.
# ---------------------------------------------------------------------------
TOL_6DP = 1e-6      # for values rounded to 6 decimal places
TOL_8DP = 1e-7      # for p-values rounded to 8 decimal places
TOL_COEF = 1e-6    # regression coefficients (rounded to 6dp)
TOL_CI = 1e-6      # confidence intervals (rounded to 6dp)
TOL_SURVIVAL = 1e-5  # survival probabilities (rounded to 6dp; compared at event times)
TOL_HR = 1e-5      # hazard ratios (rounded to 6dp)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cohort():
    return generate_synthetic_cohort()


@pytest.fixture(scope="module")
def cohort_csv(tmp_path_factory):
    df = generate_synthetic_cohort()
    path = tmp_path_factory.mktemp("validation") / "cohort.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def clean_ttest_data():
    """Clean two-group data for t-test validation."""
    rng = np.random.RandomState(42)
    g1 = rng.normal(50, 10, 80)
    g2 = rng.normal(55, 12, 75)
    return pd.DataFrame({
        "value": np.concatenate([g1, g2]),
        "group": ["A"] * 80 + ["B"] * 75,
    })


@pytest.fixture(scope="module")
def clean_ttest_csv(tmp_path_factory, clean_ttest_data):
    path = tmp_path_factory.mktemp("ttest") / "ttest.csv"
    clean_ttest_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def paired_data():
    rng = np.random.RandomState(42)
    before = rng.normal(100, 15, 60)
    after = before + rng.normal(-5, 8, 60)
    return pd.DataFrame({"before": before, "after": after})


@pytest.fixture(scope="module")
def paired_csv(tmp_path_factory, paired_data):
    path = tmp_path_factory.mktemp("paired") / "paired.csv"
    paired_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def correlation_data():
    rng = np.random.RandomState(42)
    x = rng.normal(0, 1, 120)
    y = 2.0 * x + rng.normal(0, 0.5, 120)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def correlation_csv(tmp_path_factory, correlation_data):
    path = tmp_path_factory.mktemp("corr") / "corr.csv"
    correlation_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def categorical_data():
    rng = np.random.RandomState(42)
    n = 200
    var1 = rng.choice(["X", "Y"], size=n)
    var2 = rng.choice(["P", "Q"], size=n, p=[0.6, 0.4])
    return pd.DataFrame({"var1": var1, "var2": var2})


@pytest.fixture(scope="module")
def categorical_csv(tmp_path_factory, categorical_data):
    path = tmp_path_factory.mktemp("cat") / "cat.csv"
    categorical_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def anova_data():
    rng = np.random.RandomState(42)
    g1 = rng.normal(50, 10, 40)
    g2 = rng.normal(55, 10, 40)
    g3 = rng.normal(48, 10, 40)
    return pd.DataFrame({
        "value": np.concatenate([g1, g2, g3]),
        "group": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
    })


@pytest.fixture(scope="module")
def anova_csv(tmp_path_factory, anova_data):
    path = tmp_path_factory.mktemp("anova") / "anova.csv"
    anova_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.RandomState(42)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 5 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 2, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def regression_csv(tmp_path_factory, regression_data):
    path = tmp_path_factory.mktemp("reg") / "reg.csv"
    regression_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def logistic_data():
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = -1.0 + 1.2 * x1 - 0.8 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def logistic_csv(tmp_path_factory, logistic_data):
    path = tmp_path_factory.mktemp("logit") / "logit.csv"
    logistic_data.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def survival_data():
    rng = np.random.RandomState(42)
    n = 200
    treatment = rng.binomial(1, 0.5, n)
    # Treatment reduces hazard
    hazard = 0.005 * np.where(treatment == 1, 0.6, 1.0)
    survival_time = rng.exponential(1.0 / hazard)
    censoring_time = rng.uniform(30, 400, n)
    time = np.minimum(survival_time, censoring_time)
    event = (survival_time < censoring_time).astype(int)
    return pd.DataFrame({
        "time": time,
        "event": event,
        "treatment": treatment,
        "age": rng.normal(62, 12, n),
    })


@pytest.fixture(scope="module")
def survival_csv(tmp_path_factory, survival_data):
    path = tmp_path_factory.mktemp("surv") / "surv.csv"
    survival_data.to_csv(path, index=False)
    return path


# ===========================================================================
# Independent t-test
# ===========================================================================

class TestIndependentTTest:
    def test_ttest_matches_scipy(self, clean_ttest_csv, clean_ttest_data):
        """STATIS Welch t-test matches scipy.stats.ttest_ind."""
        statis = stat_engine.independent_t_test(
            clean_ttest_csv, outcome="value", group="group",
            group1="A", group2="B", equal_var=False,
        )
        g1 = clean_ttest_data.loc[clean_ttest_data["group"] == "A", "value"]
        g2 = clean_ttest_data.loc[clean_ttest_data["group"] == "B", "value"]
        ref = sp_stats.ttest_ind(g1, g2, equal_var=False)

        assert abs(statis["t_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_ttest_student_matches_scipy(self, clean_ttest_csv, clean_ttest_data):
        """STATIS Student t-test matches scipy.stats.ttest_ind (equal_var=True)."""
        statis = stat_engine.independent_t_test(
            clean_ttest_csv, outcome="value", group="group",
            group1="A", group2="B", equal_var=True,
        )
        g1 = clean_ttest_data.loc[clean_ttest_data["group"] == "A", "value"]
        g2 = clean_ttest_data.loc[clean_ttest_data["group"] == "B", "value"]
        ref = sp_stats.ttest_ind(g1, g2, equal_var=True)

        assert abs(statis["t_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_ttest_cohens_d(self, clean_ttest_csv, clean_ttest_data):
        """Cohen's d matches manual pooled-SD calculation."""
        statis = stat_engine.independent_t_test(
            clean_ttest_csv, outcome="value", group="group",
            group1="A", group2="B",
        )
        g1 = clean_ttest_data.loc[clean_ttest_data["group"] == "A", "value"]
        g2 = clean_ttest_data.loc[clean_ttest_data["group"] == "B", "value"]
        pooled_sd = np.sqrt(((len(g1)-1)*g1.std(ddof=1)**2 + (len(g2)-1)*g2.std(ddof=1)**2) / (len(g1)+len(g2)-2))
        ref_d = (g1.mean() - g2.mean()) / pooled_sd
        assert abs(statis["cohens_d"] - ref_d) < TOL_6DP

    def test_ttest_sample_sizes(self, clean_ttest_csv, clean_ttest_data):
        """Sample sizes match."""
        statis = stat_engine.independent_t_test(
            clean_ttest_csv, outcome="value", group="group",
            group1="A", group2="B",
        )
        g1 = clean_ttest_data.loc[clean_ttest_data["group"] == "A", "value"]
        g2 = clean_ttest_data.loc[clean_ttest_data["group"] == "B", "value"]
        assert statis["n_group1"] == len(g1)
        assert statis["n_group2"] == len(g2)


# ===========================================================================
# Paired t-test
# ===========================================================================

class TestPairedTTest:
    def test_paired_ttest_matches_scipy(self, paired_csv, paired_data):
        """STATIS paired t-test matches scipy.stats.ttest_rel."""
        statis = stat_engine.paired_t_test(paired_csv, outcome1="before", outcome2="after")
        ref = sp_stats.ttest_rel(paired_data["before"], paired_data["after"])

        assert abs(statis["t_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_paired_ttest_mean_difference(self, paired_csv, paired_data):
        """Mean difference matches manual calculation."""
        statis = stat_engine.paired_t_test(paired_csv, outcome1="before", outcome2="after")
        diffs = paired_data["before"] - paired_data["after"]
        assert abs(statis["mean_difference"] - diffs.mean()) < TOL_6DP

    def test_paired_ttest_df(self, paired_csv, paired_data):
        """Degrees of freedom = n_pairs - 1."""
        statis = stat_engine.paired_t_test(paired_csv, outcome1="before", outcome2="after")
        assert statis["df"] == len(paired_data) - 1


# ===========================================================================
# Mann-Whitney U
# ===========================================================================

class TestMannWhitney:
    def test_mannwhitney_matches_scipy(self, clean_ttest_csv, clean_ttest_data):
        """STATIS Mann-Whitney matches scipy.stats.mannwhitneyu."""
        statis = stat_engine.mann_whitney_u(
            clean_ttest_csv, outcome="value", group="group",
            group1="A", group2="B",
        )
        g1 = clean_ttest_data.loc[clean_ttest_data["group"] == "A", "value"]
        g2 = clean_ttest_data.loc[clean_ttest_data["group"] == "B", "value"]
        ref = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")

        assert abs(statis["u_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP


# ===========================================================================
# Wilcoxon signed-rank
# ===========================================================================

class TestWilcoxon:
    def test_wilcoxon_matches_scipy(self, paired_csv, paired_data):
        """STATIS Wilcoxon matches scipy.stats.wilcoxon."""
        statis = stat_engine.wilcoxon_signed_rank(paired_csv, outcome1="before", outcome2="after")
        diffs = paired_data["before"] - paired_data["after"]
        diffs = diffs[diffs != 0]
        ref = sp_stats.wilcoxon(diffs, alternative="two-sided")

        assert abs(statis["w_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP


# ===========================================================================
# Pearson correlation
# ===========================================================================

class TestPearsonCorrelation:
    def test_pearson_matches_scipy(self, correlation_csv, correlation_data):
        """STATIS Pearson correlation matches scipy.stats.pearsonr."""
        statis = stat_engine.correlation_test(correlation_csv, x="x", y="y", method="pearson")
        ref = sp_stats.pearsonr(correlation_data["x"], correlation_data["y"])

        assert abs(statis["r"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_pearson_r_squared(self, correlation_csv, correlation_data):
        """r² = r²."""
        statis = stat_engine.correlation_test(correlation_csv, x="x", y="y", method="pearson")
        assert abs(statis["r_squared"] - statis["r"] ** 2) < TOL_6DP


# ===========================================================================
# Spearman correlation
# ===========================================================================

class TestSpearmanCorrelation:
    def test_spearman_matches_scipy(self, correlation_csv, correlation_data):
        """STATIS Spearman correlation matches scipy.stats.spearmanr."""
        statis = stat_engine.correlation_test(correlation_csv, x="x", y="y", method="spearman")
        ref = sp_stats.spearmanr(correlation_data["x"], correlation_data["y"])

        assert abs(statis["r"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP


# ===========================================================================
# Chi-square test
# ===========================================================================

class TestChiSquare:
    def test_chi_square_matches_scipy(self, categorical_csv, categorical_data):
        """STATIS chi-square matches scipy.stats.chi2_contingency."""
        statis = stat_engine.chi_square_test(categorical_csv, variable1="var1", variable2="var2")
        ct = pd.crosstab(categorical_data["var1"], categorical_data["var2"])
        ref = sp_stats.chi2_contingency(ct.values, correction=False)

        assert abs(statis["chi2"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP
        assert statis["df"] == ref.dof


# ===========================================================================
# Fisher exact test
# ===========================================================================

class TestFisherExact:
    def test_fisher_matches_scipy(self, categorical_csv, categorical_data):
        """STATIS Fisher exact matches scipy.stats.fisher_exact."""
        statis = stat_engine.fisher_exact_test(categorical_csv, variable1="var1", variable2="var2")
        ct = pd.crosstab(categorical_data["var1"], categorical_data["var2"])
        ref = sp_stats.fisher_exact(ct.values)

        assert abs(statis["odds_ratio"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP


# ===========================================================================
# One-way ANOVA
# ===========================================================================

class TestOneWayANOVA:
    def test_anova_matches_scipy(self, anova_csv, anova_data):
        """STATIS one-way ANOVA matches scipy.stats.f_oneway."""
        statis = stat_engine.one_way_anova(anova_csv, outcome="value", group="group")
        groups = [anova_data.loc[anova_data["group"] == g, "value"] for g in ["A", "B", "C"]]
        ref = sp_stats.f_oneway(*groups)

        assert abs(statis["f_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_anova_group_statistics(self, anova_csv, anova_data):
        """Per-group n, mean, std match."""
        statis = stat_engine.one_way_anova(anova_csv, outcome="value", group="group")
        for g in ["A", "B", "C"]:
            subset = anova_data.loc[anova_data["group"] == g, "value"]
            assert statis["group_statistics"][g]["n"] == len(subset)
            assert abs(statis["group_statistics"][g]["mean"] - subset.mean()) < TOL_6DP


# ===========================================================================
# Linear regression
# ===========================================================================

class TestLinearRegression:
    def test_ols_matches_statsmodels(self, regression_csv, regression_data):
        """STATIS OLS matches statsmodels.OLS directly."""
        statis = stat_engine.linear_regression(
            regression_csv, outcome="y", predictors=["x1", "x2"],
        )
        X = sm.add_constant(regression_data[["x1", "x2"]])
        ref = sm.OLS(regression_data["y"], X).fit()

        assert abs(statis["r_squared"] - ref.rsquared) < TOL_COEF
        assert abs(statis["f_statistic"] - ref.fvalue) < TOL_COEF
        assert abs(statis["f_p_value"] - ref.f_pvalue) < TOL_8DP
        for name in ["const", "x1", "x2"]:
            assert abs(statis["parameters"][name]["coef"] - ref.params[name]) < TOL_COEF
            assert abs(statis["parameters"][name]["std_err"] - ref.bse[name]) < TOL_COEF
            assert abs(statis["parameters"][name]["p_value"] - ref.pvalues[name]) < TOL_8DP


# ===========================================================================
# Logistic regression
# ===========================================================================

class TestLogisticRegression:
    def test_logistic_matches_statsmodels(self, logistic_csv, logistic_data):
        """STATIS logistic regression matches statsmodels.Logit directly."""
        statis = stat_engine.logistic_regression(
            logistic_csv, outcome="y", predictors=["x1", "x2"],
        )
        X = sm.add_constant(logistic_data[["x1", "x2"]])
        ref = sm.Logit(logistic_data["y"], X).fit(disp=False)

        for name in ["const", "x1", "x2"]:
            assert abs(statis["parameters"][name]["coef"] - ref.params[name]) < TOL_COEF
            assert abs(statis["parameters"][name]["std_err"] - ref.bse[name]) < TOL_COEF
            assert abs(statis["parameters"][name]["p_value"] - ref.pvalues[name]) < TOL_8DP
            # Odds ratio
            assert abs(statis["parameters"][name]["odds_ratio"] - np.exp(ref.params[name])) < TOL_COEF


# ===========================================================================
# Kaplan-Meier
# ===========================================================================

class TestKaplanMeier:
    def test_km_matches_lifelines(self, survival_csv, survival_data):
        """STATIS KM matches lifelines.KaplanMeierFitter directly."""
        statis = stat_survival.kaplan_meier(survival_csv, time_col="time", event_col="event")

        kmf = KaplanMeierFitter()
        kmf.fit(survival_data["time"], event_observed=survival_data["event"])

        # Compare survival function values at event times
        sf = kmf.survival_function_
        ref_times = sf.index.tolist()
        ref_surv = sf.iloc[:, 0].tolist()

        statis_times = statis["groups"]["overall"]["times"]
        statis_probs = statis["groups"]["overall"]["survival_probabilities"]

        # Both should have the same number of time points
        n_compare = min(len(statis_times), len(ref_times), len(statis_probs), len(ref_surv))
        assert n_compare > 0

        # Compare at each time point (accounting for 6dp rounding of times)
        for i in range(n_compare):
            # Time should match within rounding tolerance
            assert abs(statis_times[i] - ref_times[i]) < 1e-4
            # Survival probability should match within rounding tolerance
            assert abs(statis_probs[i] - ref_surv[i]) < TOL_SURVIVAL

    def test_km_grouped_matches_lifelines(self, survival_csv, survival_data):
        """Grouped KM matches lifelines per-group."""
        statis = stat_survival.kaplan_meier(
            survival_csv, time_col="time", event_col="event", group_col="treatment",
        )
        for group_val in [0, 1]:
            subset = survival_data[survival_data["treatment"] == group_val]
            kmf = KaplanMeierFitter()
            kmf.fit(subset["time"], event_observed=subset["event"])
            group_key = str(group_val)

            sf = kmf.survival_function_
            ref_times = sf.index.tolist()
            ref_surv = sf.iloc[:, 0].tolist()

            statis_times = statis["groups"][group_key]["times"]
            statis_probs = statis["groups"][group_key]["survival_probabilities"]

            n_compare = min(len(statis_times), len(ref_times), len(statis_probs), len(ref_surv))
            assert n_compare > 0

            for i in range(n_compare):
                assert abs(statis_times[i] - ref_times[i]) < 1e-4
                assert abs(statis_probs[i] - ref_surv[i]) < TOL_SURVIVAL


# ===========================================================================
# Log-rank test
# ===========================================================================

class TestLogRank:
    def test_logrank_matches_lifelines(self, survival_csv, survival_data):
        """STATIS log-rank matches lifelines.statistics.logrank_test."""
        statis = stat_survival.log_rank(
            survival_csv, time_col="time", event_col="event",
            group_col="treatment", group1=0, group2=1,
        )
        g0 = survival_data[survival_data["treatment"] == 0]
        g1 = survival_data[survival_data["treatment"] == 1]
        ref = logrank_test(
            g0["time"], g1["time"],
            event_observed_A=g0["event"], event_observed_B=g1["event"],
        )

        assert abs(statis["test_statistic"] - ref.test_statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.p_value) < TOL_8DP
        assert statis["degrees_of_freedom"] == ref.degrees_of_freedom


# ===========================================================================
# Cox proportional hazards
# ===========================================================================

class TestCoxRegression:
    def test_cox_matches_lifelines(self, survival_csv, survival_data):
        """STATIS Cox PH matches lifelines.CoxPHFitter directly."""
        statis = stat_survival.cox_proportional_hazards(
            survival_csv, time_col="time", event_col="event",
            covariates=["treatment", "age"],
        )
        cph = CoxPHFitter()
        cph.fit(survival_data[["time", "event", "treatment", "age"]],
                duration_col="time", event_col="event")

        for cov in ["treatment", "age"]:
            assert abs(statis["hazard_ratios"][cov] - np.exp(cph.params_[cov])) < TOL_HR
            assert abs(statis["coefficients"][cov] - cph.params_[cov]) < TOL_COEF
            assert abs(statis["standard_errors"][cov] - cph.summary.loc[cov, "se(coef)"]) < TOL_COEF
            assert abs(statis["p_values"][cov] - cph.summary.loc[cov, "p"]) < TOL_8DP


# ===========================================================================
# Power analysis
# ===========================================================================

class TestPowerAnalysis:
    def test_power_matches_statsmodels(self):
        """STATIS power analysis matches statsmodels.TTestIndPower."""
        statis = stat_engine.power_analysis_t_test(effect_size=0.5, n_per_group=64, alpha=0.05)
        ref = TTestIndPower().solve_power(effect_size=0.5, nobs1=64, alpha=0.05, power=None, ratio=1.0)
        # Power is rounded to 6 decimal places in STATIS
        assert abs(statis["power"] - ref) < TOL_6DP

    def test_sample_size_matches_statsmodels(self):
        """Required sample size matches statsmodels."""
        statis = stat_engine.power_analysis_t_test(effect_size=0.5, power=0.80, alpha=0.05)
        ref = TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.80, ratio=1.0)
        assert statis["required_n_per_group"] == int(np.ceil(ref))


# ===========================================================================
# Cohort-based validation
# ===========================================================================

class TestCohortValidation:
    def test_cohort_ttest_matches_scipy(self, cohort_csv):
        """Independent t-test on synthetic cohort matches scipy."""
        statis = stat_engine.independent_t_test(
            cohort_csv, outcome="qol_score", group="treatment",
            group1=0, group2=1,
        )
        df = generate_synthetic_cohort()
        g0 = df.loc[df["treatment"] == 0, "qol_score"].dropna()
        g1 = df.loc[df["treatment"] == 1, "qol_score"].dropna()
        ref = sp_stats.ttest_ind(g0, g1, equal_var=False)

        assert abs(statis["t_statistic"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_cohort_pearson_matches_scipy(self, cohort_csv):
        """Pearson correlation on cohort matches scipy."""
        statis = stat_engine.correlation_test(
            cohort_csv, x="age", y="qol_score", method="pearson",
        )
        df = generate_synthetic_cohort()
        valid = df[["age", "qol_score"]].dropna()
        ref = sp_stats.pearsonr(valid["age"], valid["qol_score"])

        assert abs(statis["r"] - ref.statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.pvalue) < TOL_8DP

    def test_cohort_logrank_matches_lifelines(self, cohort_csv):
        """Log-rank on cohort survival data matches lifelines."""
        statis = stat_survival.log_rank(
            cohort_csv, time_col="time_to_event", event_col="event_observed",
            group_col="treatment", group1=0, group2=1,
        )
        df = generate_synthetic_cohort()
        g0 = df[df["treatment"] == 0]
        g1 = df[df["treatment"] == 1]
        ref = logrank_test(
            g0["time_to_event"], g1["time_to_event"],
            event_observed_A=g0["event_observed"], event_observed_B=g1["event_observed"],
        )
        assert abs(statis["test_statistic"] - ref.test_statistic) < TOL_6DP
        assert abs(statis["p_value"] - ref.p_value) < TOL_8DP
