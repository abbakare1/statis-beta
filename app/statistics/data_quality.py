"""Data Quality Profile — structured representation of dataset quality assessment.

Deterministic. Never sends raw data to LLM. Produces a structured profile
that captures missingness, outliers, duplicates, invalid values, and summary statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

class MissingnessStatus(str, Enum):
    NONE = "none"                       # 0% missing
    LOW = "low"                         # 0 < x <= 5%
    MODERATE = "moderate"               # 5 < x <= 20%
    HIGH = "high"                       # 20 < x <= 50%
    EXTREME = "extreme"                 # > 50% missing
    ALL_MISSING = "all_missing"         # 100% missing


class MissingnessPattern(str, Enum):
    """Classification of observed missingness — NOT mechanism (MCAR/MAR/MNAR)."""
    OBSERVED_PATTERN = "observed_pattern"
    POTENTIAL_CONCERN = "potential_concern"
    REQUIRES_RESEARCHER_ASSESSMENT = "requires_researcher_assessment"


class MissingDataStrategy(str, Enum):
    NO_ACTION = "no_action"
    COMPLETE_CASE_ANALYSIS = "complete_case_analysis"
    MISSING_INDICATOR = "missing_indicator"
    SIMPLE_IMPUTATION = "simple_imputation"
    KNN_IMPUTATION = "knn_imputation"
    MULTIPLE_IMPUTATION = "multiple_imputation"


class VariableMissingness(BaseModel):
    """Missingness summary for a single variable."""
    variable: str
    variable_type: str
    n_missing: int
    n_observed: int
    total: int
    pct_missing: float
    status: MissingnessStatus
    is_outcome: bool = False
    is_predictor: bool = False
    recommendation: str | None = None


class DatasetMissingness(BaseModel):
    """Dataset-level missingness summary."""
    total_rows: int
    complete_cases: int
    incomplete_cases: int
    total_missing_cells: int
    total_cells: int
    pct_missing_overall: float
    variables: list[VariableMissingness] = []
    pattern_classification: MissingnessPattern = MissingnessPattern.OBSERVED_PATTERN
    pattern_details: str | None = None
    missingness_patterns: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

class OutlierMethod(str, Enum):
    IQR = "iqr"
    Z_SCORE = "z_score"
    MAD = "mad"


class OutlierInterpretation(str, Enum):
    POSSIBLE_DATA_ERROR = "possible_data_error"
    POSSIBLE_TRUE_EXTREME = "possible_true_extreme"
    POSSIBLE_INFLUENTIAL_OBSERVATION = "possible_influential_observation"
    UNKNOWN = "unknown"


class OutlierHandlingStrategy(str, Enum):
    RETAIN = "retain"
    EXCLUDE = "exclude"
    WINSORIZE = "winsorize"
    TRANSFORM = "transform"
    ROBUST_METHOD = "robust_method"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"


class OutlierFlag(BaseModel):
    """A single flagged outlier observation."""
    observation_index: int
    variable: str
    value: float
    method: str
    threshold: float
    interpretation: OutlierInterpretation = OutlierInterpretation.UNKNOWN
    handling_strategies: list[OutlierHandlingStrategy] = [OutlierHandlingStrategy.RETAIN]


class VariableOutliers(BaseModel):
    """Outlier summary for a single variable."""
    variable: str
    method: str
    threshold: float
    n_flagged: int
    pct_flagged: float
    min_value: float | None = None
    max_value: float | None = None
    fence_lower: float | None = None
    fence_upper: float | None = None
    flags: list[OutlierFlag] = []
    interpretation: OutlierInterpretation = OutlierInterpretation.UNKNOWN
    handling_strategies: list[OutlierHandlingStrategy] = [OutlierHandlingStrategy.RETAIN]


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

class TransformationType(str, Enum):
    LOG = "log"
    LOG1P = "log1p"
    SQRT = "sqrt"
    STANDARDIZE = "standardize"
    MIN_MAX = "min_max"
    YEO_JOHNSON = "yeo_johnson"
    BOX_COX = "box_cox"
    WINSORIZE = "winsorize"


class TransformationSafety(str, Enum):
    SAFE = "safe"
    REQUIRES_SHIFT = "requires_shift"
    BLOCKED = "blocked"
    REQUIRES_REVIEW = "requires_review"


class TransformationCandidate(BaseModel):
    """A proposed transformation with safety and domain validation."""
    transformation_type: TransformationType
    variable: str
    reason: str | None = None
    safety: TransformationSafety = TransformationSafety.SAFE
    safety_details: str | None = None
    domain_violations: list[str] = []
    parameters: dict[str, Any] = {}
    fitted_parameters: dict[str, Any] = {}
    interpretational_consequence: str | None = None


# ---------------------------------------------------------------------------
# Handling Strategy
# ---------------------------------------------------------------------------

class HandlingAction(str, Enum):
    RETAIN = "retain"
    EXCLUDE_OBSERVATIONS = "exclude_observations"
    EXCLUDE_VARIABLE = "exclude_variable"
    IMPUTE = "impute"
    TRANSFORM = "transform"
    WINSORIZE = "winsorize"
    NO_ACTION = "no_action"


class HandlingStrategy(BaseModel):
    """A proposed data handling strategy for a specific issue."""
    strategy_id: str
    action: HandlingAction
    target_variable: str | None = None
    target_observations: list[int] = []
    details: str
    impact_summary: dict[str, Any] = {}
    requires_approval: bool = True


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

class SensitivityVariant(BaseModel):
    """A single variant for sensitivity analysis."""
    variant_label: str
    description: str
    handling_changes: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Data Quality Profile (top-level)
# ---------------------------------------------------------------------------

class DataQualityProfile(BaseModel):
    """Complete data quality profile for a dataset.

    Deterministic. Generated from data alone. No LLM involvement.
    """
    dataset_id: str | None = None
    dataset_version: int | None = None
    file_path: str | None = None
    row_count: int = 0
    column_count: int = 0
    duplicate_rows: int = 0
    duplicate_subject_ids: int = 0
    constant_variables: list[str] = []
    near_constant_variables: list[str] = []

    # Missingness
    missingness: DatasetMissingness | None = None

    # Numeric summaries
    numeric_summary: dict[str, dict[str, Any]] = {}

    # Categorical summaries
    categorical_summary: dict[str, dict[str, Any]] = {}

    # Type mismatches
    potential_type_mismatches: list[dict[str, Any]] = []

    # Invalid values
    invalid_values: list[dict[str, Any]] = []

    # Range violations
    range_violations: list[dict[str, Any]] = []

    # Outliers
    outlier_summary: list[VariableOutliers] = []

    # Missing data strategies
    missing_data_strategies: list[HandlingStrategy] = []

    # Transformation candidates
    transformation_candidates: list[TransformationCandidate] = []

    # Sensitivity variants
    sensitivity_variants: list[SensitivityVariant] = []

    # Metadata
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def has_critical_issues(self) -> bool:
        """Check if there are critical data quality issues."""
        if self.missingness:
            for v in self.missingness.variables:
                if v.status in (MissingnessStatus.EXTREME, MissingnessStatus.ALL_MISSING):
                    return True
        if len(self.constant_variables) > 0:
            return True
        return False

    def get_high_impact_variables(self) -> list[str]:
        """Return variables with high/extreme missingness or many outliers."""
        high_impact = []
        if self.missingness:
            for v in self.missingness.variables:
                if v.status in (MissingnessStatus.HIGH, MissingnessStatus.EXTREME):
                    high_impact.append(v.variable)
        for o in self.outlier_summary:
            if o.pct_flagged > 10:
                if o.variable not in high_impact:
                    high_impact.append(o.variable)
        return high_impact
