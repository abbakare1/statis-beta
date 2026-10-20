"""Output Specifications — strongly typed specifications for publication outputs.

Phase 2G: Defines the contract between validated statistical results and
deterministic output rendering. Every figure, table, and report must have
a validated specification before rendering.

These specifications NEVER recompute statistics. They consume persisted
StatisticalResult objects and associated metadata.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputLifecycle(str, Enum):
    """Lifecycle states for publication outputs."""
    DRAFT = "draft"
    VALIDATING = "validating"
    VALIDATED = "validated"
    RENDERING = "rendering"
    GENERATED = "generated"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    ARCHIVED = "archived"
    VALIDATION_FAILED = "validation_failed"
    RENDERING_FAILED = "rendering_failed"


class FigureType(str, Enum):
    """Supported figure types."""
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    SCATTER_PLOT = "scatter_plot"
    BAR_CHART = "bar_chart"
    GROUPED_BOX_PLOT = "grouped_box_plot"
    MEAN_CI_PLOT = "mean_ci_plot"
    COEFFICIENT_PLOT = "coefficient_plot"
    KAPLAN_MEIER = "kaplan_meier"
    GROUPED_KAPLAN_MEIER = "grouped_kaplan_meier"
    RESIDUAL_VS_FITTED = "residual_vs_fitted"
    QQ_PLOT = "qq_plot"


class TableType(str, Enum):
    """Supported table types."""
    DESCRIPTIVE_BASELINE = "descriptive_baseline"
    INFERENTIAL_RESULTS = "inferential_results"
    REGRESSION_TABLE = "regression_table"
    SURVIVAL_TABLE = "survival_table"
    CUSTOM = "custom"


class ReportSection(str, Enum):
    """Report sections."""
    TITLE = "title"
    ABSTRACT = "abstract"
    RESEARCH_QUESTION = "research_question"
    STUDY_DESIGN = "study_design"
    DATA_SOURCE = "data_source"
    COHORT_DESCRIPTION = "cohort_description"
    DATA_QUALITY = "data_quality"
    MISSINGNESS = "missingness"
    OUTLIER_ASSESSMENT = "outlier_assessment"
    STATISTICAL_METHODS = "statistical_methods"
    PRIMARY_ANALYSIS = "primary_analysis"
    SECONDARY_ANALALYSES = "secondary_analyses"
    SENSITIVITY_ANALYSES = "sensitivity_analyses"
    EXPLORATORY_ANALYSES = "exploratory_analyses"
    RESULTS = "results"
    FIGURES = "figures"
    TABLES = "tables"
    LIMITATIONS = "limitations"
    REPRODUCIBILITY = "reproducibility"


class AnalyticalClassification(str, Enum):
    """Analytical status classification."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SENSITIVITY = "sensitivity"
    EXPLORATORY = "exploratory"


class OutputFormat(str, Enum):
    """Output file formats."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


# ---------------------------------------------------------------------------
# Base Specification
# ---------------------------------------------------------------------------

class OutputSpecification(BaseModel):
    """Base class for all output specifications."""
    output_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str | None = None
    title: str = ""
    caption: str = ""
    created_by: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_spec_hash(self) -> str:
        """Compute deterministic hash of the specification."""
        canonical = self.model_dump(mode="json")
        canonical.pop("output_id", None)
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Figure Specification
# ---------------------------------------------------------------------------

class FigureSpecification(OutputSpecification):
    """Specification for a publication figure."""
    figure_type: FigureType
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_id: str | None = None
    analysis_run_id: str | None = None
    result_fingerprint: str | None = None

    # Variables
    variable: str | None = None
    grouping_variable: str | None = None
    x_variable: str | None = None
    y_variable: str | None = None
    covariates: list[str] = Field(default_factory=list)

    # Axis labels
    x_label: str = ""
    y_label: str = ""
    legend_label: str = ""

    # Display
    units: str = ""
    decimal_places: int = 3
    category_order: list[str] = Field(default_factory=list)
    show_legend: bool = True
    show_grid: bool = False

    # Format
    output_format: OutputFormat = OutputFormat.PNG
    width_inches: float = 6.0
    height_inches: float = 4.0
    dpi: int = 300
    seed: int | None = None

    # Classification
    analytical_classification: AnalyticalClassification = AnalyticalClassification.EXPLORATORY
    prespecified_status: str = "unclear"


# ---------------------------------------------------------------------------
# Table Specification
# ---------------------------------------------------------------------------

class TableColumn(BaseModel):
    """Definition of a table column."""
    key: str
    label: str
    width: int | None = None
    align: str = "left"


class TableRow(BaseModel):
    """Definition of a table row."""
    key: str
    label: str
    level: int = 0
    bold: bool = False


class TableSpecification(OutputSpecification):
    """Specification for a publication table."""
    table_type: TableType
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_ids: list[str] = Field(default_factory=list)
    analysis_run_ids: list[str] = Field(default_factory=list)

    # Structure
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)

    # Display
    decimal_places: int = 3
    statistical_notation: str = "standard"
    footnotes: list[str] = Field(default_factory=list)
    include_p_values: bool = True
    include_ci: bool = True
    include_n: bool = True
    include_intercept: bool = False

    # Format
    output_format: OutputFormat = OutputFormat.CSV
    include_significance_marks: bool = False

    # Classification
    analytical_classification: AnalyticalClassification = AnalyticalClassification.EXPLORATORY
    prespecified_status: str = "unclear"


# ---------------------------------------------------------------------------
# Report Specification
# ---------------------------------------------------------------------------

class ReportSectionSpec(BaseModel):
    """Specification for a report section."""
    section_type: ReportSection
    title: str = ""
    content: str = ""
    include: bool = True
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class AuthorMetadata(BaseModel):
    """Author metadata for reports."""
    name: str = ""
    affiliation: str = ""
    email: str = ""


class ReportSpecification(OutputSpecification):
    """Specification for a structured report."""
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_ids: list[str] = Field(default_factory=list)
    analysis_run_ids: list[str] = Field(default_factory=list)

    # Content
    sections: list[ReportSectionSpec] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)

    # Author
    authors: list[AuthorMetadata] = Field(default_factory=list)

    # Format
    output_format: OutputFormat = OutputFormat.MARKDOWN

    # Classification
    analytical_classification: AnalyticalClassification = AnalyticalClassification.EXPLORATORY
    prespecified_status: str = "unclear"


# ---------------------------------------------------------------------------
# Validation Results
# ---------------------------------------------------------------------------

class OutputValidationResult(BaseModel):
    """Result of output specification validation."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_errors: list[str] = Field(default_factory=list)


def validate_figure_specification(
    spec: FigureSpecification,
    *,
    available_columns: list[str] | None = None,
    has_result: bool = False,
    result_method_id: str | None = None,
    has_survival_metadata: bool = False,
) -> OutputValidationResult:
    """Validate a figure specification before rendering."""
    errors: list[str] = []
    warnings: list[str] = []
    blocking: list[str] = []

    # Check required fields
    if not spec.figure_type:
        errors.append("figure_type is required")

    # Check dataset binding
    if not spec.dataset_id and not spec.analysis_run_id:
        errors.append("Either dataset_id or analysis_run_id must be specified")
        blocking.append("Missing dataset/analysis binding")

    # Check variable exists
    if spec.variable and available_columns is not None:
        if spec.variable not in available_columns:
            errors.append(f"Variable '{spec.variable}' not found in dataset")
            blocking.append(f"Variable '{spec.variable}' not found")

    if spec.grouping_variable and available_columns is not None:
        if spec.grouping_variable not in available_columns:
            errors.append(f"Grouping variable '{spec.grouping_variable}' not found")
            blocking.append(f"Grouping variable '{spec.grouping_variable}' not found")

    # Check figure-specific requirements
    if spec.figure_type in (FigureType.KAPLAN_MEIER, FigureType.GROUPED_KAPLAN_MEIER):
        if not has_survival_metadata and not has_result:
            errors.append("Kaplan-Meier figure requires survival metadata")
            blocking.append("Missing survival metadata for KM figure")

    if spec.figure_type == FigureType.GROUPED_KAPLAN_MEIER:
        if not spec.grouping_variable:
            errors.append("Grouped Kaplan-Meier requires a grouping variable")
            blocking.append("Missing grouping variable for grouped KM")

    if spec.figure_type == FigureType.COEFFICIENT_PLOT:
        if not has_result:
            errors.append("Coefficient plot requires a statistical result")
            blocking.append("Missing statistical result for coefficient plot")
        if result_method_id and result_method_id not in (
            "linear_regression", "logistic_regression", "cox_proportional_hazards"
        ):
            errors.append(
                f"Coefficient plot not supported for method '{result_method_id}'"
            )
            blocking.append(f"Unsupported method for coefficient plot: {result_method_id}")

    if spec.figure_type in (FigureType.SCATTER_PLOT, FigureType.MEAN_CI_PLOT):
        if not spec.x_variable and not spec.variable:
            errors.append(f"{spec.figure_type.value} requires x_variable or variable")
            blocking.append(f"Missing variable specification for {spec.figure_type.value}")

    # Validate dimensions
    if spec.width_inches <= 0 or spec.height_inches <= 0:
        errors.append("Figure dimensions must be positive")
        blocking.append("Invalid figure dimensions")

    if spec.dpi < 72:
        warnings.append(f"Low DPI ({spec.dpi}); consider >= 300 for publication")

    valid = len(blocking) == 0 and len(errors) == 0
    return OutputValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        blocking_errors=blocking,
    )


def validate_table_specification(
    spec: TableSpecification,
    *,
    available_columns: list[str] | None = None,
    has_result: bool = False,
    result_method_id: str | None = None,
) -> OutputValidationResult:
    """Validate a table specification before rendering."""
    errors: list[str] = []
    warnings: list[str] = []
    blocking: list[str] = []

    # Check required fields
    if not spec.table_type:
        errors.append("table_type is required")

    # Check dataset binding
    if not spec.dataset_id and not spec.analysis_run_ids:
        errors.append("Either dataset_id or analysis_run_ids must be specified")
        blocking.append("Missing dataset/analysis binding")

    # Check variables exist
    if available_columns is not None:
        for var in spec.variables:
            if var not in available_columns:
                errors.append(f"Variable '{var}' not found in dataset")
                blocking.append(f"Variable '{var}' not found")

    # Check table-specific requirements
    if spec.table_type == TableType.REGRESSION_TABLE:
        if not has_result:
            errors.append("Regression table requires a statistical result")
            blocking.append("Missing statistical result for regression table")

    if spec.table_type == TableType.SURVIVAL_TABLE:
        if not has_result:
            errors.append("Survival table requires a statistical result")
            blocking.append("Missing statistical result for survival table")

    # Validate columns
    if not spec.columns:
        warnings.append("No columns defined; renderer will use defaults")

    valid = len(blocking) == 0 and len(errors) == 0
    return OutputValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        blocking_errors=blocking,
    )


def validate_report_specification(
    spec: ReportSpecification,
    *,
    has_analyses: bool = False,
) -> OutputValidationResult:
    """Validate a report specification before rendering."""
    errors: list[str] = []
    warnings: list[str] = []
    blocking: list[str] = []

    # Check required fields
    if not spec.title:
        errors.append("Report title is required")
        blocking.append("Missing report title")

    if not spec.dataset_id and not spec.analysis_run_ids:
        errors.append("Either dataset_id or analysis_run_ids must be specified")
        blocking.append("Missing dataset/analysis binding")

    if not has_analyses and not spec.analysis_run_ids:
        errors.append("Report requires at least one analysis")
        blocking.append("No analyses specified for report")

    # Check sections
    if not spec.sections:
        warnings.append("No sections defined; renderer will use default structure")

    # Check for narrative safety
    for section in spec.sections:
        if section.content:
            # Check for causal language that might be inappropriate
            causal_terms = ["caused", "led to", "resulted in", "prevented", "improved"]
            for term in causal_terms:
                if term.lower() in section.content.lower():
                    warnings.append(
                        f"Section '{section.section_type.value}' contains potentially "
                        f"causal language: '{term}'"
                    )

    valid = len(blocking) == 0 and len(errors) == 0
    return OutputValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        blocking_errors=blocking,
    )
