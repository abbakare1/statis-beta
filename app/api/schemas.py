"""API Schemas — Pydantic request/response contracts for output endpoints.

Phase 2H: Explicit schemas for all output API interactions. SQLAlchemy
models are never exposed directly as API contracts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared response pieces
# ---------------------------------------------------------------------------

class ProvenanceInfo(BaseModel):
    """Provenance metadata included in output responses."""
    output_id: str | None = None
    output_type: str | None = None
    output_spec_hash: str | None = None
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_id: str | None = None
    analysis_run_id: str | None = None
    result_fingerprint: str | None = None
    artifact_hash: str | None = None
    artifact_format: str | None = None
    artifact_size_bytes: int | None = None
    output_fingerprint: str | None = None
    statis_version: str | None = None
    python_version: str | None = None
    platform: str | None = None
    generated_at: str | None = None
    analytical_classification: str | None = None
    prespecified_status: str | None = None
    lifecycle_status: str | None = None


class OutputResponse(BaseModel):
    """Generic output response payload."""
    id: str
    output_type: str
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    specification_hash: str | None = None
    artifact_path: str | None = None
    artifact_hash: str | None = None
    artifact_format: str | None = None
    lifecycle: str
    allowed_transitions: list[str] = Field(default_factory=list)
    is_immutable: bool = False
    validation_errors: list[str] | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    analytical_classification: str = "exploratory"
    prespecified_status: str = "unclear"
    provenance: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    # Type-specific
    title: str | None = None
    output_format: str | None = None
    # Figure-specific
    figure_type: str | None = None
    analysis_version_id: str | None = None
    analysis_run_id: str | None = None
    result_fingerprint: str | None = None
    # Table/Report-specific
    table_type: str | None = None
    analysis_version_ids: list[str] | None = None
    analysis_run_ids: list[str] | None = None


class ServiceEnvelope(BaseModel):
    """Envelope wrapping service results in API responses."""
    success: bool
    output: OutputResponse | None = None
    outputs: list[OutputResponse] | None = None
    count: int | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    error_code: str | None = None
    # Validation-specific
    output_id: str | None = None
    valid: bool | None = None
    lifecycle: str | None = None
    # Render-specific
    artifact_hash: str | None = None
    artifact_format: str | None = None
    artifact_size_bytes: int | None = None
    idempotent: bool | None = None
    equivalent_output_id: str | None = None
    message: str | None = None
    # Artifact-specific
    artifact_path: str | None = None
    mime_type: str | None = None
    integrity_verified: bool | None = None
    # Package-specific
    package: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    verified: bool | None = None
    verification: dict[str, Any] | None = None
    package_path: str | None = None
    includes_raw_data: bool | None = None
    contents: list[dict[str, Any]] | None = None
    # Data payload (for generic pass-through)
    data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Figure schemas
# ---------------------------------------------------------------------------

class FigureCreateRequest(BaseModel):
    """Request to create a new publication figure."""
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_id: str | None = None
    analysis_run_id: str | None = None
    result_fingerprint: str | None = None

    figure_type: str
    title: str = ""
    caption: str = ""

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
    output_format: str = "png"
    width_inches: float = 6.0
    height_inches: float = 4.0
    dpi: int = 300
    seed: int | None = None

    # Classification
    analytical_classification: str = "exploratory"
    prespecified_status: str = "unclear"

    # Metadata
    created_by: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_spec_dict(self) -> dict[str, Any]:
        """Convert to a FigureSpecification-compatible dict."""
        return {
            "project_id": self.project_id,
            "title": self.title,
            "caption": self.caption,
            "created_by": self.created_by,
            "metadata": self.metadata,
            "figure_type": self.figure_type,
            "dataset_id": self.dataset_id,
            "dataset_content_hash": self.dataset_content_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "analysis_version_id": self.analysis_version_id,
            "analysis_run_id": self.analysis_run_id,
            "result_fingerprint": self.result_fingerprint,
            "variable": self.variable,
            "grouping_variable": self.grouping_variable,
            "x_variable": self.x_variable,
            "y_variable": self.y_variable,
            "covariates": self.covariates,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "legend_label": self.legend_label,
            "units": self.units,
            "decimal_places": self.decimal_places,
            "category_order": self.category_order,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "output_format": self.output_format,
            "width_inches": self.width_inches,
            "height_inches": self.height_inches,
            "dpi": self.dpi,
            "seed": self.seed,
            "analytical_classification": self.analytical_classification,
            "prespecified_status": self.prespecified_status,
        }


# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------

class TableColumnSchema(BaseModel):
    key: str
    label: str
    width: int | None = None
    align: str = "left"


class TableRowSchema(BaseModel):
    key: str
    label: str
    level: int = 0
    bold: bool = False


class TableCreateRequest(BaseModel):
    """Request to create a new publication table."""
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_ids: list[str] = Field(default_factory=list)
    analysis_run_ids: list[str] = Field(default_factory=list)

    table_type: str
    title: str = ""
    caption: str = ""

    # Structure
    columns: list[TableColumnSchema] = Field(default_factory=list)
    rows: list[TableRowSchema] = Field(default_factory=list)
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
    output_format: str = "csv"
    include_significance_marks: bool = False

    # Classification
    analytical_classification: str = "exploratory"
    prespecified_status: str = "unclear"

    created_by: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_spec_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "caption": self.caption,
            "created_by": self.created_by,
            "metadata": self.metadata,
            "table_type": self.table_type,
            "dataset_id": self.dataset_id,
            "dataset_content_hash": self.dataset_content_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "analysis_version_ids": self.analysis_version_ids,
            "analysis_run_ids": self.analysis_run_ids,
            "columns": [c.model_dump() for c in self.columns],
            "rows": [r.model_dump() for r in self.rows],
            "variables": self.variables,
            "decimal_places": self.decimal_places,
            "statistical_notation": self.statistical_notation,
            "footnotes": self.footnotes,
            "include_p_values": self.include_p_values,
            "include_ci": self.include_ci,
            "include_n": self.include_n,
            "include_intercept": self.include_intercept,
            "output_format": self.output_format,
            "include_significance_marks": self.include_significance_marks,
            "analytical_classification": self.analytical_classification,
            "prespecified_status": self.prespecified_status,
        }


# ---------------------------------------------------------------------------
# Report schemas
# ---------------------------------------------------------------------------

class ReportSectionSchema(BaseModel):
    section_type: str
    title: str = ""
    content: str = ""
    include: bool = True
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class AuthorSchema(BaseModel):
    name: str = ""
    affiliation: str = ""
    email: str = ""


class ReportCreateRequest(BaseModel):
    """Request to create a new publication report."""
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_ids: list[str] = Field(default_factory=list)
    analysis_run_ids: list[str] = Field(default_factory=list)

    title: str
    caption: str = ""

    sections: list[ReportSectionSchema] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    authors: list[AuthorSchema] = Field(default_factory=list)

    output_format: str = "markdown"

    analytical_classification: str = "exploratory"
    prespecified_status: str = "unclear"

    created_by: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_spec_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "caption": self.caption,
            "created_by": self.created_by,
            "metadata": self.metadata,
            "dataset_id": self.dataset_id,
            "dataset_content_hash": self.dataset_content_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "analysis_version_ids": self.analysis_version_ids,
            "analysis_run_ids": self.analysis_run_ids,
            "sections": [s.model_dump() for s in self.sections],
            "figure_ids": self.figure_ids,
            "table_ids": self.table_ids,
            "authors": [a.model_dump() for a in self.authors],
            "output_format": self.output_format,
            "analytical_classification": self.analytical_classification,
            "prespecified_status": self.prespecified_status,
        }


# ---------------------------------------------------------------------------
# Reproducibility schemas
# ---------------------------------------------------------------------------

class ReproducibilityCreateRequest(BaseModel):
    """Request to create a reproducibility package."""
    package_id: str | None = None
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_run_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    output_provenances: list[dict[str, Any]] = Field(default_factory=list)
    random_seed: int | None = None
    include_raw_data: bool = False


class ReproducibilityVerifyRequest(BaseModel):
    """Request to verify a reproducibility package."""
    expected_dataset_hash: str | None = None


# ---------------------------------------------------------------------------
# Lifecycle request schemas
# ---------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    """Explicit approval request for an output."""
    approved_by: str = "researcher"
    notes: str = ""


class RenderRequest(BaseModel):
    """Optional render request (no body required, but schema for docs)."""
    pass


class ValidateRequest(BaseModel):
    """Optional validate request (no body required)."""
    pass
