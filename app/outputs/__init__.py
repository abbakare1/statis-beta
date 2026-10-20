"""Output Specification System — Phase 2G.

Strongly typed specifications for publication outputs (figures, tables, reports).
These specifications never recompute statistics; they consume persisted results.
"""

from app.outputs.specification import (
    OutputLifecycle,
    FigureType,
    TableType,
    ReportSection,
    AnalyticalClassification,
    OutputFormat,
    OutputSpecification,
    FigureSpecification,
    TableSpecification,
    TableColumn,
    TableRow,
    ReportSectionSpec,
    AuthorMetadata,
    ReportSpecification,
    OutputValidationResult,
    validate_figure_specification,
    validate_table_specification,
    validate_report_specification,
)

__all__ = [
    "OutputLifecycle",
    "FigureType",
    "TableType",
    "ReportSection",
    "AnalyticalClassification",
    "OutputFormat",
    "OutputSpecification",
    "FigureSpecification",
    "TableSpecification",
    "TableColumn",
    "TableRow",
    "ReportSectionSpec",
    "AuthorMetadata",
    "ReportSpecification",
    "OutputValidationResult",
    "validate_figure_specification",
    "validate_table_specification",
    "validate_report_specification",
]
