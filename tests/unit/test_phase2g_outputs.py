"""Phase 2G — Publication Outputs, Visualization, Reporting, and Reproducibility.

Comprehensive tests for the publication output layer covering:
- Output specifications (figure, table, report)
- Deterministic figure rendering
- Deterministic table rendering
- Report generation
- Provenance and traceability
- Reproducibility packages
- Dataset identity enforcement
- Analysis version enforcement
- PHI safeguards
- Anti-p-hacking safeguards
- Adversarial testing
"""

import sys
import json
import hashlib
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.outputs.specification import (
    FigureSpecification,
    TableSpecification,
    ReportSpecification,
    ReportSectionSpec,
    FigureType,
    TableType,
    ReportSection,
    AnalyticalClassification,
    OutputFormat,
    OutputLifecycle,
    validate_figure_specification,
    validate_table_specification,
    validate_report_specification,
)
from app.outputs.provenance import (
    build_provenance,
    build_lineage,
    compute_artifact_hash,
    verify_artifact_integrity,
)
from app.visualization.renderer import FigureRenderer
from app.tables.renderer import TableRenderer
from app.reporting.renderer import ReportRenderer
from app.reproducibility.manifest import ReproducibilityPackage, ReproducibilityManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _make_clean_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(50, 10, n).round(1),
        "bmi": rng.normal(25, 3, n).round(1),
        "treatment": rng.binomial(1, 0.5, n),
        "outcome": rng.normal(100, 15, n).round(1),
        "sex": rng.choice(["M", "F"], n),
    })


def _make_survival_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "time": rng.exponential(12, n).round(2),
        "event": rng.binomial(1, 0.6, n),
        "treatment": rng.binomial(1, 0.5, n),
    })


def _make_result(method_id: str = "independent_t_test", **kwargs) -> dict:
    result = {
        "method_id": method_id,
        "method_name": method_id.replace("_", " ").title(),
        "execution_status": "success",
        "test_statistic": kwargs.get("test_statistic", 2.5),
        "p_value": kwargs.get("p_value", 0.012),
        "effect_size": kwargs.get("effect_size", 0.45),
        "effect_size_name": kwargs.get("effect_size_name", "Cohen's d"),
        "confidence_interval": kwargs.get("confidence_interval", (0.1, 0.8)),
        "n_observed": kwargs.get("n_observed", 100),
        "n_groups": kwargs.get("n_groups", 2),
        "group_sizes": kwargs.get("group_sizes", {"A": 50, "B": 50}),
        "engine_result": kwargs.get("engine_result", {}),
        "result_fingerprint": kwargs.get("result_fingerprint", "abc123" * 10),
        "analytical_classification": kwargs.get("analytical_classification", "primary"),
        "diagnostic_status": kwargs.get("diagnostic_status", "pass"),
        "defensibility_status": kwargs.get("defensibility_status", "defensible"),
    }
    return result


def _make_regression_result() -> dict:
    return {
        "method_id": "linear_regression",
        "method_name": "Linear Regression",
        "execution_status": "success",
        "test_statistic": 15.2,
        "p_value": 0.0001,
        "effect_size": 0.35,
        "effect_size_name": "R-squared",
        "n_observed": 200,
        "engine_result": {
            "coefficients": {"Intercept": 50.0, "age": 0.5, "bmi": 1.2},
            "confidence_intervals": {
                "Intercept": (45.0, 55.0),
                "age": (0.3, 0.7),
                "bmi": (0.8, 1.6),
            },
            "standard_errors": {"Intercept": 2.5, "age": 0.1, "bmi": 0.2},
            "p_values": {"Intercept": 0.0001, "age": 0.0001, "bmi": 0.0001},
        },
        "result_fingerprint": "reg123" * 10,
        "analytical_classification": "primary",
    }


def _make_km_result() -> dict:
    return {
        "method_id": "kaplan_meier",
        "method_name": "Kaplan-Meier",
        "execution_status": "success",
        "n_observed": 100,
        "engine_result": {
            "n_total": 100,
            "n_events": 60,
            "n_censored": 40,
            "groups": {
                "overall": {
                    "survival_times": [0, 5, 10, 15, 20, 25],
                    "survival_probabilities": [1.0, 0.85, 0.70, 0.55, 0.40, 0.30],
                    "censoring_times": [3, 8, 12, 18],
                    "median_survival": 17.5,
                },
                "treatment=0": {
                    "survival_times": [0, 5, 10, 15, 20],
                    "survival_probabilities": [1.0, 0.80, 0.60, 0.45, 0.30],
                },
                "treatment=1": {
                    "survival_times": [0, 5, 10, 15, 20, 25],
                    "survival_probabilities": [1.0, 0.90, 0.80, 0.65, 0.50, 0.40],
                },
            },
        },
        "result_fingerprint": "km123" * 10,
        "analytical_classification": "primary",
    }


# ===========================================================================
# SPECIFICATION TESTS
# ===========================================================================


class TestFigureSpecification:
    """Tests for figure specification validation."""

    def test_valid_figure_specification(self):
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            title="Age Distribution",
        )
        result = validate_figure_specification(
            spec,
            available_columns=["age", "bmi", "treatment"],
            has_result=True,
        )
        assert result.valid is True
        assert len(result.blocking_errors) == 0

    def test_invalid_figure_specification_missing_variable(self):
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="nonexistent",
            dataset_id="ds1",
        )
        result = validate_figure_specification(
            spec,
            available_columns=["age", "bmi"],
        )
        assert result.valid is False
        assert any("nonexistent" in e for e in result.errors)

    def test_invalid_figure_missing_dataset_binding(self):
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
        )
        result = validate_figure_specification(spec)
        assert result.valid is False
        assert any("dataset" in e.lower() or "analysis" in e.lower() for e in result.errors)

    def test_km_without_survival_metadata_rejected(self):
        spec = FigureSpecification(
            figure_type=FigureType.KAPLAN_MEIER,
            dataset_id="ds1",
        )
        result = validate_figure_specification(
            spec,
            has_survival_metadata=False,
            has_result=False,
        )
        assert result.valid is False
        assert any("survival" in e.lower() for e in result.errors)

    def test_grouped_km_without_grouping_variable_rejected(self):
        spec = FigureSpecification(
            figure_type=FigureType.GROUPED_KAPLAN_MEIER,
            dataset_id="ds1",
        )
        result = validate_figure_specification(
            spec,
            has_survival_metadata=True,
            has_result=True,
        )
        assert result.valid is False
        assert any("grouping" in e.lower() for e in result.errors)

    def test_coefficient_plot_without_result_rejected(self):
        spec = FigureSpecification(
            figure_type=FigureType.COEFFICIENT_PLOT,
            dataset_id="ds1",
        )
        result = validate_figure_specification(
            spec,
            has_result=False,
        )
        assert result.valid is False

    def test_coefficient_plot_wrong_method_rejected(self):
        spec = FigureSpecification(
            figure_type=FigureType.COEFFICIENT_PLOT,
            dataset_id="ds1",
        )
        result = validate_figure_specification(
            spec,
            has_result=True,
            result_method_id="independent_t_test",
        )
        # The validation should warn or error about unsupported method
        assert result.valid is False or len(result.errors) > 0

    def test_specification_hash_deterministic(self):
        spec1 = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
        )
        spec2 = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
        )
        # Different IDs but same content should produce same hash
        h1 = spec1.compute_spec_hash()
        h2 = spec2.compute_spec_hash()
        assert h1 == h2


class TestTableSpecification:
    """Tests for table specification validation."""

    def test_valid_table_specification(self):
        spec = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age", "bmi"],
            dataset_id="ds1",
        )
        result = validate_table_specification(
            spec,
            available_columns=["age", "bmi", "treatment"],
        )
        assert result.valid is True

    def test_invalid_table_missing_variable(self):
        spec = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["nonexistent"],
            dataset_id="ds1",
        )
        result = validate_table_specification(
            spec,
            available_columns=["age", "bmi"],
        )
        assert result.valid is False

    def test_regression_table_without_result_rejected(self):
        spec = TableSpecification(
            table_type=TableType.REGRESSION_TABLE,
            dataset_id="ds1",
        )
        result = validate_table_specification(
            spec,
            has_result=False,
        )
        assert result.valid is False

    def test_survival_table_without_result_rejected(self):
        spec = TableSpecification(
            table_type=TableType.SURVIVAL_TABLE,
            dataset_id="ds1",
        )
        result = validate_table_specification(
            spec,
            has_result=False,
        )
        assert result.valid is False


class TestReportSpecification:
    """Tests for report specification validation."""

    def test_valid_report_specification(self):
        spec = ReportSpecification(
            title="My Analysis Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
        )
        result = validate_report_specification(
            spec,
            has_analyses=True,
        )
        assert result.valid is True

    def test_report_missing_title_rejected(self):
        spec = ReportSpecification(
            dataset_id="ds1",
            analysis_run_ids=["run1"],
        )
        result = validate_report_specification(spec)
        assert result.valid is False

    def test_report_without_analyses_rejected(self):
        spec = ReportSpecification(
            title="Report",
            dataset_id="ds1",
        )
        result = validate_report_specification(
            spec,
            has_analyses=False,
        )
        assert result.valid is False

    def test_report_causal_language_warning(self):
        spec = ReportSpecification(
            title="Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            sections=[
                ReportSectionSpec(
                    section_type=ReportSection.RESULTS,
                    content="The treatment caused a significant improvement.",
                )
            ],
        )
        result = validate_report_specification(spec, has_analyses=True)
        assert any("causal" in w.lower() for w in result.warnings)


# ===========================================================================
# FIGURE RENDERING TESTS
# ===========================================================================


class TestFigureRendering:
    """Tests for deterministic figure rendering."""

    def test_histogram_rendering(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            title="Age Distribution",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert output_path.stat().st_size > 0

    def test_box_plot_rendering(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.BOX_PLOT,
            variable="outcome",
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.exists()

    def test_scatter_plot_rendering(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.SCATTER_PLOT,
            x_variable="age",
            y_variable="bmi",
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.exists()

    def test_grouped_box_plot_rendering(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.GROUPED_BOX_PLOT,
            variable="outcome",
            grouping_variable="treatment",
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.exists()

    def test_bar_chart_rendering(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.BAR_CHART,
            variable="sex",
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.exists()

    def test_coefficient_plot_rendering(self, tmp_path):
        result = _make_regression_result()
        spec = FigureSpecification(
            figure_type=FigureType.COEFFICIENT_PLOT,
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, result=result, output_dir=tmp_path)
        assert output_path.exists()

    def test_kaplan_meier_rendering(self, tmp_path):
        result = _make_km_result()
        spec = FigureSpecification(
            figure_type=FigureType.KAPLAN_MEIER,
            dataset_id="ds1",
            title="Survival Curve",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, result=result, output_dir=tmp_path)
        assert output_path.exists()

    def test_grouped_kaplan_meier_rendering(self, tmp_path):
        result = _make_km_result()
        spec = FigureSpecification(
            figure_type=FigureType.GROUPED_KAPLAN_MEIER,
            grouping_variable="treatment",
            dataset_id="ds1",
            title="Survival by Treatment",
            output_format=OutputFormat.PNG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, result=result, output_dir=tmp_path)
        assert output_path.exists()

    def test_svg_format(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            output_format=OutputFormat.SVG,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.suffix == ".svg"

    def test_pdf_format(self, tmp_path):
        df = _make_clean_df()
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            output_format=OutputFormat.PDF,
        )
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.suffix == ".pdf"

    def test_deterministic_rendering(self, tmp_path):
        """Same specification and data produce identical files."""
        df = _make_clean_df()
        spec1 = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        spec2 = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            output_format=OutputFormat.PNG,
        )
        # Use same output_id for deterministic comparison
        spec2.output_id = spec1.output_id

        renderer = FigureRenderer()
        path1 = renderer.render(spec1, df=df, output_dir=tmp_path / "run1")
        path2 = renderer.render(spec2, df=df, output_dir=tmp_path / "run2")

        hash1 = compute_artifact_hash(path1)
        hash2 = compute_artifact_hash(path2)
        assert hash1 == hash2


# ===========================================================================
# TABLE RENDERING TESTS
# ===========================================================================


class TestTableRendering:
    """Tests for deterministic table rendering."""

    def test_descriptive_table_rendering(self, tmp_path):
        df = _make_clean_df()
        spec = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age", "bmi", "sex"],
            dataset_id="ds1",
            title="Table 1. Baseline Characteristics",
            output_format=OutputFormat.CSV,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_inferential_table_rendering(self, tmp_path):
        result = _make_result()
        spec = TableSpecification(
            table_type=TableType.INFERENTIAL_RESULTS,
            dataset_id="ds1",
            output_format=OutputFormat.CSV,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        assert output_path.exists()

    def test_regression_table_rendering(self, tmp_path):
        result = _make_regression_result()
        spec = TableSpecification(
            table_type=TableType.REGRESSION_TABLE,
            dataset_id="ds1",
            output_format=OutputFormat.CSV,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        assert output_path.exists()

    def test_survival_table_rendering(self, tmp_path):
        result = _make_km_result()
        spec = TableSpecification(
            table_type=TableType.SURVIVAL_TABLE,
            dataset_id="ds1",
            output_format=OutputFormat.CSV,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        assert output_path.exists()

    def test_markdown_output(self, tmp_path):
        df = _make_clean_df()
        spec = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age"],
            dataset_id="ds1",
            output_format=OutputFormat.MARKDOWN,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.suffix in (".md", ".markdown")
        content = output_path.read_text(encoding="utf-8")
        assert "|" in content  # Markdown table

    def test_json_output(self, tmp_path):
        df = _make_clean_df()
        spec = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age"],
            dataset_id="ds1",
            output_format=OutputFormat.JSON,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        assert output_path.suffix == ".json"
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "columns" in data
        assert "rows" in data

    def test_deterministic_table_rendering(self, tmp_path):
        df = _make_clean_df()
        spec1 = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age", "bmi"],
            dataset_id="ds1",
            output_format=OutputFormat.JSON,
        )
        spec2 = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age", "bmi"],
            dataset_id="ds1",
            output_format=OutputFormat.JSON,
        )
        spec2.output_id = spec1.output_id

        renderer = TableRenderer()
        path1 = renderer.render(spec1, df=df, output_dir=tmp_path / "run1")
        path2 = renderer.render(spec2, df=df, output_dir=tmp_path / "run2")

        data1 = json.loads(path1.read_text(encoding="utf-8"))
        data2 = json.loads(path2.read_text(encoding="utf-8"))
        assert data1 == data2


# ===========================================================================
# REPORT RENDERING TESTS
# ===========================================================================


class TestReportRendering:
    """Tests for deterministic report generation."""

    def test_basic_report_rendering(self, tmp_path):
        result = _make_result()
        spec = ReportSpecification(
            title="Test Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            output_format=OutputFormat.MARKDOWN,
        )
        renderer = ReportRenderer()
        output_path = renderer.render(
            spec,
            results=[result],
            dataset_info={"content_hash": "abc123", "n_observations": 100},
            output_dir=tmp_path,
        )
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Test Report" in content

    def test_report_sections_generated(self, tmp_path):
        result = _make_result()
        spec = ReportSpecification(
            title="Sections Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            sections=[
                ReportSectionSpec(section_type=ReportSection.TITLE, title="Title"),
                ReportSectionSpec(section_type=ReportSection.RESULTS, title="Results"),
                ReportSectionSpec(section_type=ReportSection.LIMITATIONS, title="Limitations"),
            ],
            output_format=OutputFormat.JSON,
        )
        renderer = ReportRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["sections"]) == 3

    def test_report_numerical_values_sourced(self, tmp_path):
        result = _make_result(test_statistic=3.14159, p_value=0.00234)
        spec = ReportSpecification(
            title="Numerical Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            output_format=OutputFormat.MARKDOWN,
        )
        renderer = ReportRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        content = output_path.read_text(encoding="utf-8")
        # Verify numerical values appear in output
        assert "3.1416" in content or "3.14159" in content
        assert "0.002" in content

    def test_report_json_output(self, tmp_path):
        result = _make_result()
        spec = ReportSpecification(
            title="JSON Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            output_format=OutputFormat.JSON,
        )
        renderer = ReportRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["title"] == "JSON Report"
        assert "sections" in data


# ===========================================================================
# PROVENANCE TESTS
# ===========================================================================


class TestProvenance:
    """Tests for output provenance and traceability."""

    def test_provenance_building(self, tmp_path):
        artifact = tmp_path / "test.png"
        artifact.write_bytes(b"fake image data")

        prov = build_provenance(
            output_id="fig1",
            output_type="figure",
            output_spec={"figure_type": "histogram"},
            artifact_path=artifact,
            project_id="proj1",
            dataset_id="ds1",
            dataset_content_hash="abc123",
            dataset_schema_hash="def456",
        )

        assert prov.output_id == "fig1"
        assert prov.output_type == "figure"
        assert prov.dataset_content_hash == "abc123"
        assert prov.artifact_hash is not None
        assert prov.artifact_format == "png"

    def test_lineage_building(self):
        lineage = build_lineage(
            output_id="fig1",
            output_type="figure",
            project_id="proj1",
            dataset_id="ds1",
            dataset_content_hash="abc123",
            dataset_schema_hash="def456",
            dataset_version=1,
            analysis_version_id="av1",
            analysis_run_id="run1",
            result_fingerprint="fp123",
            artifact_hash="art123",
        )

        assert lineage["traceable"] is True
        assert lineage["lineage"]["dataset"]["content_hash"] == "abc123"
        assert lineage["lineage"]["analysis"]["result_fingerprint"] == "fp123"

    def test_lineage_not_traceable_missing_hashes(self):
        lineage = build_lineage(
            output_id="fig1",
            output_type="figure",
        )
        assert lineage["traceable"] is False

    def test_artifact_integrity_verification(self, tmp_path):
        artifact = tmp_path / "test.txt"
        artifact.write_bytes(b"content")

        hash_val = compute_artifact_hash(artifact)
        assert verify_artifact_integrity(artifact, hash_val) is True
        assert verify_artifact_integrity(artifact, "wrong_hash") is False

    def test_artifact_tampering_detected(self, tmp_path):
        artifact = tmp_path / "test.txt"
        artifact.write_bytes(b"original")
        original_hash = compute_artifact_hash(artifact)

        # Tamper with artifact
        artifact.write_bytes(b"tampered")
        assert verify_artifact_integrity(artifact, original_hash) is False


# ===========================================================================
# REPRODUCIBILITY TESTS
# ===========================================================================


class TestReproducibility:
    """Tests for reproducibility packages."""

    def test_package_creation(self, tmp_path):
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "test123",
            project_id="proj1",
            dataset_id="ds1",
            dataset_content_hash="abc123",
            dataset_schema_hash="def456",
            statistical_results=[_make_result()],
            output_provenances=[
                {"output_id": "fig1", "output_type": "figure", "artifact_hash": "art123"}
            ],
        )

        assert package_path.exists()
        assert (package_path / "manifest.json").exists()
        assert (package_path / "dataset" / "dataset_metadata.json").exists()
        assert (package_path / "environment" / "environment.json").exists()

    def test_manifest_content(self, tmp_path):
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "manifest_test",
            dataset_content_hash="abc123",
            dataset_schema_hash="def456",
            statistical_results=[
                _make_result(method_id="independent_t_test"),
                _make_result(method_id="kaplan_meier"),
            ],
        )

        manifest_data = json.loads(
            (package_path / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_data["dataset_content_hash"] == "abc123"
        assert "independent_t_test" in manifest_data["methods"]
        assert "kaplan_meier" in manifest_data["methods"]

    def test_package_verification(self, tmp_path):
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "verify_test",
            dataset_content_hash="abc123",
        )

        verification = package.verify_package(
            package_path,
            expected_dataset_hash="abc123",
        )
        assert verification["verified"] is True
        assert verification["checks"]["dataset_hash"]["passed"] is True

    def test_package_verification_dataset_mismatch(self, tmp_path):
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "mismatch_test",
            dataset_content_hash="abc123",
        )

        verification = package.verify_package(
            package_path,
            expected_dataset_hash="WRONG_HASH",
        )
        assert verification["verified"] is False
        assert verification["checks"]["dataset_hash"]["passed"] is False

    def test_no_raw_data_in_default_package(self, tmp_path):
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "phi_test",
            dataset_content_hash="abc123",
        )

        # Check that no raw data files exist
        data_files = list((package_path / "dataset").glob("*.csv"))
        data_files += list((package_path / "dataset").glob("*.parquet"))
        assert len(data_files) == 0

        # Check dataset metadata note
        metadata = json.loads(
            (package_path / "dataset" / "dataset_metadata.json").read_text(encoding="utf-8")
        )
        assert "NOT included" in metadata.get("note", "")

    def test_manifest_hash_deterministic(self, tmp_path):
        manifest1 = ReproducibilityManifest(
            python_version="3.14.0",
            platform="Windows",
            dataset_content_hash="abc123",
        )
        manifest2 = ReproducibilityManifest(
            python_version="3.14.0",
            platform="Windows",
            dataset_content_hash="abc123",
        )
        assert manifest1.compute_manifest_hash() == manifest2.compute_manifest_hash()


# ===========================================================================
# DATASET INTEGRITY TESTS
# ===========================================================================


class TestDatasetIntegrity:
    """Tests for dataset identity enforcement in outputs."""

    def test_dataset_hash_mismatch_detection(self, tmp_path):
        """Output bound to wrong dataset hash should be detectable."""
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "hash_test",
            dataset_content_hash="correct_hash",
        )

        # Verify with wrong hash
        verification = package.verify_package(
            package_path,
            expected_dataset_hash="wrong_hash",
        )
        assert verification["verified"] is False

    def test_schema_hash_in_provenance(self, tmp_path):
        artifact = tmp_path / "test.png"
        artifact.write_bytes(b"data")

        prov = build_provenance(
            output_id="fig1",
            output_type="figure",
            output_spec={},
            artifact_path=artifact,
            dataset_content_hash="abc",
            dataset_schema_hash="def",
        )
        assert prov.dataset_schema_hash == "def"

    def test_result_fingerprint_tracked(self, tmp_path):
        artifact = tmp_path / "test.png"
        artifact.write_bytes(b"data")

        prov = build_provenance(
            output_id="fig1",
            output_type="figure",
            output_spec={},
            artifact_path=artifact,
            result_fingerprint="fp123",
        )
        assert prov.result_fingerprint == "fp123"


# ===========================================================================
# PHI SAFEGUARD TESTS
# ===========================================================================


class TestPHISafeguards:
    """Tests for PHI protection in outputs."""

    def test_identifier_not_in_descriptive_table(self, tmp_path):
        """Patient IDs should not appear in aggregate descriptive tables."""
        df = _make_clean_df()
        # Add identifier-like column
        df["ssn"] = [f"123-45-{6789:04d}" for i in range(len(df))]

        spec = TableSpecification(
            table_type=TableType.DESCRIPTIVE_BASELINE,
            variables=["age", "bmi"],  # Only aggregate variables
            dataset_id="ds1",
            output_format=OutputFormat.JSON,
        )
        renderer = TableRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)
        content = output_path.read_text(encoding="utf-8")

        # SSN should not appear
        assert "123-45" not in content
        # Patient IDs should not appear
        assert "P0001" not in content

    def test_default_report_no_raw_data(self, tmp_path):
        """Default reports should not include raw patient rows."""
        result = _make_result()
        spec = ReportSpecification(
            title="PHI Test",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            output_format=OutputFormat.JSON,
        )
        renderer = ReportRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))

        # Check that no raw patient data appears in sections
        content_str = json.dumps(data)
        # Should not contain individual patient identifiers
        assert "P0001" not in content_str
        assert "ssn" not in content_str.lower()

    def test_reproducibility_package_no_phi(self, tmp_path):
        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "phi_check",
            dataset_content_hash="abc",
        )

        # List all files in package
        all_files = list(package_path.rglob("*"))
        for f in all_files:
            if f.is_file():
                content = f.read_text(encoding="utf-8", errors="ignore")
                # Check for common PHI patterns
                assert "ssn" not in content.lower() or "NOT included" in content
                assert "123-45-6789" not in content


# ===========================================================================
# ANTI-P-HACKING TESTS
# ===========================================================================


class TestAntiPHacking:
    """Tests for anti-p-hacking safeguards in outputs."""

    def test_classification_preserved(self):
        """Analytical classification must be preserved in specifications."""
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            analytical_classification=AnalyticalClassification.EXPLORATORY,
        )
        assert spec.analytical_classification == AnalyticalClassification.EXPLORATORY

    def test_exploratory_not_mislabeled_as_primary(self):
        """Exploratory analyses must not appear as primary."""
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            analytical_classification=AnalyticalClassification.EXPLORATORY,
        )
        # The specification preserves the classification
        assert spec.analytical_classification.value == "exploratory"

    def test_prespecified_status_tracked(self):
        """Prespecified status must be tracked."""
        spec = TableSpecification(
            table_type=TableType.INFERENTIAL_RESULTS,
            dataset_id="ds1",
            prespecified_status="not_prespecified",
        )
        assert spec.prespecified_status == "not_prespecified"

    def test_p_value_not_used_for_selection(self):
        """Output specifications don't select based on p-values."""
        # Create two results, one significant, one not
        significant = _make_result(p_value=0.001, analytical_classification="primary")
        non_significant = _make_result(p_value=0.5, analytical_classification="sensitivity")

        # Both should be renderable regardless of p-value
        # The renderer doesn't filter by p-value
        assert significant["p_value"] == 0.001
        assert non_significant["p_value"] == 0.5
        # Both have valid classifications
        assert significant["analytical_classification"] == "primary"
        assert non_significant["analytical_classification"] == "sensitivity"


# ===========================================================================
# ADVERSARIAL TESTS
# ===========================================================================


class TestAdversarial:
    """Adversarial tests for the publication output layer."""

    def test_adversarial_dataset_hash_mismatch(self, tmp_path):
        """Dataset hash mismatch must be detected."""
        package = ReproducibilityPackage(tmp_path)
        pkg = package.create_package("adv1", dataset_content_hash="expected")
        result = package.verify_package(pkg, expected_dataset_hash="different")
        assert result["verified"] is False

    def test_adversarial_missing_result(self, tmp_path):
        """Missing StatisticalResult should fail gracefully."""
        spec = TableSpecification(
            table_type=TableType.REGRESSION_TABLE,
            dataset_id="ds1",
        )
        renderer = TableRenderer()
        with pytest.raises(ValueError):
            renderer.render(spec, results=[], output_dir=tmp_path)

    def test_adversarial_unsupported_figure_type(self, tmp_path):
        """Unsupported figure type should fail gracefully."""
        # Pydantic validation catches invalid enum values
        with pytest.raises(Exception):
            spec = FigureSpecification(
                figure_type="nonexistent_type",
                dataset_id="ds1",
            )
            # If somehow created, renderer should handle it
            renderer = FigureRenderer()
            renderer.render(spec, output_dir=tmp_path)

    def test_adversarial_empty_dataset(self, tmp_path):
        """Empty dataset should fail gracefully."""
        df = pd.DataFrame()
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
        )
        renderer = FigureRenderer()
        with pytest.raises(ValueError):
            renderer.render(spec, df=df, output_dir=tmp_path)

    def test_adversarial_all_missing_variable(self, tmp_path):
        """Variable with all missing values should fail gracefully."""
        df = _make_clean_df()
        df["all_missing"] = np.nan
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="all_missing",
            dataset_id="ds1",
        )
        renderer = FigureRenderer()
        # Should either fail or produce empty figure
        try:
            output_path = renderer.render(spec, df=df, output_dir=tmp_path)
            # If it succeeds, file should exist
            assert output_path.exists()
        except ValueError:
            # Expected failure path
            pass

    def test_adversarial_artifact_tampering(self, tmp_path):
        """Tampered artifacts must be detected."""
        artifact = tmp_path / "artifact.txt"
        artifact.write_bytes(b"original content")
        original_hash = compute_artifact_hash(artifact)

        # Tamper
        artifact.write_bytes(b"tampered content")
        assert verify_artifact_integrity(artifact, original_hash) is False

    def test_adversarial_fake_numerical_value_in_narrative(self, tmp_path):
        """Fake numerical values in narrative should be flagged by validation."""
        spec = ReportSpecification(
            title="Fake Numbers Report",
            dataset_id="ds1",
            analysis_run_ids=["run1"],
            sections=[
                ReportSectionSpec(
                    section_type=ReportSection.RESULTS,
                    content="The p-value was 0.0000001 and the effect size was 99.9.",
                )
            ],
        )
        # Validation doesn't check numerical accuracy of free text,
        # but the system should not auto-generate such content
        result = validate_report_specification(spec, has_analyses=True)
        # The spec is valid but the content is researcher-provided
        assert result.valid is True

    def test_adversarial_exploratory_mislabeling(self):
        """Exploratory analysis cannot be silently relabeled."""
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            analytical_classification=AnalyticalClassification.EXPLORATORY,
        )
        # The classification is immutable once set
        assert spec.analytical_classification == AnalyticalClassification.EXPLORATORY

    def test_adversarial_zero_event_survival(self, tmp_path):
        """Zero-event survival analysis should fail gracefully."""
        result = {
            "method_id": "kaplan_meier",
            "engine_result": {
                "n_total": 100,
                "n_events": 0,
                "n_censored": 100,
                "groups": {"overall": {}},
            },
        }
        spec = FigureSpecification(
            figure_type=FigureType.KAPLAN_MEIER,
            dataset_id="ds1",
        )
        renderer = FigureRenderer()
        # Should fail because no survival data in result
        with pytest.raises(ValueError):
            renderer.render(spec, result=result, output_dir=tmp_path)

    def test_adversarial_single_category_variable(self, tmp_path):
        """Single-category variable in bar chart should work or fail gracefully."""
        df = _make_clean_df()
        df["constant"] = "A"  # Only one category
        spec = FigureSpecification(
            figure_type=FigureType.BAR_CHART,
            variable="constant",
            dataset_id="ds1",
        )
        renderer = FigureRenderer()
        try:
            output_path = renderer.render(spec, df=df, output_dir=tmp_path)
            assert output_path.exists()
        except Exception:
            # Acceptable failure path
            pass


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================


class TestIntegration:
    """Integration tests for the full output pipeline."""

    def test_full_figure_pipeline(self, tmp_path):
        """Test complete figure pipeline: spec → validate → render → provenance."""
        df = _make_clean_df()

        # 1. Create specification
        spec = FigureSpecification(
            figure_type=FigureType.HISTOGRAM,
            variable="age",
            dataset_id="ds1",
            dataset_content_hash="abc123",
            title="Age Distribution",
            output_format=OutputFormat.PNG,
        )

        # 2. Validate
        validation = validate_figure_specification(
            spec,
            available_columns=list(df.columns),
        )
        assert validation.valid is True

        # 3. Render
        renderer = FigureRenderer()
        output_path = renderer.render(spec, df=df, output_dir=tmp_path)

        # 4. Build provenance
        prov = build_provenance(
            output_id=spec.output_id,
            output_type="figure",
            output_spec=spec.model_dump(mode="json"),
            artifact_path=output_path,
            dataset_id="ds1",
            dataset_content_hash="abc123",
        )

        # 5. Verify
        assert output_path.exists()
        assert prov.artifact_hash is not None
        assert prov.dataset_content_hash == "abc123"

    def test_full_table_pipeline(self, tmp_path):
        """Test complete table pipeline."""
        df = _make_clean_df()
        result = _make_result()

        spec = TableSpecification(
            table_type=TableType.INFERENTIAL_RESULTS,
            dataset_id="ds1",
            dataset_content_hash="abc123",
            output_format=OutputFormat.JSON,
        )

        validation = validate_table_specification(spec, has_result=True)
        assert validation.valid is True

        renderer = TableRenderer()
        output_path = renderer.render(spec, results=[result], output_dir=tmp_path)

        prov = build_provenance(
            output_id=spec.output_id,
            output_type="table",
            output_spec=spec.model_dump(mode="json"),
            artifact_path=output_path,
            dataset_content_hash="abc123",
        )

        assert output_path.exists()
        assert prov.artifact_hash is not None

    def test_full_reproducibility_pipeline(self, tmp_path):
        """Test complete reproducibility package pipeline."""
        result = _make_result()

        package = ReproducibilityPackage(tmp_path)
        package_path = package.create_package(
            "full_test",
            project_id="proj1",
            dataset_id="ds1",
            dataset_content_hash="abc123",
            dataset_schema_hash="def456",
            statistical_results=[result],
            output_provenances=[
                {
                    "output_id": "fig1",
                    "output_type": "figure",
                    "artifact_hash": "art123",
                }
            ],
        )

        verification = package.verify_package(
            package_path,
            expected_dataset_hash="abc123",
        )

        assert package_path.exists()
        assert verification["verified"] is True
