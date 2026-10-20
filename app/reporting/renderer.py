"""Report Renderer — deterministic structured report generation.

Phase 2G: Generates structured reports from validated statistical results.
Narrative content is constrained by deterministic results; AI-generated
text is clearly labeled and never overwrites researcher interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.outputs.specification import (
    ReportSpecification,
    ReportSection,
    ReportSectionSpec,
    OutputFormat,
)


class ReportRenderer:
    """Deterministic report renderer."""

    def render(
        self,
        spec: ReportSpecification,
        *,
        results: list[dict[str, Any]] | None = None,
        dataset_info: dict[str, Any] | None = None,
        quality_info: dict[str, Any] | None = None,
        output_dir: Path,
    ) -> Path:
        """Render a report based on specification.

        Returns the path to the generated artifact.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build report content
        report_content = self._build_report(
            spec,
            results=results,
            dataset_info=dataset_info,
            quality_info=quality_info,
        )

        # Save
        output_path = output_dir / f"{spec.output_id}.{spec.output_format.value}"
        self._save_report(report_content, spec, output_path)

        return output_path

    def _build_report(
        self,
        spec: ReportSpecification,
        *,
        results: list[dict[str, Any]] | None,
        dataset_info: dict[str, Any] | None,
        quality_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build structured report content."""
        sections = []

        # Use specified sections or generate default
        if spec.sections:
            for section_spec in spec.sections:
                if not section_spec.include:
                    continue
                content = self._generate_section_content(
                    section_spec,
                    results=results,
                    dataset_info=dataset_info,
                    quality_info=quality_info,
                )
                sections.append({
                    "section_type": section_spec.section_type.value,
                    "title": section_spec.title or section_spec.section_type.value.replace("_", " ").title(),
                    "content": content,
                })
        else:
            # Generate default structure
            default_sections = [
                ReportSection.TITLE,
                ReportSection.RESEARCH_QUESTION,
                ReportSection.STUDY_DESIGN,
                ReportSection.DATA_SOURCE,
                ReportSection.COHORT_DESCRIPTION,
                ReportSection.STATISTICAL_METHODS,
                ReportSection.RESULTS,
                ReportSection.LIMITATIONS,
                ReportSection.REPRODUCIBILITY,
            ]
            for section_type in default_sections:
                content = self._generate_section_content(
                    ReportSectionSpec(section_type=section_type),
                    results=results,
                    dataset_info=dataset_info,
                    quality_info=quality_info,
                )
                sections.append({
                    "section_type": section_type.value,
                    "title": section_type.value.replace("_", " ").title(),
                    "content": content,
                })

        return {
            "title": spec.title,
            "sections": sections,
            "authors": [a.model_dump() for a in spec.authors],
            "classification": spec.analytical_classification.value,
            "prespecified_status": spec.prespecified_status,
            "dataset_id": spec.dataset_id,
            "analysis_run_ids": spec.analysis_run_ids,
        }

    def _generate_section_content(
        self,
        section_spec: ReportSectionSpec,
        *,
        results: list[dict[str, Any]] | None,
        dataset_info: dict[str, Any] | None,
        quality_info: dict[str, Any] | None,
    ) -> str:
        """Generate content for a report section."""
        section_type = section_spec.section_type

        if section_spec.content:
            # Researcher-provided content takes precedence
            return section_spec.content

        # Generate deterministic content from results
        if section_type == ReportSection.TITLE:
            return ""

        elif section_type == ReportSection.ABSTRACT:
            return self._generate_abstract(results, dataset_info)

        elif section_type == ReportSection.RESEARCH_QUESTION:
            if dataset_info and dataset_info.get("research_question"):
                return dataset_info["research_question"]
            return "[Research question not specified]"

        elif section_type == ReportSection.STUDY_DESIGN:
            if dataset_info and dataset_info.get("study_design"):
                return dataset_info["study_design"]
            return "[Study design not specified]"

        elif section_type == ReportSection.DATA_SOURCE:
            return self._generate_data_source(dataset_info)

        elif section_type == ReportSection.COHORT_DESCRIPTION:
            return self._generate_cohort_description(results, dataset_info)

        elif section_type == ReportSection.DATA_QUALITY:
            return self._generate_data_quality(quality_info)

        elif section_type == ReportSection.MISSINGNESS:
            return self._generate_missingness(quality_info)

        elif section_type == ReportSection.STATISTICAL_METHODS:
            return self._generate_statistical_methods(results)

        elif section_type == ReportSection.RESULTS:
            return self._generate_results(results)

        elif section_type == ReportSection.PRIMARY_ANALYSIS:
            return self._generate_primary_analysis(results)

        elif section_type == ReportSection.SECONDARY_ANALALYSES:
            return self._generate_analysis_by_classification(results, "secondary")

        elif section_type == ReportSection.SENSITIVITY_ANALYSES:
            return self._generate_analysis_by_classification(results, "sensitivity")

        elif section_type == ReportSection.EXPLORATORY_ANALYSES:
            return self._generate_analysis_by_classification(results, "exploratory")

        elif section_type == ReportSection.LIMITATIONS:
            return self._generate_limitations(results)

        elif section_type == ReportSection.REPRODUCIBILITY:
            return self._generate_reproducibility_info(results, dataset_info)

        else:
            return ""

    def _generate_abstract(
        self,
        results: list[dict[str, Any]] | None,
        dataset_info: dict[str, Any] | None,
    ) -> str:
        """Generate abstract from deterministic results."""
        parts = []

        if dataset_info and dataset_info.get("n_observations"):
            parts.append(
                f"This analysis was conducted on a dataset with "
                f"{dataset_info['n_observations']} observations."
            )

        if results:
            methods = [r.get("method_id", "") for r in results if r.get("method_id")]
            if methods:
                parts.append(
                    f"Statistical methods included: {', '.join(methods)}."
                )

        if not parts:
            return "[Abstract requires additional information]"

        return " ".join(parts)

    def _generate_data_source(self, dataset_info: dict[str, Any] | None) -> str:
        """Generate data source description."""
        if not dataset_info:
            return "[Data source information not available]"

        parts = []
        if dataset_info.get("content_hash"):
            parts.append(f"Dataset content hash: `{dataset_info['content_hash'][:16]}...`")
        if dataset_info.get("schema_hash"):
            parts.append(f"Schema hash: `{dataset_info['schema_hash'][:16]}...`")
        if dataset_info.get("n_rows"):
            parts.append(f"Rows: {dataset_info['n_rows']}")
        if dataset_info.get("n_cols"):
            parts.append(f"Columns: {dataset_info['n_cols']}")

        return "\n".join(parts) if parts else "[Data source information not available]"

    def _generate_cohort_description(
        self,
        results: list[dict[str, Any]] | None,
        dataset_info: dict[str, Any] | None,
    ) -> str:
        """Generate cohort description from results."""
        if not results:
            return "[Cohort description requires results]"

        parts = []
        for result in results:
            n = result.get("n_observed")
            if n:
                method = result.get("method_id", "Analysis")
                parts.append(f"{method}: N = {n}")

        return "\n".join(parts) if parts else "[Cohort description requires results]"

    def _generate_data_quality(self, quality_info: dict[str, Any] | None) -> str:
        """Generate data quality section."""
        if not quality_info:
            return "[Data quality information not available]"

        parts = []
        if quality_info.get("n_duplicates"):
            parts.append(f"Duplicate rows: {quality_info['n_duplicates']}")
        if quality_info.get("n_invalid_values"):
            parts.append(f"Invalid values: {quality_info['n_invalid_values']}")

        return "\n".join(parts) if parts else "[No data quality issues detected]"

    def _generate_missingness(self, quality_info: dict[str, Any] | None) -> str:
        """Generate missingness section."""
        if not quality_info or not quality_info.get("missingness"):
            return "[Missingness information not available]"

        parts = ["Variable-level missingness:"]
        for var, pct in quality_info["missingness"].items():
            parts.append(f"- {var}: {pct:.1f}%")

        return "\n".join(parts)

    def _generate_statistical_methods(self, results: list[dict[str, Any]] | None) -> str:
        """Generate statistical methods section."""
        if not results:
            return "[No statistical methods specified]"

        parts = []
        for result in results:
            method = result.get("method_id", "")
            method_name = result.get("method_name", method)
            parts.append(f"- {method_name} (`{method}`)")

        return "\n".join(parts)

    def _generate_results(self, results: list[dict[str, Any]] | None) -> str:
        """Generate results section from deterministic values."""
        if not results:
            return "[No results available]"

        parts = []
        for result in results:
            method = result.get("method_id", "")
            parts.append(f"### {method}")

            # Test statistic
            if result.get("test_statistic") is not None:
                parts.append(f"- Test statistic: {result['test_statistic']:.4f}")

            # P-value
            if result.get("p_value") is not None:
                p = result["p_value"]
                if p < 0.001:
                    parts.append(f"- P value: <0.001")
                else:
                    parts.append(f"- P value: {p:.3f}")

            # Effect size
            if result.get("effect_size") is not None:
                effect_name = result.get("effect_size_name", "Effect size")
                parts.append(f"- {effect_name}: {result['effect_size']:.4f}")

            # Confidence interval
            if result.get("confidence_interval"):
                ci = result["confidence_interval"]
                parts.append(f"- 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

            # Sample size
            if result.get("n_observed"):
                parts.append(f"- N: {result['n_observed']}")

            parts.append("")

        return "\n".join(parts)

    def _generate_primary_analysis(self, results: list[dict[str, Any]] | None) -> str:
        """Generate primary analysis section."""
        primary = [
            r for r in (results or [])
            if r.get("analytical_classification") == "primary"
        ]
        if not primary:
            return "[No primary analysis specified]"

        return self._generate_results(primary)

    def _generate_analysis_by_classification(
        self,
        results: list[dict[str, Any]] | None,
        classification: str,
    ) -> str:
        """Generate section for analyses of a specific classification."""
        filtered = [
            r for r in (results or [])
            if r.get("analytical_classification") == classification
        ]
        if not filtered:
            return f"[No {classification} analyses]"

        return self._generate_results(filtered)

    def _generate_limitations(self, results: list[dict[str, Any]] | None) -> str:
        """Generate limitations section."""
        parts = ["The following limitations should be considered:"]

        if results:
            # Check for diagnostics warnings
            for result in results:
                if result.get("diagnostic_status") == "warning":
                    parts.append("- Diagnostic warnings were present for some analyses")
                if result.get("defensibility_status") == "defensible_with_caveats":
                    parts.append("- Some analyses have methodological caveats")

        parts.append("- Results should be interpreted in the context of the study design")
        parts.append("- Multiple comparisons may affect Type I error rates")

        return "\n".join(parts)

    def _generate_reproducibility_info(
        self,
        results: list[dict[str, Any]] | None,
        dataset_info: dict[str, Any] | None,
    ) -> str:
        """Generate reproducibility information."""
        parts = ["### Reproducibility Information"]

        if dataset_info:
            if dataset_info.get("content_hash"):
                parts.append(f"- Dataset content hash: `{dataset_info['content_hash']}`")
            if dataset_info.get("schema_hash"):
                parts.append(f"- Schema hash: `{dataset_info['schema_hash']}`")

        if results:
            parts.append("\n### Analysis Fingerprints")
            for result in results:
                if result.get("result_fingerprint"):
                    parts.append(
                        f"- {result.get('method_id', 'Analysis')}: "
                        f"`{result['result_fingerprint'][:16]}...`"
                    )

        parts.append("\n### Software Environment")
        parts.append("- STATIS version: 2g")
        parts.append("- Python: 3.14")
        parts.append("- Engine: scipy + statsmodels + lifelines")

        return "\n".join(parts)

    def _save_report(
        self,
        report_content: dict[str, Any],
        spec: ReportSpecification,
        output_path: Path,
    ) -> None:
        """Save report to file."""
        if spec.output_format == OutputFormat.JSON:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_content, f, indent=2)

        elif spec.output_format == OutputFormat.MARKDOWN:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# {report_content['title']}\n\n")
                f.write(f"**Classification:** {report_content['classification']}\n")
                f.write(f"**Prespecified:** {report_content['prespecified_status']}\n\n")

                if report_content.get("authors"):
                    f.write("## Authors\n\n")
                    for author in report_content["authors"]:
                        f.write(f"- {author.get('name', '')} ({author.get('affiliation', '')})\n")
                    f.write("\n")

                for section in report_content["sections"]:
                    f.write(f"## {section['title']}\n\n")
                    f.write(section["content"] + "\n\n")

        elif spec.output_format == OutputFormat.HTML:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"<!DOCTYPE html>\n<html>\n<head>\n<title>{report_content['title']}</title>\n")
                f.write("<style>body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }</style>\n")
                f.write("</head>\n<body>\n")
                f.write(f"<h1>{report_content['title']}</h1>\n")
                f.write(f"<p><strong>Classification:</strong> {report_content['classification']}</p>\n")
                f.write(f"<p><strong>Prespecified:</strong> {report_content['prespecified_status']}</p>\n")

                for section in report_content["sections"]:
                    f.write(f"<h2>{section['title']}</h2>\n")
                    # Convert markdown-ish content to simple HTML
                    content = section["content"].replace("\n", "<br>\n")
                    f.write(f"<p>{content}</p>\n")

                f.write("</body>\n</html>\n")

        else:
            # Default: JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_content, f, indent=2)
