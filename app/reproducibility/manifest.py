"""Reproducibility Package — manifest generation and verification.

Phase 2G: Creates reproducibility packages containing metadata, provenance,
specifications, results, and outputs — but NOT raw patient-level data by default.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReproducibilityManifest(BaseModel):
    """Machine-readable manifest for a reproducibility package."""
    statis_version: str = "2g"
    python_version: str
    platform: str
    generated_at: str = ""

    # Dataset identity
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    dataset_version: int | None = None

    # Analyses
    analysis_version_ids: list[str] = Field(default_factory=list)
    analysis_run_ids: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)

    # Outputs
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    report_ids: list[str] = Field(default_factory=list)

    # Artifact hashes
    artifact_hashes: dict[str, str] = Field(default_factory=dict)

    # Environment
    dependencies: dict[str, str] = Field(default_factory=dict)

    # Seed
    random_seed: int | None = None

    def compute_manifest_hash(self) -> str:
        """Compute deterministic hash of the manifest."""
        canonical = self.model_dump(mode="json")
        canonical.pop("generated_at", None)  # Exclude timestamp from hash
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode()).hexdigest()


class ReproducibilityPackage:
    """Generator for reproducibility packages."""

    def __init__(self, output_base_dir: Path):
        self.output_base_dir = output_base_dir

    def create_package(
        self,
        package_id: str,
        *,
        project_id: str | None = None,
        dataset_id: str | None = None,
        dataset_content_hash: str | None = None,
        dataset_schema_hash: str | None = None,
        dataset_version: int | None = None,
        analysis_specs: list[dict[str, Any]] | None = None,
        statistical_results: list[dict[str, Any]] | None = None,
        output_provenances: list[dict[str, Any]] | None = None,
        audit_entries: list[dict[str, Any]] | None = None,
        random_seed: int | None = None,
        include_raw_data: bool = False,
    ) -> Path:
        """Create a reproducibility package directory.

        Returns the path to the package directory.
        """
        package_dir = self.output_base_dir / f"STATIS_REPRODUCIBILITY_{package_id}"
        package_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (package_dir / "dataset").mkdir(exist_ok=True)
        (package_dir / "analyses").mkdir(exist_ok=True)
        (package_dir / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
        (package_dir / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
        (package_dir / "outputs" / "report").mkdir(parents=True, exist_ok=True)
        (package_dir / "provenance").mkdir(exist_ok=True)
        (package_dir / "environment").mkdir(exist_ok=True)

        # Build manifest
        manifest = self._build_manifest(
            package_id=package_id,
            dataset_id=dataset_id,
            dataset_content_hash=dataset_content_hash,
            dataset_schema_hash=dataset_schema_hash,
            dataset_version=dataset_version,
            analysis_specs=analysis_specs,
            statistical_results=statistical_results,
            output_provenances=output_provenances,
            random_seed=random_seed,
        )

        # Write manifest
        with open(package_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2, default=str)

        # Write dataset metadata
        dataset_metadata = {
            "dataset_id": dataset_id,
            "content_hash": dataset_content_hash,
            "schema_hash": dataset_schema_hash,
            "version": dataset_version,
            "note": "Raw patient-level data is NOT included in default reproducibility packages."
        }
        with open(package_dir / "dataset" / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(dataset_metadata, f, indent=2)

        # Write analysis specifications
        if analysis_specs:
            with open(package_dir / "analyses" / "analysis_specifications.json", "w", encoding="utf-8") as f:
                json.dump(analysis_specs, f, indent=2, default=str)

        # Write statistical results
        if statistical_results:
            with open(package_dir / "analyses" / "statistical_results.json", "w", encoding="utf-8") as f:
                json.dump(statistical_results, f, indent=2, default=str)

        # Write provenance
        lineage = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "dataset_content_hash": dataset_content_hash,
            "dataset_schema_hash": dataset_schema_hash,
            "analysis_run_ids": [r.get("run_id") for r in (statistical_results or [])],
            "output_provenances": output_provenances or [],
        }
        with open(package_dir / "provenance" / "lineage.json", "w", encoding="utf-8") as f:
            json.dump(lineage, f, indent=2, default=str)

        if audit_entries:
            with open(package_dir / "provenance" / "audit_ledger.json", "w", encoding="utf-8") as f:
                json.dump(audit_entries, f, indent=2, default=str)

        # Write environment
        environment = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "statis_version": "2g",
            "dependencies": self._get_dependencies(),
        }
        with open(package_dir / "environment" / "environment.json", "w", encoding="utf-8") as f:
            json.dump(environment, f, indent=2)

        return package_dir

    def _build_manifest(
        self,
        *,
        package_id: str,
        dataset_id: str | None,
        dataset_content_hash: str | None,
        dataset_schema_hash: str | None,
        dataset_version: int | None,
        analysis_specs: list[dict[str, Any]] | None,
        statistical_results: list[dict[str, Any]] | None,
        output_provenances: list[dict[str, Any]] | None,
        random_seed: int | None,
    ) -> ReproducibilityManifest:
        """Build the reproducibility manifest."""
        methods = []
        analysis_version_ids = []
        analysis_run_ids = []

        for result in (statistical_results or []):
            if result.get("method_id"):
                methods.append(result["method_id"])
            if result.get("analysis_version_id"):
                analysis_version_ids.append(result["analysis_version_id"])
            if result.get("run_id"):
                analysis_run_ids.append(result["run_id"])

        figure_ids = []
        table_ids = []
        report_ids = []
        artifact_hashes = {}

        for prov in (output_provenances or []):
            output_type = prov.get("output_type", "")
            output_id = prov.get("output_id", "")
            if output_type == "figure":
                figure_ids.append(output_id)
            elif output_type == "table":
                table_ids.append(output_id)
            elif output_type == "report":
                report_ids.append(output_id)

            if prov.get("artifact_hash"):
                artifact_hashes[output_id] = prov["artifact_hash"]

        return ReproducibilityManifest(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            generated_at=datetime.now(timezone.utc).isoformat(),
            dataset_id=dataset_id,
            dataset_content_hash=dataset_content_hash,
            dataset_schema_hash=dataset_schema_hash,
            dataset_version=dataset_version,
            analysis_version_ids=list(set(analysis_version_ids)),
            analysis_run_ids=list(set(analysis_run_ids)),
            methods=list(set(methods)),
            figure_ids=figure_ids,
            table_ids=table_ids,
            report_ids=report_ids,
            artifact_hashes=artifact_hashes,
            dependencies=self._get_dependencies(),
            random_seed=random_seed,
        )

    def _get_dependencies(self) -> dict[str, str]:
        """Get dependency versions."""
        deps = {}
        for package in ["scipy", "statsmodels", "numpy", "pandas", "matplotlib", "lifelines"]:
            try:
                module = __import__(package)
                deps[package] = getattr(module, "__version__", "unknown")
            except ImportError:
                deps[package] = "not installed"
        return deps

    def verify_package(
        self,
        package_dir: Path,
        *,
        expected_dataset_hash: str | None = None,
        expected_result_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Verify a reproducibility package.

        Returns verification results including pass/fail status and details.
        """
        manifest_path = package_dir / "manifest.json"
        if not manifest_path.exists():
            return {"verified": False, "error": "manifest.json not found"}

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        results = {
            "verified": True,
            "manifest_hash": None,
            "checks": {},
            "errors": [],
            "warnings": [],
        }

        # Check dataset hash
        if expected_dataset_hash:
            actual_hash = manifest_data.get("dataset_content_hash")
            if actual_hash != expected_dataset_hash:
                results["checks"]["dataset_hash"] = {
                    "passed": False,
                    "expected": expected_dataset_hash,
                    "actual": actual_hash,
                }
                results["errors"].append("Dataset hash mismatch")
                results["verified"] = False
            else:
                results["checks"]["dataset_hash"] = {"passed": True}

        # Check result fingerprint
        if expected_result_fingerprint:
            # Check if any artifact hashes match
            artifact_hashes = manifest_data.get("artifact_hashes", {})
            if expected_result_fingerprint not in artifact_hashes.values():
                results["checks"]["result_fingerprint"] = {
                    "passed": False,
                    "expected": expected_result_fingerprint,
                }
                results["warnings"].append("Result fingerprint not found in artifact hashes")
            else:
                results["checks"]["result_fingerprint"] = {"passed": True}

        # Check required files exist
        required_files = [
            "manifest.json",
            "dataset/dataset_metadata.json",
            "environment/environment.json",
        ]
        for req_file in required_files:
            if not (package_dir / req_file).exists():
                results["checks"][req_file] = {"passed": False}
                results["errors"].append(f"Required file missing: {req_file}")
                results["verified"] = False
            else:
                results["checks"][req_file] = {"passed": True}

        # Compute manifest hash
        manifest = ReproducibilityManifest(**{
            k: v for k, v in manifest_data.items()
            if k in ReproducibilityManifest.model_fields
        })
        results["manifest_hash"] = manifest.compute_manifest_hash()

        return results
