"""Output Provenance — traceability for publication outputs.

Phase 2G: Every generated output must be traceable to:
Project → Dataset Identity → Dataset Version → Analysis Specification
→ Analysis Version → Statistical Result → Output Specification → Artifact

This module computes deterministic hashes and builds lineage records.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def compute_artifact_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a generated artifact file."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def compute_content_hash(data: bytes | str) -> str:
    """Compute SHA-256 hash of content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_specification_hash(spec: dict[str, Any]) -> str:
    """Compute deterministic hash of an output specification."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OutputProvenance(BaseModel):
    """Provenance record for a generated output."""
    output_id: str
    output_type: str  # figure, table, report
    output_spec_hash: str

    # Source bindings
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_content_hash: str | None = None
    dataset_schema_hash: str | None = None
    analysis_version_id: str | None = None
    analysis_run_id: str | None = None
    result_fingerprint: str | None = None

    # Artifact
    artifact_path: str | None = None
    artifact_hash: str | None = None
    artifact_format: str | None = None
    artifact_size_bytes: int | None = None

    # Environment
    statis_version: str = "2g"
    python_version: str
    platform: str

    # Timestamps (metadata only — not part of artifact content)
    generated_at: str

    # Classification
    analytical_classification: str = "exploratory"
    prespecified_status: str = "unclear"
    lifecycle_status: str = "generated"


def build_provenance(
    output_id: str,
    output_type: str,
    output_spec: dict[str, Any],
    *,
    artifact_path: Path | None = None,
    project_id: str | None = None,
    dataset_id: str | None = None,
    dataset_content_hash: str | None = None,
    dataset_schema_hash: str | None = None,
    analysis_version_id: str | None = None,
    analysis_run_id: str | None = None,
    result_fingerprint: str | None = None,
    analytical_classification: str = "exploratory",
    prespecified_status: str = "unclear",
    lifecycle_status: str = "generated",
) -> OutputProvenance:
    """Build a complete provenance record for a generated output."""
    spec_hash = compute_specification_hash(output_spec)

    artifact_hash = None
    artifact_size = None
    artifact_format = None
    artifact_path_str = None

    if artifact_path and artifact_path.exists():
        artifact_hash = compute_artifact_hash(artifact_path)
        artifact_size = artifact_path.stat().st_size
        artifact_format = artifact_path.suffix.lstrip(".").lower()
        artifact_path_str = str(artifact_path)

    return OutputProvenance(
        output_id=output_id,
        output_type=output_type,
        output_spec_hash=spec_hash,
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_content_hash=dataset_content_hash,
        dataset_schema_hash=dataset_schema_hash,
        analysis_version_id=analysis_version_id,
        analysis_run_id=analysis_run_id,
        result_fingerprint=result_fingerprint,
        artifact_path=artifact_path_str,
        artifact_hash=artifact_hash,
        artifact_format=artifact_format,
        artifact_size_bytes=artifact_size,
        statis_version="2g",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        analytical_classification=analytical_classification,
        prespecified_status=prespecified_status,
        lifecycle_status=lifecycle_status,
    )


def verify_artifact_integrity(
    artifact_path: Path,
    expected_hash: str,
) -> bool:
    """Verify that an artifact matches its expected hash."""
    if not artifact_path.exists():
        return False
    actual_hash = compute_artifact_hash(artifact_path)
    return actual_hash == expected_hash


def build_lineage(
    output_id: str,
    output_type: str,
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    dataset_content_hash: str | None = None,
    dataset_schema_hash: str | None = None,
    dataset_version: int | None = None,
    analysis_version_id: str | None = None,
    analysis_run_id: str | None = None,
    specification_id: str | None = None,
    result_fingerprint: str | None = None,
    artifact_path: str | None = None,
    artifact_hash: str | None = None,
) -> dict[str, Any]:
    """Build a complete lineage chain for an output."""
    return {
        "output_id": output_id,
        "output_type": output_type,
        "lineage": {
            "project": {"id": project_id},
            "dataset": {
                "id": dataset_id,
                "content_hash": dataset_content_hash,
                "schema_hash": dataset_schema_hash,
                "version": dataset_version,
            },
            "analysis": {
                "version_id": analysis_version_id,
                "run_id": analysis_run_id,
                "specification_id": specification_id,
                "result_fingerprint": result_fingerprint,
            },
            "output": {
                "specification_id": specification_id,
                "artifact_path": artifact_path,
                "artifact_hash": artifact_hash,
            },
        },
        "traceable": all([
            dataset_content_hash is not None,
            result_fingerprint is not None,
            artifact_hash is not None,
        ]),
    }
