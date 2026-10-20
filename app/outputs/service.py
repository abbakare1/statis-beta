"""Output Service — orchestrates publication output lifecycle via the API.

Phase 2H: Coordinates specification → validation → rendering → artifact
persistence → provenance → audit → lifecycle transitions. Routes call into
this service; no business logic lives in API routes.

Never recomputes statistics. Never lets AI generate numerical results.
Enforces dataset identity, analysis version binding, and immutable
approved/archived outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from app.core.database import get_session
from app.memory.audit import log_audit
from app.models import (
    Dataset,
    AnalysisRunRecord,
    FigureRecord,
    TableRecord,
    ReportRecord,
    OutputArtifactRecord,
    ReproducibilityPackageRecord,
)
from app.outputs.lifecycle import (
    OutputLifecycleError,
    validate_transition,
    is_immutable,
    requires_revalidation_before_render,
    get_allowed_transitions,
)
from app.outputs.provenance import build_provenance, build_lineage
from app.outputs.specification import (
    OutputLifecycle,
    FigureSpecification,
    TableSpecification,
    ReportSpecification,
    validate_figure_specification,
    validate_table_specification,
    validate_report_specification,
    OutputValidationResult,
)


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

class OutputServiceError(Exception):
    """Structured service error with an API-mapped error code."""

    def __init__(self, code: str, message: str, *, detail: Any = None):
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(f"{code}: {message}")


# Error code constants
OUTPUT_NOT_FOUND = "OUTPUT_NOT_FOUND"
INVALID_OUTPUT_STATE = "INVALID_OUTPUT_STATE"
VALIDATION_FAILED = "VALIDATION_FAILED"
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
DATASET_IDENTITY_MISMATCH = "DATASET_IDENTITY_MISMATCH"
DATASET_UNREADABLE = "DATASET_UNREADABLE"
ANALYSIS_VERSION_NOT_FOUND = "ANALYSIS_VERSION_NOT_FOUND"
ANALYSIS_VERSION_MISMATCH = "ANALYSIS_VERSION_MISMATCH"
ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
ARTIFACT_INTEGRITY_FAILURE = "ARTIFACT_INTEGRITY_FAILURE"
UNSUPPORTED_OUTPUT_TYPE = "UNSUPPORTED_OUTPUT_TYPE"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
PROVENANCE_FAILURE = "PROVENANCE_FAILURE"
RENDERING_FAILURE = "RENDERING_FAILURE"
APPROVAL_NOT_ALLOWED = "APPROVAL_NOT_ALLOWED"
ARCHIVE_NOT_ALLOWED = "ARCHIVE_NOT_ALLOWED"
CROSS_PROJECT_BINDING = "CROSS_PROJECT_BINDING"
INVALID_TRANSITION = "INVALID_TRANSITION"
PATH_TRAVERSAL_BLOCKED = "PATH_TRAVERSAL_BLOCKED"


# MIME types for artifact retrieval
_MIME_TYPES = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "csv": "text/csv",
    "json": "application/json",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "html": "text/html",
}


# ---------------------------------------------------------------------------
# Service result
# ---------------------------------------------------------------------------

@dataclass
class OutputServiceResult:
    """Result of a service operation."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_code: str | None = None
    http_status: int = 200

    def to_response(self) -> dict[str, Any]:
        """Convert to API response dict."""
        out: dict[str, Any] = {"success": self.success, **self.data}
        if self.errors:
            out["errors"] = self.errors
        if self.warnings:
            out["warnings"] = self.warnings
        if self.error_code:
            out["error_code"] = self.error_code
        return out


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

OUTPUT_MODELS: dict[str, type] = {
    "figure": FigureRecord,
    "table": TableRecord,
    "report": ReportRecord,
}

_OUTPUT_SUBDIR = {
    "figure": "figures",
    "table": "tables",
    "report": "reports",
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OutputService:
    """Orchestrates publication output operations for one output type."""

    def __init__(
        self,
        output_type: str,
        session: AsyncSession,
        *,
        audit_func=None,
        artifact_base_dir: Path | None = None,
    ):
        if output_type not in OUTPUT_MODELS:
            raise OutputServiceError(
                UNSUPPORTED_OUTPUT_TYPE,
                f"Unsupported output type: {output_type!r}. "
                f"Must be one of {sorted(OUTPUT_MODELS)}",
            )
        self.output_type = output_type
        self.session = session
        self.audit_func = audit_func or log_audit
        settings = get_settings()
        self.artifact_base_dir = artifact_base_dir or settings.OUTPUT_DIR
        self.model = OUTPUT_MODELS[output_type]

    # ------------------------------------------------------------------
    # Artifact path security
    # ------------------------------------------------------------------

    def _resolve_artifact_path(self, artifact_path: str) -> Path:
        """Resolve an artifact path, enforcing it stays within the base dir."""
        resolved = Path(artifact_path).resolve()
        base = self.artifact_base_dir.resolve()
        if not str(resolved).startswith(str(base)):
            raise OutputServiceError(
                PATH_TRAVERSAL_BLOCKED,
                f"Artifact path escapes allowed directory: {artifact_path!r}",
            )
        return resolved

    def _artifact_dir(self) -> Path:
        """Return the artifact subdirectory for this output type."""
        d = self.artifact_base_dir / _OUTPUT_SUBDIR[self.output_type]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _get_record(self, output_id: str):
        """Load a record by ID or raise OUTPUT_NOT_FOUND."""
        record = await self.session.get(self.model, output_id)
        if record is None:
            raise OutputServiceError(
                OUTPUT_NOT_FOUND,
                f"{self.output_type.capitalize()} {output_id!r} not found",
            )
        return record

    async def _load_dataset(self, dataset_id: str) -> Dataset:
        """Load a dataset or raise DATASET_NOT_FOUND."""
        dataset = await self.session.get(Dataset, dataset_id)
        if dataset is None:
            raise OutputServiceError(
                DATASET_NOT_FOUND,
                f"Dataset {dataset_id!r} not found",
            )
        return dataset

    async def _load_analysis_run(self, run_id: str) -> AnalysisRunRecord:
        """Load an analysis run or raise ANALYSIS_VERSION_NOT_FOUND."""
        run = await self.session.get(AnalysisRunRecord, run_id)
        if run is None:
            raise OutputServiceError(
                ANALYSIS_VERSION_NOT_FOUND,
                f"Analysis run {run_id!r} not found",
            )
        return run

    async def _check_project_binding(
        self, project_id: str | None, dataset: Dataset | None
    ) -> None:
        """Reject cross-project dataset binding."""
        if project_id and dataset and dataset.project_id != project_id:
            raise OutputServiceError(
                CROSS_PROJECT_BINDING,
                f"Dataset {dataset.id!r} belongs to project "
                f"{dataset.project_id!r}, not {project_id!r}",
            )

    # ------------------------------------------------------------------
    # Fingerprint / idempotency
    # ------------------------------------------------------------------

    def _compute_fingerprint(
        self,
        *,
        dataset_content_hash: str | None,
        dataset_schema_hash: str | None,
        analysis_version_id: str | None,
        specification_hash: str,
    ) -> str:
        """Compute deterministic fingerprint for equivalent-output detection."""
        parts = [
            dataset_content_hash or "",
            dataset_schema_hash or "",
            analysis_version_id or "",
            specification_hash,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    async def _find_equivalent(
        self,
        fingerprint: str,
        specification_hash: str,
        *,
        dataset_content_hash: str | None,
        analysis_version_id: str | None,
    ):
        """Find an existing output with equivalent fingerprint/hash.

        Checks provenance_json for the fingerprint, falling back to
        specification_hash + dataset_content_hash matching.
        """
        stmt = sa.select(self.model).where(
            self.model.provenance_json.isnot(None)
        )
        result = await self.session.execute(stmt)
        for record in result.scalars().all():
            try:
                prov = json.loads(record.provenance_json or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if prov.get("output_fingerprint") == fingerprint:
                return record
            # Fallback: same spec hash + same dataset content hash
            if (
                record.specification_hash == specification_hash
                and dataset_content_hash
                and record.dataset_content_hash == dataset_content_hash
                and analysis_version_id
                and record.analysis_version_id == analysis_version_id
            ):
                return record
        return None

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(self, spec_data: dict[str, Any]) -> OutputServiceResult:
        """Create a new output record from a specification dict."""
        try:
            spec_obj, spec_hash = self._build_spec(spec_data)

            project_id = spec_data.get("project_id")
            dataset_id = spec_data.get("dataset_id")
            dataset = None
            if dataset_id:
                dataset = await self._load_dataset(dataset_id)
                await self._check_project_binding(project_id, dataset)

            record = self._build_record(spec_obj, spec_hash, spec_data, dataset)
            self.session.add(record)
            await self.session.commit()
            await self.session.refresh(record)

            await self._audit(
                action=f"{self.output_type}_created",
                decision=f"Created {self.output_type} {record.id}",
                project_id=project_id,
                output_id=record.id,
            )

            return OutputServiceResult(
                success=True,
                data={"output": self._serialize(record)},
            )
        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    def _build_spec(self, spec_data: dict[str, Any]):
        """Build a specification object from a dict, return (obj, hash)."""
        from pydantic import ValidationError
        try:
            if self.output_type == "figure":
                spec = FigureSpecification(**spec_data)
                return spec, spec.compute_spec_hash()
            if self.output_type == "table":
                spec = TableSpecification(**spec_data)
                return spec, spec.compute_spec_hash()
            if self.output_type == "report":
                spec = ReportSpecification(**spec_data)
                return spec, spec.compute_spec_hash()
        except ValidationError as e:
            raise OutputServiceError(
                "VALIDATION_FAILED",
                f"Specification validation failed: {e}",
                detail={"errors": e.errors()},
            )
        raise OutputServiceError(
            UNSUPPORTED_OUTPUT_TYPE,
            f"Cannot build specification for type {self.output_type!r}",
        )

    def _build_record(
        self,
        spec_obj: Any,
        spec_hash: str,
        spec_data: dict[str, Any],
        dataset: Dataset | None,
    ):
        """Create a database record from a specification."""
        common = {
            "project_id": spec_data.get("project_id"),
            "dataset_id": spec_data.get("dataset_id"),
            "dataset_content_hash": spec_data.get("dataset_content_hash")
            or (dataset.content_hash if dataset else None),
            "dataset_schema_hash": spec_data.get("dataset_schema_hash"),
            "specification_json": spec_obj.model_dump_json(),
            "specification_hash": spec_hash,
            "lifecycle": OutputLifecycle.DRAFT.value,
            "analytical_classification": spec_data.get(
                "analytical_classification", "exploratory"
            ),
            "prespecified_status": spec_data.get("prespecified_status", "unclear"),
        }

        if self.output_type == "figure":
            return FigureRecord(
                **common,
                figure_type=spec_data.get("figure_type", ""),
                analysis_version_id=spec_data.get("analysis_version_id"),
                analysis_run_id=spec_data.get("analysis_run_id"),
                result_fingerprint=spec_data.get("result_fingerprint"),
                artifact_format=spec_data.get("output_format", "png"),
            )
        if self.output_type == "table":
            return TableRecord(
                **common,
                table_type=spec_data.get("table_type", ""),
                analysis_version_ids_json=json.dumps(
                    spec_data.get("analysis_version_ids", [])
                ),
                analysis_run_ids_json=json.dumps(
                    spec_data.get("analysis_run_ids", [])
                ),
                result_fingerprints_json=json.dumps(
                    spec_data.get("result_fingerprints", [])
                ),
                artifact_format=spec_data.get("output_format", "csv"),
            )
        if self.output_type == "report":
            return ReportRecord(
                **common,
                title=spec_data.get("title", ""),
                analysis_version_ids_json=json.dumps(
                    spec_data.get("analysis_version_ids", [])
                ),
                analysis_run_ids_json=json.dumps(
                    spec_data.get("analysis_run_ids", [])
                ),
                artifact_format=spec_data.get("output_format", "markdown"),
            )
        raise OutputServiceError(
            UNSUPPORTED_OUTPUT_TYPE,
            f"Cannot build record for type {self.output_type!r}",
        )

    # ------------------------------------------------------------------
    # Retrieve / list
    # ------------------------------------------------------------------

    async def get(self, output_id: str) -> OutputServiceResult:
        """Retrieve a single output by ID."""
        try:
            record = await self._get_record(output_id)
            return OutputServiceResult(
                success=True,
                data={"output": self._serialize(record)},
            )
        except OutputServiceError as e:
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def list(
        self,
        *,
        project_id: str | None = None,
        dataset_id: str | None = None,
        analysis_version_id: str | None = None,
        status: str | None = None,
        analytical_classification: str | None = None,
        prespecified_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> OutputServiceResult:
        """List outputs with optional filters."""
        try:
            stmt = sa.select(self.model).order_by(self.model.created_at.desc())

            if project_id:
                stmt = stmt.where(self.model.project_id == project_id)
            if dataset_id:
                stmt = stmt.where(self.model.dataset_id == dataset_id)
            if status:
                stmt = stmt.where(self.model.lifecycle == status)
            if analytical_classification:
                stmt = stmt.where(
                    self.model.analytical_classification
                    == analytical_classification
                )
            if prespecified_status:
                stmt = stmt.where(
                    self.model.prespecified_status == prespecified_status
                )
            if analysis_version_id:
                if self.output_type == "figure":
                    stmt = stmt.where(
                        self.model.analysis_version_id == analysis_version_id
                    )
                else:
                    # JSON list fields — use LIKE for simplicity
                    stmt = stmt.where(
                        self.model.analysis_version_ids_json.like(
                            f'%"{analysis_version_id}"%'
                        )
                    )

            stmt = stmt.limit(min(limit, 500)).offset(max(offset, 0))
            result = await self.session.execute(stmt)
            records = result.scalars().all()

            return OutputServiceResult(
                success=True,
                data={
                    "outputs": [self._serialize(r) for r in records],
                    "count": len(records),
                },
            )
        except OutputServiceError as e:
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    async def validate(self, output_id: str) -> OutputServiceResult:
        """Validate an output specification.

        Uses the same validation functions the renderer uses internally.
        Transitions lifecycle to VALIDATING → VALIDATED / VALIDATION_FAILED.
        """
        try:
            record = await self._get_record(output_id)

            if is_immutable(record.lifecycle):
                raise OutputServiceError(
                    INVALID_OUTPUT_STATE,
                    f"Cannot validate {self.output_type} in immutable state "
                    f"{record.lifecycle!r}",
                )

            # Begin validation transition
            validate_transition(record.lifecycle, OutputLifecycle.VALIDATING)
            record.lifecycle = OutputLifecycle.VALIDATING.value
            await self.session.commit()

            # Build spec object
            spec_obj = self._record_to_spec(record)

            # Gather validation context
            available_columns: list[str] | None = None
            dataset = None
            if record.dataset_id:
                try:
                    dataset = await self._load_dataset(record.dataset_id)
                    available_columns = await self._load_columns(dataset)
                except OutputServiceError:
                    pass

            has_result = False
            result_method_id: str | None = None
            has_survival = False
            has_analyses = False
            analysis_run_ids = self._record_analysis_run_ids(record)
            if analysis_run_ids:
                has_analyses = True
                for run_id in analysis_run_ids:
                    try:
                        run = await self._load_analysis_run(run_id)
                        if run.result_json:
                            has_result = True
                            try:
                                result_data = json.loads(run.result_json)
                                result_method_id = result_data.get("method_id")
                                if result_data.get("survival_metadata"):
                                    has_survival = True
                            except json.JSONDecodeError:
                                pass
                    except OutputServiceError:
                        continue

            # Call Phase 2G validators
            if self.output_type == "figure":
                validation = validate_figure_specification(
                    spec_obj,
                    available_columns=available_columns,
                    has_result=has_result,
                    result_method_id=result_method_id,
                    has_survival_metadata=has_survival,
                )
            elif self.output_type == "table":
                validation = validate_table_specification(
                    spec_obj,
                    available_columns=available_columns,
                    has_result=has_result,
                    result_method_id=result_method_id,
                )
            elif self.output_type == "report":
                validation = validate_report_specification(
                    spec_obj,
                    has_analyses=has_analyses,
                )
            else:
                validation = OutputValidationResult(valid=False, errors=["Unknown output type"])

            # Apply validation result
            record.validation_errors_json = (
                json.dumps(validation.errors) if validation.errors else None
            )

            if validation.valid:
                record.lifecycle = OutputLifecycle.VALIDATED.value
                await self.session.commit()
                await self._audit(
                    action=f"{self.output_type}_validated",
                    decision=f"Validation passed for {self.output_type} {record.id}",
                    project_id=record.project_id,
                    output_id=record.id,
                )
                return OutputServiceResult(
                    success=True,
                    data={
                        "output_id": record.id,
                        "valid": True,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                        "lifecycle": record.lifecycle,
                    },
                )
            else:
                record.lifecycle = OutputLifecycle.VALIDATION_FAILED.value
                await self.session.commit()
                await self._audit(
                    action=f"{self.output_type}_validation_failed",
                    decision=(
                        f"Validation failed for {self.output_type} {record.id}: "
                        f"{len(validation.blocking_errors)} blocking errors"
                    ),
                    project_id=record.project_id,
                    output_id=record.id,
                    warnings=validation.errors,
                )
                return OutputServiceResult(
                    success=False,
                    data={
                        "output_id": record.id,
                        "valid": False,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                        "lifecycle": record.lifecycle,
                    },
                    errors=validation.errors,
                    error_code=VALIDATION_FAILED,
                    http_status=422,
                )

        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )
        except OutputLifecycleError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[str(e)],
                error_code=INVALID_TRANSITION,
                http_status=409,
            )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    async def render(self, output_id: str) -> OutputServiceResult:
        """Render an output to an artifact.

        Always revalidates before rendering. Verifies dataset identity
        and analysis version binding. Idempotent: equivalent outputs
        return the existing artifact.
        """
        try:
            record = await self._get_record(output_id)

            if is_immutable(record.lifecycle):
                raise OutputServiceError(
                    INVALID_OUTPUT_STATE,
                    f"Cannot render {self.output_type} in immutable state "
                    f"{record.lifecycle!r}",
                )

            # Idempotency check: equivalent output already generated?
            spec_obj = self._record_to_spec(record)
            spec_hash = record.specification_hash or spec_obj.compute_spec_hash()
            record_analysis_version_id = getattr(record, "analysis_version_id", None)
            if record_analysis_version_id is None:
                # Tables and reports store analysis version IDs as a JSON list
                av_ids = _safe_json_loads(
                    getattr(record, "analysis_version_ids_json", None)
                ) or []
                record_analysis_version_id = av_ids[0] if av_ids else None
            fingerprint = self._compute_fingerprint(
                dataset_content_hash=record.dataset_content_hash,
                dataset_schema_hash=record.dataset_schema_hash,
                analysis_version_id=record_analysis_version_id,
                specification_hash=spec_hash,
            )

            if record.artifact_hash and record.lifecycle in (
                OutputLifecycle.GENERATED.value,
                OutputLifecycle.REVIEW_REQUIRED.value,
                OutputLifecycle.APPROVED.value,
            ):
                # Already has an artifact in a post-render state
                return OutputServiceResult(
                    success=True,
                    data={
                        "output": self._serialize(record),
                        "idempotent": True,
                        "message": "Output already has a generated artifact",
                    },
                )

            # Check for equivalent output from another record
            if record.lifecycle not in (
                OutputLifecycle.APPROVED.value,
                OutputLifecycle.ARCHIVED.value,
            ):
                equivalent = await self._find_equivalent(
                    fingerprint, spec_hash,
                    dataset_content_hash=record.dataset_content_hash,
                    analysis_version_id=record_analysis_version_id,
                )
                if equivalent and equivalent.id != record.id and equivalent.artifact_hash:
                    # Adopt existing artifact (idempotent render)
                    record.artifact_path = equivalent.artifact_path
                    record.artifact_hash = equivalent.artifact_hash
                    record.artifact_format = equivalent.artifact_format
                    record.provenance_json = equivalent.provenance_json
                    record.lifecycle = OutputLifecycle.GENERATED.value
                    await self.session.commit()
                    await self._audit(
                        action=f"{self.output_type}_rendered",
                        decision=(
                            f"Idempotent render: adopted artifact from "
                            f"equivalent output {equivalent.id}"
                        ),
                        project_id=record.project_id,
                        output_id=record.id,
                    )
                    return OutputServiceResult(
                        success=True,
                        data={
                            "output": self._serialize(record),
                            "idempotent": True,
                            "equivalent_output_id": equivalent.id,
                        },
                    )

            # Enforce lifecycle for rendering
            if requires_revalidation_before_render(record.lifecycle):
                raise OutputServiceError(
                    INVALID_OUTPUT_STATE,
                    f"Cannot render from state {record.lifecycle!r}. "
                    f"Must revalidate first.",
                )

            if record.lifecycle == OutputLifecycle.DRAFT.value:
                # Allow render from draft — rendering itself validates.
                # No lifecycle transition check needed for DRAFT → RENDERING.
                record.lifecycle = OutputLifecycle.RENDERING.value
            elif record.lifecycle in (
                OutputLifecycle.VALIDATED.value,
                OutputLifecycle.GENERATED.value,
                OutputLifecycle.REVIEW_REQUIRED.value,
            ):
                validate_transition(record.lifecycle, OutputLifecycle.RENDERING)
                record.lifecycle = OutputLifecycle.RENDERING.value
            else:
                raise OutputServiceError(
                    INVALID_OUTPUT_STATE,
                    f"Cannot render from state {record.lifecycle!r}",
                )
            await self.session.commit()

            # --- Load data ---
            dataset = None
            df = None
            if record.dataset_id:
                dataset = await self._load_dataset(record.dataset_id)

                # Verify dataset identity if hashes are bound
                if record.dataset_content_hash and dataset.stored_path:
                    from app.data.dataset_identity import verify_dataset_identity
                    identity_check = verify_dataset_identity(
                        dataset.stored_path,
                        expected_content_hash=record.dataset_content_hash,
                        expected_schema_hash=record.dataset_schema_hash,
                    )
                    if not identity_check["valid"]:
                        record.lifecycle = OutputLifecycle.VALIDATION_FAILED.value
                        await self.session.commit()
                        raise OutputServiceError(
                            DATASET_IDENTITY_MISMATCH,
                            "Dataset identity verification failed: "
                            + "; ".join(identity_check["mismatches"]),
                            detail=identity_check,
                        )

                # Load DataFrame
                try:
                    import pandas as pd
                    stored = Path(dataset.stored_path)
                    if stored.exists():
                        suffix = stored.suffix.lower()
                        if suffix == ".csv":
                            df = pd.read_csv(stored)
                        elif suffix == ".tsv":
                            df = pd.read_csv(stored, sep="\t")
                        elif suffix in (".xlsx", ".xls"):
                            df = pd.read_excel(stored)
                except Exception as e:
                    raise OutputServiceError(
                        DATASET_UNREADABLE,
                        f"Could not load dataset {record.dataset_id!r}: {e}",
                    )

            # Load analysis results
            results = []
            analysis_run_ids = self._record_analysis_run_ids(record)
            for run_id in analysis_run_ids:
                try:
                    run = await self._load_analysis_run(run_id)
                    if run.result_json:
                        results.append(json.loads(run.result_json))
                except (OutputServiceError, json.JSONDecodeError):
                    continue

            # --- Render ---
            try:
                artifact_path = self._dispatch_render(
                    spec_obj, df=df, results=results
                )
            except OutputServiceError:
                raise
            except Exception as e:
                record.lifecycle = OutputLifecycle.RENDERING_FAILED.value
                await self.session.commit()
                await self._audit(
                    action=f"{self.output_type}_rendering_failed",
                    decision=f"Rendering failed for {record.id}: {e}",
                    project_id=record.project_id,
                    output_id=record.id,
                )
                raise OutputServiceError(
                    RENDERING_FAILURE,
                    f"Rendering failed: {e}",
                )

            # --- Artifact persistence ---
            resolved_path = self._resolve_artifact_path(str(artifact_path))
            if not resolved_path.exists():
                record.lifecycle = OutputLifecycle.RENDERING_FAILED.value
                await self.session.commit()
                raise OutputServiceError(
                    ARTIFACT_NOT_FOUND,
                    f"Renderer reported success but artifact not found at "
                    f"{artifact_path!r}",
                )

            artifact_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            artifact_size = resolved_path.stat().st_size
            artifact_format = resolved_path.suffix.lstrip(".").lower()

            # Build provenance
            spec_dict = json.loads(record.specification_json)
            record_result_fp = getattr(record, "result_fingerprint", None)
            if record_result_fp is None:
                fps = _safe_json_loads(
                    getattr(record, "result_fingerprints_json", None)
                ) or []
                record_result_fp = fps[0] if fps else None
            provenance = build_provenance(
                output_id=record.id,
                output_type=self.output_type,
                output_spec=spec_dict,
                project_id=record.project_id,
                dataset_id=record.dataset_id,
                dataset_content_hash=record.dataset_content_hash,
                dataset_schema_hash=record.dataset_schema_hash,
                analysis_version_id=record_analysis_version_id,
                analysis_run_id=(
                    analysis_run_ids[0] if analysis_run_ids else None
                ),
                result_fingerprint=record_result_fp,
                analytical_classification=record.analytical_classification,
                prespecified_status=record.prespecified_status,
                lifecycle_status=OutputLifecycle.GENERATED.value,
            )
            # Set artifact fields (build_provenance without artifact_path
            # doesn't compute them)
            prov_dict = provenance.model_dump()
            prov_dict["artifact_path"] = str(resolved_path)
            prov_dict["artifact_hash"] = artifact_hash
            prov_dict["artifact_format"] = artifact_format
            prov_dict["artifact_size_bytes"] = artifact_size
            prov_dict["output_fingerprint"] = fingerprint

            # Update record
            record.artifact_path = str(resolved_path)
            record.artifact_hash = artifact_hash
            record.artifact_format = artifact_format
            record.provenance_json = json.dumps(prov_dict, default=str)
            record.lifecycle = OutputLifecycle.GENERATED.value
            await self.session.commit()

            # Persist artifact record
            artifact_record = OutputArtifactRecord(
                output_type=self.output_type,
                output_id=record.id,
                artifact_path=str(resolved_path),
                artifact_hash=artifact_hash,
                artifact_format=artifact_format,
                artifact_size_bytes=artifact_size,
                dataset_content_hash=record.dataset_content_hash,
                result_fingerprint=record_result_fp,
                specification_hash=record.specification_hash,
                verified=False,
            )
            self.session.add(artifact_record)
            await self.session.commit()

            # Audit
            await self._audit(
                action=f"{self.output_type}_rendered",
                decision=(
                    f"Rendered {self.output_type} {record.id} "
                    f"(hash={artifact_hash[:12]}...)"
                ),
                project_id=record.project_id,
                output_id=record.id,
                output_summary=f"artifact_hash={artifact_hash}",
            )

            return OutputServiceResult(
                success=True,
                data={
                    "output": self._serialize(record),
                    "artifact_hash": artifact_hash,
                    "artifact_format": artifact_format,
                    "artifact_size_bytes": artifact_size,
                    "idempotent": False,
                },
            )

        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )
        except OutputLifecycleError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[str(e)],
                error_code=INVALID_TRANSITION,
                http_status=409,
            )

    def _dispatch_render(
        self,
        spec_obj: Any,
        *,
        df=None,
        results: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Dispatch to the appropriate Phase 2G renderer."""
        output_dir = self._artifact_dir()

        if self.output_type == "figure":
            from app.visualization.renderer import FigureRenderer
            renderer = FigureRenderer()
            result = results[0] if results else None
            return renderer.render(spec_obj, df=df, result=result, output_dir=output_dir)

        if self.output_type == "table":
            from app.tables.renderer import TableRenderer
            renderer = TableRenderer()
            return renderer.render(spec_obj, df=df, results=results, output_dir=output_dir)

        if self.output_type == "report":
            from app.reporting.renderer import ReportRenderer
            renderer = ReportRenderer()
            return renderer.render(
                spec_obj,
                results=results,
                dataset_info=None,
                quality_info=None,
                output_dir=output_dir,
            )

        raise OutputServiceError(
            UNSUPPORTED_OUTPUT_TYPE,
            f"No renderer for type {self.output_type!r}",
        )

    # ------------------------------------------------------------------
    # Artifact retrieval
    # ------------------------------------------------------------------

    async def get_artifact(self, output_id: str) -> OutputServiceResult:
        """Retrieve artifact metadata and verify integrity."""
        try:
            record = await self._get_record(output_id)

            if not record.artifact_path:
                raise OutputServiceError(
                    ARTIFACT_NOT_FOUND,
                    f"No artifact generated for {self.output_type} {output_id!r}",
                )

            resolved = self._resolve_artifact_path(record.artifact_path)
            if not resolved.exists():
                raise OutputServiceError(
                    ARTIFACT_NOT_FOUND,
                    f"Artifact file not found at recorded path",
                )

            # Verify integrity
            actual_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
            integrity_ok = actual_hash == record.artifact_hash
            if not integrity_ok:
                raise OutputServiceError(
                    ARTIFACT_INTEGRITY_FAILURE,
                    "Artifact hash mismatch — file may have been tampered with",
                    detail={
                        "expected_hash": record.artifact_hash,
                        "actual_hash": actual_hash,
                    },
                )

            mime_type = _MIME_TYPES.get(
                (record.artifact_format or resolved.suffix.lstrip(".")).lower(),
                "application/octet-stream",
            )

            return OutputServiceResult(
                success=True,
                data={
                    "output_id": record.id,
                    "artifact_path": str(resolved),
                    "artifact_hash": record.artifact_hash,
                    "artifact_format": record.artifact_format,
                    "artifact_size_bytes": resolved.stat().st_size,
                    "mime_type": mime_type,
                    "integrity_verified": True,
                },
            )
        except OutputServiceError as e:
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def read_artifact_bytes(self, output_id: str) -> tuple[bytes, str, str]:
        """Read artifact content. Returns (bytes, mime_type, filename).

        Raises OutputServiceError on any failure.
        """
        meta = await self.get_artifact(output_id)
        if not meta.success:
            raise OutputServiceError(
                meta.error_code or ARTIFACT_NOT_FOUND,
                meta.errors[0] if meta.errors else "Artifact retrieval failed",
            )
        path = Path(meta.data["artifact_path"])
        content = path.read_bytes()
        return (
            content,
            meta.data["mime_type"],
            path.name,
        )

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    async def approve(
        self,
        output_id: str,
        *,
        approved_by: str = "researcher",
        notes: str = "",
    ) -> OutputServiceResult:
        """Approve an output. Explicit approval required."""
        try:
            record = await self._get_record(output_id)

            if record.lifecycle == OutputLifecycle.APPROVED.value:
                return OutputServiceResult(
                    success=True,
                    data={
                        "output": self._serialize(record),
                        "message": "Already approved",
                    },
                )

            if record.lifecycle != OutputLifecycle.REVIEW_REQUIRED.value:
                raise OutputServiceError(
                    APPROVAL_NOT_ALLOWED,
                    f"Cannot approve {self.output_type} in state "
                    f"{record.lifecycle!r}. Must be in 'review_required'. "
                    f"Current allowed transitions: "
                    f"{get_allowed_transitions(record.lifecycle)}",
                )

            from datetime import datetime, timezone
            record.lifecycle = OutputLifecycle.APPROVED.value
            record.approved_by = approved_by
            record.approved_at = datetime.now(timezone.utc)
            await self.session.commit()

            await self._audit(
                action=f"{self.output_type}_approved",
                decision=(
                    f"Approved {self.output_type} {record.id} by {approved_by}"
                ),
                project_id=record.project_id,
                output_id=record.id,
                researcher_decision="approve",
                rationale=notes or None,
            )

            return OutputServiceResult(
                success=True,
                data={"output": self._serialize(record)},
            )
        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def archive(self, output_id: str) -> OutputServiceResult:
        """Archive an output. Archived outputs remain immutable + retrievable."""
        try:
            record = await self._get_record(output_id)

            if record.lifecycle == OutputLifecycle.ARCHIVED.value:
                return OutputServiceResult(
                    success=True,
                    data={
                        "output": self._serialize(record),
                        "message": "Already archived",
                    },
                )

            if record.lifecycle != OutputLifecycle.APPROVED.value:
                raise OutputServiceError(
                    ARCHIVE_NOT_ALLOWED,
                    f"Cannot archive {self.output_type} in state "
                    f"{record.lifecycle!r}. Must be in 'approved'. "
                    f"Current allowed transitions: "
                    f"{get_allowed_transitions(record.lifecycle)}",
                )

            record.lifecycle = OutputLifecycle.ARCHIVED.value
            await self.session.commit()

            await self._audit(
                action=f"{self.output_type}_archived",
                decision=f"Archived {self.output_type} {record.id}",
                project_id=record.project_id,
                output_id=record.id,
            )

            return OutputServiceResult(
                success=True,
                data={"output": self._serialize(record)},
            )
        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def request_review(self, output_id: str) -> OutputServiceResult:
        """Transition from GENERATED to REVIEW_REQUIRED."""
        try:
            record = await self._get_record(output_id)
            validate_transition(
                record.lifecycle, OutputLifecycle.REVIEW_REQUIRED
            )
            record.lifecycle = OutputLifecycle.REVIEW_REQUIRED.value
            await self.session.commit()

            await self._audit(
                action=f"{self.output_type}_review_requested",
                decision=f"Review requested for {self.output_type} {record.id}",
                project_id=record.project_id,
                output_id=record.id,
            )

            return OutputServiceResult(
                success=True,
                data={"output": self._serialize(record)},
            )
        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )
        except OutputLifecycleError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[str(e)],
                error_code=INVALID_TRANSITION,
                http_status=409,
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize(self, record) -> dict[str, Any]:
        """Serialize a database record to an API response dict."""
        out: dict[str, Any] = {
            "id": record.id,
            "output_type": self.output_type,
            "project_id": record.project_id,
            "dataset_id": record.dataset_id,
            "dataset_content_hash": record.dataset_content_hash,
            "dataset_schema_hash": record.dataset_schema_hash,
            "specification_hash": record.specification_hash,
            "artifact_path": record.artifact_path,
            "artifact_hash": record.artifact_hash,
            "artifact_format": record.artifact_format,
            "lifecycle": record.lifecycle,
            "allowed_transitions": get_allowed_transitions(record.lifecycle),
            "is_immutable": is_immutable(record.lifecycle),
            "validation_errors": _safe_json_loads(record.validation_errors_json),
            "approved_by": record.approved_by,
            "approved_at": (
                record.approved_at.isoformat() if record.approved_at else None
            ),
            "analytical_classification": record.analytical_classification,
            "prespecified_status": record.prespecified_status,
            "provenance": _safe_json_loads(record.provenance_json),
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }

        # Type-specific fields
        if self.output_type == "figure":
            out["figure_type"] = record.figure_type
            out["analysis_version_id"] = record.analysis_version_id
            out["analysis_run_id"] = record.analysis_run_id
            out["result_fingerprint"] = getattr(record, "result_fingerprint", None)
            spec = _safe_json_loads(record.specification_json)
            if spec:
                out["title"] = spec.get("title", "")
                out["output_format"] = spec.get("output_format", "")
        elif self.output_type == "table":
            out["table_type"] = record.table_type
            out["analysis_version_ids"] = _safe_json_loads(
                record.analysis_version_ids_json
            )
            out["analysis_run_ids"] = _safe_json_loads(
                record.analysis_run_ids_json
            )
            spec = _safe_json_loads(record.specification_json)
            if spec:
                out["title"] = spec.get("title", "")
                out["output_format"] = spec.get("output_format", "")
        elif self.output_type == "report":
            out["title"] = record.title
            out["analysis_version_ids"] = _safe_json_loads(
                record.analysis_version_ids_json
            )
            out["analysis_run_ids"] = _safe_json_loads(
                record.analysis_run_ids_json
            )
            spec = _safe_json_loads(record.specification_json)
            if spec:
                out["output_format"] = spec.get("output_format", "")

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_to_spec(self, record):
        """Rebuild a specification object from a stored record."""
        spec_data = json.loads(record.specification_json)
        if self.output_type == "figure":
            return FigureSpecification(**spec_data)
        if self.output_type == "table":
            return TableSpecification(**spec_data)
        if self.output_type == "report":
            return ReportSpecification(**spec_data)
        raise OutputServiceError(
            UNSUPPORTED_OUTPUT_TYPE,
            f"Cannot rebuild spec for type {self.output_type!r}",
        )

    def _record_analysis_run_ids(self, record) -> list[str]:
        """Extract analysis run IDs from a record."""
        if self.output_type == "figure":
            return [record.analysis_run_id] if record.analysis_run_id else []
        return _safe_json_loads(
            getattr(record, "analysis_run_ids_json", None)
        ) or []

    async def _load_columns(self, dataset: Dataset) -> list[str] | None:
        """Load column names from a dataset file."""
        try:
            import pandas as pd
            stored = Path(dataset.stored_path)
            if not stored.exists():
                return None
            suffix = stored.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(stored, nrows=0)
            elif suffix == ".tsv":
                df = pd.read_csv(stored, sep="\t", nrows=0)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(stored, nrows=0)
            else:
                return None
            return df.columns.tolist()
        except Exception:
            return None

    async def _audit(
        self,
        *,
        action: str,
        decision: str,
        project_id: str | None = None,
        output_id: str | None = None,
        researcher_decision: str | None = None,
        rationale: str | None = None,
        output_summary: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Write an audit ledger entry."""
        try:
            await self.audit_func(
                action=action,
                agent="output_service",
                decision=decision,
                project_id=project_id,
                researcher_decision=researcher_decision,
                rationale=rationale,
                output_summary=output_summary or (f"output_id={output_id}" if output_id else None),
                warnings=warnings,
            )
        except Exception:
            # Audit failure must not block the operation
            pass


# ---------------------------------------------------------------------------
# Reproducibility Service
# ---------------------------------------------------------------------------

class ReproducibilityService:
    """Orchestrates reproducibility package operations."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        audit_func=None,
        artifact_base_dir: Path | None = None,
    ):
        self.session = session
        self.audit_func = audit_func or log_audit
        settings = get_settings()
        self.artifact_base_dir = artifact_base_dir or settings.OUTPUT_DIR

    async def create(self, data: dict[str, Any]) -> OutputServiceResult:
        """Create a reproducibility package."""
        try:
            package_id = data.get("package_id") or _generate_package_id()
            project_id = data.get("project_id")
            dataset_id = data.get("dataset_id")
            include_raw_data = data.get("include_raw_data", False)

            # Verify dataset if provided
            dataset = None
            if dataset_id:
                dataset = await self.session.get(Dataset, dataset_id)
                if dataset is None:
                    raise OutputServiceError(
                        DATASET_NOT_FOUND,
                        f"Dataset {dataset_id!r} not found",
                    )
                if project_id and dataset.project_id != project_id:
                    raise OutputServiceError(
                        CROSS_PROJECT_BINDING,
                        f"Dataset belongs to project {dataset.project_id!r}, "
                        f"not {project_id!r}",
                    )

            # Check duplicate package_id
            existing = await self.session.execute(
                sa.select(ReproducibilityPackageRecord).where(
                    ReproducibilityPackageRecord.package_id == package_id
                )
            )
            if existing.scalar_one_or_none():
                raise OutputServiceError(
                    INVALID_OUTPUT_STATE,
                    f"Reproducibility package {package_id!r} already exists",
                )

            # Gather analysis runs
            analysis_run_ids = data.get("analysis_run_ids", [])
            statistical_results = []
            for run_id in analysis_run_ids:
                run = await self.session.get(AnalysisRunRecord, run_id)
                if run and run.result_json:
                    statistical_results.append(json.loads(run.result_json))

            # Create package via Phase 2G implementation
            from app.reproducibility.manifest import ReproducibilityPackage
            package_gen = ReproducibilityPackage(self.artifact_base_dir)
            package_path = package_gen.create_package(
                package_id,
                project_id=project_id,
                dataset_id=dataset_id,
                dataset_content_hash=(
                    dataset.content_hash if dataset else data.get("dataset_content_hash")
                ),
                dataset_schema_hash=data.get("dataset_schema_hash"),
                dataset_version=dataset.version if dataset else None,
                statistical_results=statistical_results,
                output_provenances=data.get("output_provenances"),
                random_seed=data.get("random_seed"),
                include_raw_data=include_raw_data,
            )

            # Read manifest
            manifest_path = package_path / "manifest.json"
            manifest_data = {}
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            manifest_hash = hashlib.sha256(
                json.dumps(manifest_data, sort_keys=True, default=str).encode()
            ).hexdigest()

            # Persist record
            record = ReproducibilityPackageRecord(
                package_id=package_id,
                package_path=str(package_path),
                project_id=project_id,
                dataset_id=dataset_id,
                dataset_content_hash=(
                    dataset.content_hash if dataset else data.get("dataset_content_hash")
                ),
                dataset_schema_hash=data.get("dataset_schema_hash"),
                manifest_json=json.dumps(manifest_data, default=str),
                manifest_hash=manifest_hash,
                verified=False,
                analysis_run_ids_json=json.dumps(analysis_run_ids),
                output_ids_json=json.dumps(data.get("output_ids", [])),
                includes_raw_data=include_raw_data,
            )
            self.session.add(record)
            await self.session.commit()
            await self.session.refresh(record)

            await self._audit(
                action="reproducibility_package_created",
                decision=f"Created reproducibility package {package_id}",
                project_id=project_id,
                output_summary=f"package_id={package_id}",
            )

            return OutputServiceResult(
                success=True,
                data={
                    "package": self._serialize(record),
                },
            )
        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def get(self, package_id: str) -> OutputServiceResult:
        """Retrieve a reproducibility package by package_id."""
        try:
            record = await self._get_record(package_id)
            return OutputServiceResult(
                success=True,
                data={"package": self._serialize(record)},
            )
        except OutputServiceError as e:
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def get_manifest(self, package_id: str) -> OutputServiceResult:
        """Retrieve the manifest of a reproducibility package."""
        try:
            record = await self._get_record(package_id)
            manifest = _safe_json_loads(record.manifest_json) or {}
            return OutputServiceResult(
                success=True,
                data={
                    "package_id": record.package_id,
                    "manifest": manifest,
                    "manifest_hash": record.manifest_hash,
                },
            )
        except OutputServiceError as e:
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def verify(
        self,
        package_id: str,
        *,
        expected_dataset_hash: str | None = None,
    ) -> OutputServiceResult:
        """Verify a reproducibility package."""
        try:
            record = await self._get_record(package_id)

            package_path = Path(record.package_path)
            if not package_path.exists():
                raise OutputServiceError(
                    ARTIFACT_NOT_FOUND,
                    f"Package directory not found at {record.package_path!r}",
                )

            from app.reproducibility.manifest import ReproducibilityPackage
            package_gen = ReproducibilityPackage(self.artifact_base_dir)
            verification = package_gen.verify_package(
                package_path,
                expected_dataset_hash=expected_dataset_hash
                or record.dataset_content_hash,
            )

            # Update record
            record.verified = verification.get("verified", False)
            from datetime import datetime, timezone
            record.verified_at = datetime.now(timezone.utc)
            record.verification_results_json = json.dumps(
                verification, default=str
            )
            await self.session.commit()

            await self._audit(
                action="reproducibility_package_verified",
                decision=(
                    f"Verification {'passed' if verification.get('verified') else 'failed'} "
                    f"for package {package_id}"
                ),
                project_id=record.project_id,
                warnings=verification.get("errors"),
            )

            return OutputServiceResult(
                success=verification.get("verified", False),
                data={
                    "package_id": record.package_id,
                    "verified": verification.get("verified", False),
                    "verification": verification,
                },
                errors=verification.get("errors", []),
                error_code=(
                    None if verification.get("verified")
                    else ARTIFACT_INTEGRITY_FAILURE
                ),
                http_status=200 if verification.get("verified") else 409,
            )
        except OutputServiceError as e:
            await self.session.rollback()
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def get_artifact_info(self, package_id: str) -> OutputServiceResult:
        """Get package artifact directory info."""
        try:
            record = await self._get_record(package_id)
            package_path = Path(record.package_path)

            if not package_path.exists():
                raise OutputServiceError(
                    ARTIFACT_NOT_FOUND,
                    f"Package directory not found",
                )

            # List package contents
            contents = []
            for p in sorted(package_path.rglob("*")):
                if p.is_file():
                    contents.append({
                        "path": str(p.relative_to(package_path)),
                        "size_bytes": p.stat().st_size,
                    })

            return OutputServiceResult(
                success=True,
                data={
                    "package_id": record.package_id,
                    "package_path": str(package_path),
                    "manifest_hash": record.manifest_hash,
                    "verified": record.verified,
                    "includes_raw_data": record.includes_raw_data,
                    "contents": contents,
                },
            )
        except OutputServiceError as e:
            return OutputServiceResult(
                success=False,
                errors=[e.message],
                error_code=e.code,
                http_status=_http_status_for_code(e.code),
            )

    async def _get_record(self, package_id: str) -> ReproducibilityPackageRecord:
        record = await self.session.execute(
            sa.select(ReproducibilityPackageRecord).where(
                ReproducibilityPackageRecord.package_id == package_id
            )
        )
        rec = record.scalar_one_or_none()
        if rec is None:
            # Also try by primary key ID
            rec = await self.session.get(
                ReproducibilityPackageRecord, package_id
            )
        if rec is None:
            raise OutputServiceError(
                OUTPUT_NOT_FOUND,
                f"Reproducibility package {package_id!r} not found",
            )
        return rec

    def _serialize(self, record: ReproducibilityPackageRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "package_id": record.package_id,
            "package_path": record.package_path,
            "project_id": record.project_id,
            "dataset_id": record.dataset_id,
            "dataset_content_hash": record.dataset_content_hash,
            "dataset_schema_hash": record.dataset_schema_hash,
            "manifest_hash": record.manifest_hash,
            "verified": record.verified,
            "verified_at": (
                record.verified_at.isoformat() if record.verified_at else None
            ),
            "includes_raw_data": record.includes_raw_data,
            "analysis_run_ids": _safe_json_loads(record.analysis_run_ids_json),
            "output_ids": _safe_json_loads(record.output_ids_json),
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }

    async def _audit(self, **kwargs) -> None:
        try:
            await self.audit_func(
                agent="reproducibility_service",
                **kwargs,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_output_service(
    output_type: str,
    session: AsyncSession,
    *,
    artifact_base_dir: Path | None = None,
) -> OutputService:
    """Factory for output services."""
    return OutputService(
        output_type,
        session,
        artifact_base_dir=artifact_base_dir,
    )


def get_reproducibility_service(
    session: AsyncSession,
    *,
    artifact_base_dir: Path | None = None,
) -> ReproducibilityService:
    """Factory for the reproducibility service."""
    return ReproducibilityService(
        session,
        artifact_base_dir=artifact_base_dir,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(value: str | None) -> Any:
    """Safely parse JSON, returning None on failure."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _generate_package_id() -> str:
    """Generate a unique reproducibility package ID."""
    import uuid
    return f"repro_{uuid.uuid4().hex[:12]}"


def _http_status_for_code(code: str) -> int:
    """Map error codes to HTTP status codes."""
    mapping = {
        OUTPUT_NOT_FOUND: 404,
        INVALID_OUTPUT_STATE: 409,
        VALIDATION_FAILED: 422,
        DATASET_NOT_FOUND: 404,
        DATASET_IDENTITY_MISMATCH: 409,
        DATASET_UNREADABLE: 422,
        ANALYSIS_VERSION_NOT_FOUND: 404,
        ANALYSIS_VERSION_MISMATCH: 409,
        ARTIFACT_NOT_FOUND: 404,
        ARTIFACT_INTEGRITY_FAILURE: 409,
        UNSUPPORTED_OUTPUT_TYPE: 400,
        UNSUPPORTED_FORMAT: 400,
        PROVENANCE_FAILURE: 422,
        RENDERING_FAILURE: 500,
        APPROVAL_NOT_ALLOWED: 409,
        ARCHIVE_NOT_ALLOWED: 409,
        CROSS_PROJECT_BINDING: 409,
        INVALID_TRANSITION: 409,
        PATH_TRAVERSAL_BLOCKED: 400,
    }
    return mapping.get(code, 400)
