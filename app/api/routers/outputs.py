"""Output API Routers — FastAPI routes for publication outputs.

Phase 2H: Thin HTTP layer over the OutputService. Routes handle request
parsing, dependency injection, HTTP status mapping, and response
serialization only. Business logic lives in app/outputs/service.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.outputs.service import (
    OutputService,
    OutputServiceResult,
    get_output_service,
    get_reproducibility_service,
    ReproducibilityService,
)
from app.api.schemas import (
    ServiceEnvelope,
    OutputResponse,
    FigureCreateRequest,
    TableCreateRequest,
    ReportCreateRequest,
    ReproducibilityCreateRequest,
    ReproducibilityVerifyRequest,
    ApprovalRequest,
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncSession:
    """Provide a database session for request handlers."""
    async with get_session() as session:
        yield session


def _get_service(output_type: str, session: AsyncSession) -> OutputService:
    return get_output_service(output_type, session)


def _result_to_envelope(result: OutputServiceResult) -> ServiceEnvelope:
    """Convert a service result to an API envelope."""
    envelope = ServiceEnvelope(success=result.success)
    if result.data:
        if "output" in result.data:
            envelope.output = OutputResponse(**result.data["output"])
        if "outputs" in result.data:
            envelope.outputs = [
                OutputResponse(**o) for o in result.data["outputs"]
            ]
        if "count" in result.data:
            envelope.count = result.data["count"]
        if "package" in result.data:
            envelope.package = result.data["package"]
        # Pass through other data keys
        for key in (
            "output_id", "valid", "lifecycle", "artifact_hash",
            "artifact_format", "artifact_size_bytes", "idempotent",
            "equivalent_output_id", "message", "artifact_path",
            "mime_type", "integrity_verified", "manifest", "manifest_hash",
            "verified", "verification", "package_path",
            "includes_raw_data", "contents",
        ):
            if key in result.data:
                setattr(envelope, key, result.data[key])
    if result.errors:
        envelope.errors = result.errors
    if result.warnings:
        envelope.warnings = result.warnings
    if result.error_code:
        envelope.error_code = result.error_code
    return envelope


def _raise_for_result(result: OutputServiceResult) -> ServiceEnvelope:
    """Raise HTTPException on failure; return envelope on success."""
    if not result.success:
        raise HTTPException(
            status_code=result.http_status,
            detail={
                "error_code": result.error_code,
                "errors": result.errors,
            },
        )
    return _result_to_envelope(result)


# ---------------------------------------------------------------------------
# Generic output router builder
# ---------------------------------------------------------------------------

def _build_output_router(output_type: str, tags: list[str]) -> APIRouter:
    """Build a CRUD + lifecycle router for an output type."""
    router = APIRouter(tags=tags)
    base = f"/api/outputs/{output_type}s"
    plural = output_type

    @router.post(base, response_model=ServiceEnvelope, status_code=201)
    async def create_output(
        request: FigureCreateRequest | TableCreateRequest | ReportCreateRequest,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        spec_dict = request.to_spec_dict()
        result = await service.create(spec_dict)
        return _raise_for_result(result)

    @router.get(base, response_model=ServiceEnvelope)
    async def list_outputs(
        project_id: str | None = None,
        dataset_id: str | None = None,
        analysis_version_id: str | None = None,
        status: str | None = None,
        analytical_classification: str | None = None,
        prespecified_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.list(
            project_id=project_id,
            dataset_id=dataset_id,
            analysis_version_id=analysis_version_id,
            status=status,
            analytical_classification=analytical_classification,
            prespecified_status=prespecified_status,
            limit=limit,
            offset=offset,
        )
        return _raise_for_result(result)

    @router.get(f"{base}/{{output_id}}", response_model=ServiceEnvelope)
    async def get_output(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.get(output_id)
        return _raise_for_result(result)

    @router.post(f"{base}/{{output_id}}/validate", response_model=ServiceEnvelope)
    async def validate_output(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.validate(output_id)
        return _raise_for_result(result)

    @router.post(f"{base}/{{output_id}}/render", response_model=ServiceEnvelope)
    async def render_output(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.render(output_id)
        return _raise_for_result(result)

    @router.get(f"{base}/{{output_id}}/artifact", response_model=ServiceEnvelope)
    async def get_artifact(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.get_artifact(output_id)
        return _raise_for_result(result)

    @router.get(f"{base}/{{output_id}}/artifact/download")
    async def download_artifact(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        try:
            content, mime_type, filename = await service.read_artifact_bytes(
                output_id
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        return Response(
            content=content,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    @router.post(f"{base}/{{output_id}}/approve", response_model=ServiceEnvelope)
    async def approve_output(
        output_id: str,
        request: ApprovalRequest | None = None,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        req = request or ApprovalRequest()
        result = await service.approve(
            output_id,
            approved_by=req.approved_by,
            notes=req.notes,
        )
        return _raise_for_result(result)

    @router.post(f"{base}/{{output_id}}/review", response_model=ServiceEnvelope)
    async def request_review(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.request_review(output_id)
        return _raise_for_result(result)

    @router.post(f"{base}/{{output_id}}/archive", response_model=ServiceEnvelope)
    async def archive_output(
        output_id: str,
        session: AsyncSession = Depends(get_db),
    ):
        service = _get_service(output_type, session)
        result = await service.archive(output_id)
        return _raise_for_result(result)

    return router


# ---------------------------------------------------------------------------
# Specific routers
# ---------------------------------------------------------------------------

figure_router = _build_output_router("figure", ["Figures"])
table_router = _build_output_router("table", ["Tables"])
report_router = _build_output_router("report", ["Reports"])


# ---------------------------------------------------------------------------
# Reproducibility router
# ---------------------------------------------------------------------------

repro_router = APIRouter(tags=["Reproducibility"])


@repro_router.post("/api/outputs/reproducibility", response_model=ServiceEnvelope, status_code=201)
async def create_package(
    request: ReproducibilityCreateRequest,
    session: AsyncSession = Depends(get_db),
):
    service = get_reproducibility_service(session)
    result = await service.create(request.model_dump())
    return _raise_for_result(result)


@repro_router.get("/api/outputs/reproducibility/{package_id}", response_model=ServiceEnvelope)
async def get_package(
    package_id: str,
    session: AsyncSession = Depends(get_db),
):
    service = get_reproducibility_service(session)
    result = await service.get(package_id)
    return _raise_for_result(result)


@repro_router.get("/api/outputs/reproducibility/{package_id}/manifest", response_model=ServiceEnvelope)
async def get_manifest(
    package_id: str,
    session: AsyncSession = Depends(get_db),
):
    service = get_reproducibility_service(session)
    result = await service.get_manifest(package_id)
    return _raise_for_result(result)


@repro_router.post("/api/outputs/reproducibility/{package_id}/verify", response_model=ServiceEnvelope)
async def verify_package(
    package_id: str,
    request: ReproducibilityVerifyRequest | None = None,
    session: AsyncSession = Depends(get_db),
):
    service = get_reproducibility_service(session)
    req = request or ReproducibilityVerifyRequest()
    result = await service.verify(
        package_id,
        expected_dataset_hash=req.expected_dataset_hash,
    )
    return _raise_for_result(result)


@repro_router.get("/api/outputs/reproducibility/{package_id}/artifact", response_model=ServiceEnvelope)
async def get_package_artifact(
    package_id: str,
    session: AsyncSession = Depends(get_db),
):
    service = get_reproducibility_service(session)
    result = await service.get_artifact_info(package_id)
    return _raise_for_result(result)
