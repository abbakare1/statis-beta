"""STATIS FastAPI Application — main entry point."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from config.settings import get_settings
from app.core.database import init_db, close_db, get_session
from app.orchestration.orchestrator import Orchestrator
from app.statistics.engine import (
    descriptive_statistics, independent_t_test, paired_t_test,
    chi_square_test, correlation_test, one_way_anova,
    logistic_regression, linear_regression, mann_whitney_u,
    wilcoxon_signed_rank, fisher_exact_test, power_analysis_t_test,
)
from app.memory.audit import get_audit_log, log_audit
from app.models import Project, Dataset, User
from app.security.auth import (
    authenticate_user, create_access_token, create_invitation,
    redeem_invitation, list_users, deactivate_user, activate_user,
)
from app.security.dependencies import get_current_user, get_current_admin_user, get_project_owner, validate_project_ownership
from app.data import plan_persistence
import sqlalchemy as sa

import json
import shutil


settings = get_settings()
orchestrator = Orchestrator()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("STATIS database initialized")
    yield
    await close_db()
    logger.info("STATIS shutdown complete")


app = FastAPI(
    title="STATIS",
    description="Scientific Statistical Analysis and Research Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Output API Routers (Phase 2H)
# ---------------------------------------------------------------------------

from app.api.routers import figure_router, table_router, report_router, repro_router  # noqa: E402

app.include_router(figure_router)
app.include_router(table_router)
app.include_router(report_router)
app.include_router(repro_router)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@app.post("/api/projects")
async def create_project(
    name: str = Form(...),
    description: str = Form(""),
    research_question: str = Form(""),
    study_design: str = Form(""),
    current_user: User = Depends(get_current_user),
):
    # Check project quota
    async with get_session() as session:
        result = await session.execute(
            sa.select(sa.func.count(Project.id)).where(Project.owner_id == current_user.id)
        )
        project_count = result.scalar()
        if project_count >= current_user.max_projects:
            raise HTTPException(429, f"Project limit reached ({current_user.max_projects} projects)")
        
        project = Project(
            name=name,
            description=description or None,
            research_question=research_question or None,
            study_design=study_design or None,
            owner_id=current_user.id,
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)
    return {"id": project.id, "name": project.name, "status": project.status, "owner_id": project.owner_id}


@app.get("/api/projects")
async def list_projects(current_user: User = Depends(get_current_user)):
    async with get_session() as session:
        # Filter by owner
        result = await session.execute(
            sa.select(Project)
            .where(Project.owner_id == current_user.id)
            .order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()
    return [
        {"id": p.id, "name": p.name, "status": p.status, "created_at": p.created_at.isoformat()}
        for p in projects
    ]


# ---------------------------------------------------------------------------
# Project Summary (Phase 2I — dashboard aggregation)
# ---------------------------------------------------------------------------

@app.get("/api/projects/summary")
async def project_summary():
    """Aggregate project information for the researcher dashboard."""
    async with get_session() as session:
        projects = (await session.execute(sa.select(Project).order_by(Project.created_at.desc()))).scalars().all()
        if not projects:
            return {"projects": [], "count": 0}

        project_ids = [p.id for p in projects]

        # Dataset counts
        from app.models import Dataset
        ds_counts = {}
        ds_rows = (await session.execute(
            sa.select(Dataset.project_id, sa.func.count(Dataset.id)).where(
                Dataset.project_id.in_(project_ids)
            ).group_by(Dataset.project_id)
        )).all()
        for pid, cnt in ds_rows:
            ds_counts[pid] = cnt

        # Analysis plan counts
        from app.models import AnalysisPlanRecord
        plan_counts = {}
        plan_rows = (await session.execute(
            sa.select(AnalysisPlanRecord.project_id, sa.func.count(AnalysisPlanRecord.id)).where(
                AnalysisPlanRecord.project_id.in_(project_ids)
            ).group_by(AnalysisPlanRecord.project_id)
        )).all()
        for pid, cnt in plan_rows:
            plan_counts[pid] = cnt

        # Analysis run counts
        from app.models import AnalysisRunRecord
        run_counts = {}
        run_rows = (await session.execute(
            sa.select(AnalysisRunRecord.project_id, sa.func.count(AnalysisRunRecord.id)).where(
                AnalysisRunRecord.project_id.in_(project_ids)
            ).group_by(AnalysisRunRecord.project_id)
        )).all()
        for pid, cnt in run_rows:
            run_counts[pid] = cnt

        # Output counts (figures + tables + reports)
        from app.models import FigureRecord, TableRecord, ReportRecord
        output_counts = {}
        for model in (FigureRecord, TableRecord, ReportRecord):
            rows = (await session.execute(
                sa.select(model.project_id, sa.func.count(model.id)).where(
                    model.project_id.in_(project_ids)
                ).group_by(model.project_id)
            )).all()
            for pid, cnt in rows:
                output_counts[pid] = output_counts.get(pid, 0) + cnt

        # Last audit timestamp per project
        from app.models import AuditEntry
        last_audit = {}
        audit_rows = (await session.execute(
            sa.select(AuditEntry.project_id, sa.func.max(AuditEntry.timestamp)).where(
                AuditEntry.project_id.in_(project_ids)
            ).group_by(AuditEntry.project_id)
        )).all()
        for pid, ts in audit_rows:
            last_audit[pid] = ts.isoformat() if ts else None

        results = []
        for p in projects:
            results.append({
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "research_question": p.research_question,
                "study_design": p.study_design,
                "status": p.status,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat() if p.updated_at else p.created_at.isoformat(),
                "dataset_count": ds_counts.get(p.id, 0),
                "analysis_plan_count": plan_counts.get(p.id, 0),
                "analysis_run_count": run_counts.get(p.id, 0),
                "output_count": output_counts.get(p.id, 0),
                "last_audit_at": last_audit.get(p.id),
            })
        return {"projects": results, "count": len(results)}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, project: Project = Depends(get_project_owner)):
    return {
        "id": project.id, "name": project.name,
        "description": project.description,
        "research_question": project.research_question,
        "study_design": project.study_design,
        "status": project.status,
        "created_at": project.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Project Workflow Status (Phase 2I — project workspace overview)
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project_id}/workflow")
async def get_project_workflow(project_id: str, project: Project = Depends(get_project_owner)):
    """Derive workflow status for a project from actual backend state."""
    async with get_session() as session:

        # Datasets
        from app.models import Dataset
        datasets = (await session.execute(
            sa.select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
        )).scalars().all()
        dataset_count = len(datasets)

        # Data quality profiles
        from app.models import QualityProfileRecord
        quality_profiles = (await session.execute(
            sa.select(QualityProfileRecord).where(QualityProfileRecord.dataset_id.in_(
                [d.id for d in datasets]
            ) if datasets else sa.false())
        )).scalars().all()

        # Analysis plans
        from app.models import AnalysisPlanRecord, AnalysisEntryRecord
        plans = (await session.execute(
            sa.select(AnalysisPlanRecord).where(AnalysisPlanRecord.project_id == project_id)
        )).scalars().all()
        plan_count = len(plans)

        # Analysis entries (analyses within plans)
        plan_ids = [p.id for p in plans]
        entries = []
        if plan_ids:
            entries = (await session.execute(
                sa.select(AnalysisEntryRecord).where(AnalysisEntryRecord.plan_id.in_(plan_ids))
            )).scalars().all()

        # Analysis versions (validation/approval state)
        from app.models import AnalysisVersionRecord
        entry_ids = [e.id for e in entries]
        versions = []
        if entry_ids:
            versions = (await session.execute(
                sa.select(AnalysisVersionRecord).where(AnalysisVersionRecord.analysis_id.in_(entry_ids))
            )).scalars().all()

        # Analysis runs
        from app.models import AnalysisRunRecord
        runs = (await session.execute(
            sa.select(AnalysisRunRecord).where(AnalysisRunRecord.project_id == project_id)
        )).scalars().all()

        # Outputs
        from app.models import FigureRecord, TableRecord, ReportRecord
        figures = (await session.execute(
            sa.select(FigureRecord).where(FigureRecord.project_id == project_id)
        )).scalars().all()
        tables = (await session.execute(
            sa.select(TableRecord).where(TableRecord.project_id == project_id)
        )).scalars().all()
        reports = (await session.execute(
            sa.select(ReportRecord).where(ReportRecord.project_id == project_id)
        )).scalars().all()

        # Variable roles
        from app.models import VariableRoleResolutionRecord
        variable_roles = (await session.execute(
            sa.select(VariableRoleResolutionRecord).where(VariableRoleResolutionRecord.project_id == project_id)
        )).scalars().all()

        # Reproducibility packages
        from app.models import ReproducibilityPackageRecord
        repro_packages = (await session.execute(
            sa.select(ReproducibilityPackageRecord).where(ReproducibilityPackageRecord.project_id == project_id)
        )).scalars().all()

        # Derive workflow step states
        has_research_question = bool(project.research_question)
        has_study_design = bool(project.study_design)
        has_dataset = dataset_count > 0
        has_quality = len(quality_profiles) > 0
        has_variable_roles = len(variable_roles) > 0
        has_power = any(e.power_assessment for e in entries if e.power_assessment)
        has_analysis_plan = plan_count > 0
        has_validated = any(v.validation_status == "passed" for v in versions)
        has_approved = any(v.approval_status == "approved" for v in versions)
        has_executed = any(r.success for r in runs)
        has_results = len(runs) > 0
        has_figures = len(figures) > 0
        has_tables = len(tables) > 0
        has_reports = len(reports) > 0
        has_repro = len(repro_packages) > 0

        # Derive overall status
        if has_repro and has_reports:
            workflow_status = "complete"
        elif has_approved and not has_executed:
            workflow_status = "needs_review"
        elif has_analysis_plan and not has_approved:
            workflow_status = "needs_review"
        elif has_dataset and not has_quality:
            workflow_status = "needs_review"
        elif has_dataset:
            workflow_status = "in_progress"
        elif has_research_question or has_study_design:
            workflow_status = "in_progress"
        else:
            workflow_status = "not_started"

        # Unresolved review items
        review_items = []
        if has_dataset and not has_quality:
            review_items.append("Data quality assessment required")
        if has_analysis_plan and not has_approved:
            review_items.append("Analysis plan requires review")
        if has_approved and not has_executed:
            review_items.append("Approved analyses ready for execution")
        if any(v.validation_status == "failed" for v in versions):
            review_items.append("Some analyses have validation failures")
        if any(f.lifecycle in ("draft", "validation_failed") for f in figures):
            review_items.append("Some figures require validation")
        if any(t.lifecycle in ("draft", "validation_failed") for t in tables):
            review_items.append("Some tables require validation")

        def _step_state(done: bool, needs_review: bool = False) -> str:
            if needs_review:
                return "needs_review"
            return "complete" if done else "not_started"

        workflow_steps = [
            {"step": "research_question", "label": "Research Question", "state": _step_state(has_research_question)},
            {"step": "study_design", "label": "Study Design", "state": _step_state(has_study_design)},
            {"step": "dataset", "label": "Dataset", "state": _step_state(has_dataset)},
            {"step": "data_quality", "label": "Data Quality", "state": _step_state(has_quality, has_dataset and not has_quality)},
            {"step": "variable_roles", "label": "Variable Roles", "state": _step_state(has_variable_roles)},
            {"step": "power", "label": "Power", "state": _step_state(has_power)},
            {"step": "analysis_plan", "label": "Analysis Plan", "state": _step_state(has_analysis_plan, has_analysis_plan and not has_approved)},
            {"step": "researcher_review", "label": "Researcher Review", "state": _step_state(has_approved, has_analysis_plan and not has_approved)},
            {"step": "statistical_analysis", "label": "Statistical Analysis", "state": _step_state(has_executed)},
            {"step": "diagnostics", "label": "Diagnostics", "state": _step_state(has_results and any(getattr(r, 'diagnostic_summary', None) for r in runs))},
            {"step": "defensibility", "label": "Defensibility", "state": _step_state(has_results and any(getattr(r, 'defensibility_status', None) and r.defensibility_status != 'not_assessed' for r in runs))},
            {"step": "results", "label": "Results", "state": _step_state(has_results)},
            {"step": "figures_tables", "label": "Figures & Tables", "state": _step_state(has_figures or has_tables)},
            {"step": "report", "label": "Report", "state": _step_state(has_reports)},
            {"step": "reproducibility", "label": "Reproducibility", "state": _step_state(has_repro)},
        ]

        return {
            "project_id": project_id,
            "project_name": project.name,
            "study_design": project.study_design,
            "research_question": project.research_question,
            "workflow_status": workflow_status,
            "review_items": review_items,
            "steps": workflow_steps,
            "counts": {
                "datasets": dataset_count,
                "quality_profiles": len(quality_profiles),
                "analysis_plans": plan_count,
                "analyses": len(entries),
                "analysis_versions": len(versions),
                "analysis_runs": len(runs),
                "figures": len(figures),
                "tables": len(tables),
                "reports": len(reports),
                "variable_roles": len(variable_roles),
                "reproducibility_packages": len(repro_packages),
            },
        }


# ---------------------------------------------------------------------------
# Project-scoped Resources (Phase 2I)
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project_id}/plans")
async def list_project_plans(project_id: str, project: Project = Depends(get_project_owner)):
    """List analysis plans for a project."""
    from app.data.plan_store import list_plans
    plans = await list_plans(project_id=project_id)
    return {"plans": plans, "count": len(plans)}


@app.get("/api/projects/{project_id}/runs")
async def list_project_runs(project_id: str, limit: int = 100, project: Project = Depends(get_project_owner)):
    """List analysis runs for a project."""
    from app.data.run_store import list_analysis_runs
    runs = await list_analysis_runs(project_id=project_id, limit=limit)
    return {"runs": runs, "count": len(runs)}


@app.get("/api/projects/{project_id}/outputs")
async def list_project_outputs(project_id: str, project: Project = Depends(get_project_owner)):
    """List all publication outputs (figures, tables, reports) for a project."""
    from app.models import FigureRecord, TableRecord, ReportRecord

    async with get_session() as session:
        figures = (await session.execute(
            sa.select(FigureRecord).where(FigureRecord.project_id == project_id).order_by(FigureRecord.created_at.desc())
        )).scalars().all()
        tables = (await session.execute(
            sa.select(TableRecord).where(TableRecord.project_id == project_id).order_by(TableRecord.created_at.desc())
        )).scalars().all()
        reports = (await session.execute(
            sa.select(ReportRecord).where(ReportRecord.project_id == project_id).order_by(ReportRecord.created_at.desc())
        )).scalars().all()

    def _fig_dict(f):
        return {
            "id": f.id, "type": "figure", "figure_type": f.figure_type,
            "lifecycle": f.lifecycle, "analytical_classification": f.analytical_classification,
            "prespecified_status": f.prespecified_status, "artifact_format": f.artifact_format,
            "artifact_hash": f.artifact_hash, "created_at": f.created_at.isoformat() if f.created_at else None,
        }

    def _tbl_dict(t):
        return {
            "id": t.id, "type": "table", "table_type": t.table_type,
            "lifecycle": t.lifecycle, "analytical_classification": t.analytical_classification,
            "prespecified_status": t.prespecified_status, "artifact_format": t.artifact_format,
            "artifact_hash": t.artifact_hash, "created_at": t.created_at.isoformat() if t.created_at else None,
        }

    def _rep_dict(r):
        return {
            "id": r.id, "type": "report", "title": r.title,
            "lifecycle": r.lifecycle, "analytical_classification": r.analytical_classification,
            "prespecified_status": r.prespecified_status, "artifact_format": r.artifact_format,
            "artifact_hash": r.artifact_hash, "created_at": r.created_at.isoformat() if r.created_at else None,
        }

    return {
        "figures": [_fig_dict(f) for f in figures],
        "tables": [_tbl_dict(t) for t in tables],
        "reports": [_rep_dict(r) for r in reports],
        "count": len(figures) + len(tables) + len(reports),
    }


@app.get("/api/projects/{project_id}/variable-roles")
async def list_project_variable_roles(project_id: str, project: Project = Depends(get_project_owner)):
    """List variable role resolutions for a project."""
    from app.models import VariableRoleResolutionRecord
    async with get_session() as session:
        roles = (await session.execute(
            sa.select(VariableRoleResolutionRecord).where(
                VariableRoleResolutionRecord.project_id == project_id
            ).order_by(VariableRoleResolutionRecord.created_at.desc())
        )).scalars().all()
    return {
        "variable_roles": [
            {
                "id": r.id,
                "variable_name": r.variable_name,
                "variable_dtype": r.variable_dtype,
                "proposed_role": r.proposed_role,
                "role_source": r.role_source,
                "confidence": r.confidence,
                "status": r.status,
                "requires_researcher_review": r.requires_researcher_review,
                "reasoning_summary": r.reasoning_summary,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in roles
        ],
        "count": len(roles),
    }


@app.get("/api/projects/{project_id}/quality-profile")
async def get_project_quality_profile(project_id: str, project: Project = Depends(get_project_owner)):
    """Get data quality profiles for all datasets in a project."""
    from app.models import Dataset, QualityProfileRecord
    async with get_session() as session:
        datasets = (await session.execute(
            sa.select(Dataset).where(Dataset.project_id == project_id)
        )).scalars().all()
        dataset_ids = [d.id for d in datasets]
        profiles = []
        if dataset_ids:
            profiles = (await session.execute(
                sa.select(QualityProfileRecord).where(QualityProfileRecord.dataset_id.in_(dataset_ids))
            )).scalars().all()
    return {
        "profiles": [
            {
                "id": p.id,
                "dataset_id": p.dataset_id,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "profile": json.loads(p.profile_json) if p.profile_json else {},
            }
            for p in profiles
        ],
        "count": len(profiles),
    }


@app.get("/api/projects/{project_id}/reproducibility")
async def list_project_reproducibility(project_id: str, project: Project = Depends(get_project_owner)):
    """List reproducibility packages for a project."""
    from app.models import ReproducibilityPackageRecord
    async with get_session() as session:
        packages = (await session.execute(
            sa.select(ReproducibilityPackageRecord).where(
                ReproducibilityPackageRecord.project_id == project_id
            ).order_by(ReproducibilityPackageRecord.created_at.desc())
        )).scalars().all()
    return {
        "packages": [
            {
                "id": p.id,
                "package_id": p.package_id,
                "verified": p.verified,
                "verified_at": p.verified_at.isoformat() if p.verified_at else None,
                "manifest_hash": p.manifest_hash,
                "includes_raw_data": p.includes_raw_data,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in packages
        ],
        "count": len(packages),
    }


# ---------------------------------------------------------------------------
# Dataset Upload
# ---------------------------------------------------------------------------

def _parse_dataset_preview(stored_path: str, head_n: int = 10) -> dict:
    """Parse a stored dataset file and return shape + preview metadata.

    Read-only. Never modifies the source file. Backend remains authoritative.
    """
    import pandas as pd
    path = Path(stored_path)
    if not path.exists():
        return {"error": "stored file not found"}

    suffix = path.suffix.lower()
    try:
        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, nrows=None)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return {"error": f"unsupported file type: {suffix}"}
    except Exception as e:
        return {"error": f"parse failure: {type(e).__name__}: {e}"}

    columns_meta = []
    for col in df.columns:
        series = df[col]
        columns_meta.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)),
        })

    head_df = df.head(head_n)
    head = json.loads(head_df.to_json(orient="records", date_format="iso"))

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": columns_meta,
        "head": head,
        "file_size_bytes": int(path.stat().st_size),
    }


@app.post("/api/projects/{project_id}/upload")
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    # Validate project ownership
    project = await validate_project_ownership(project_id, current_user)
    
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "Empty file — no data to upload.")
    # Use user-specific upload limit
    user_max_mb = current_user.max_upload_mb
    if len(raw) > user_max_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {user_max_mb} MB limit for your account.")

    upload_path = settings.UPLOAD_DIR / f"{project_id}_{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(raw)

    preview = _parse_dataset_preview(str(upload_path))
    if "error" in preview:
        raise HTTPException(422, f"Cannot parse dataset: {preview['error']}")

    # Canonical dataset identity (row/column-order-invariant SHA-256).
    # The stored content_hash must be the canonical identity — every
    # downstream consumer (executor guard, output render, provenance)
    # verifies against compute_dataset_identity, never a raw file hash.
    from app.data.dataset_identity import compute_dataset_identity
    try:
        identity = compute_dataset_identity(upload_path)
    except Exception as e:
        raise HTTPException(422, f"Cannot parse dataset: {type(e).__name__}: {e}")
    content_hash = identity["content_hash"]

    dataset = Dataset(
        project_id=project_id,
        original_filename=file.filename,
        stored_path=str(upload_path),
        content_hash=content_hash,
        row_count=identity["row_count"],
        column_count=identity["column_count"],
    )
    async with get_session() as session:
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

    return {
        "id": dataset.id,
        "filename": file.filename,
        "content_hash": content_hash,
        "row_count": preview.get("n_rows"),
        "column_count": preview.get("n_columns"),
        "preview": preview,
    }


@app.get("/api/projects/{project_id}/datasets")
async def list_datasets(project_id: str, project: Project = Depends(get_project_owner)):
    async with get_session() as session:
        result = await session.execute(
            sa.select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
        )
        datasets = result.scalars().all()
    return [
        {"id": d.id, "filename": d.original_filename, "version": d.version,
         "content_hash": d.content_hash, "stored_path": d.stored_path,
         "row_count": d.row_count, "column_count": d.column_count,
         "created_at": d.created_at.isoformat()}
        for d in datasets
    ]


@app.get("/api/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, head: int = 10):
    """Return read-only shape + preview for a stored dataset."""
    async with get_session() as session:
        result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    preview = _parse_dataset_preview(dataset.stored_path, head_n=max(1, min(head, 50)))
    if "error" in preview:
        raise HTTPException(422, preview["error"])
    return {
        "dataset_id": dataset_id,
        "filename": dataset.original_filename,
        "content_hash": dataset.content_hash,
        **preview,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/run")
async def run_pipeline(request: Request):
    body = await request.json()
    project_id = body.get("project_id")
    dataset_id = body.get("dataset_id")
    research_question = body.get("research_question", "")
    study_design = body.get("study_design", "")

    if not project_id or not dataset_id:
        raise HTTPException(400, "project_id and dataset_id are required")

    # Get dataset path
    async with get_session() as session:
        result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    pipeline_request = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "file_path": dataset.stored_path,
        "original_filename": dataset.original_filename,
        "research_question": research_question,
        "study_design": study_design,
        "hypotheses": body.get("hypotheses", []),
        "variables_of_interest": body.get("variables_of_interest", []),
        "covariates": body.get("covariates", []),
        "outcomes": body.get("outcomes", []),
    }

    result = await orchestrator.run_full_pipeline(pipeline_request)
    return result


# ---------------------------------------------------------------------------
# Statistical Analyses
# ---------------------------------------------------------------------------

@app.post("/api/analysis/descriptive")
async def run_descriptive(request: Request):
    body = await request.json()
    file_path = body.get("file_path")
    if not file_path:
        raise HTTPException(400, "file_path is required")
    return descriptive_statistics(Path(file_path), columns=body.get("columns"))


@app.post("/api/analysis/t-test")
async def run_t_test(request: Request):
    body = await request.json()
    file_path = body.get("file_path")
    if not file_path:
        raise HTTPException(400, "file_path is required")
    return independent_t_test(
        Path(file_path),
        outcome=body["outcome"],
        group=body["group"],
        group1=body.get("group1"),
        group2=body.get("group2"),
        equal_var=body.get("equal_var", False),
        alternative=body.get("alternative", "two-sided"),
    )


@app.post("/api/analysis/paired-t-test")
async def run_paired_t_test(request: Request):
    body = await request.json()
    return paired_t_test(
        Path(body["file_path"]),
        outcome1=body["outcome1"],
        outcome2=body["outcome2"],
        alternative=body.get("alternative", "two-sided"),
    )


@app.post("/api/analysis/chi-square")
async def run_chi_square(request: Request):
    body = await request.json()
    return chi_square_test(
        Path(body["file_path"]),
        variable1=body["variable1"],
        variable2=body["variable2"],
    )


@app.post("/api/analysis/correlation")
async def run_correlation(request: Request):
    body = await request.json()
    return correlation_test(
        Path(body["file_path"]),
        x=body["x"],
        y=body["y"],
        method=body.get("method", "pearson"),
    )


@app.post("/api/analysis/anova")
async def run_anova(request: Request):
    body = await request.json()
    return one_way_anova(
        Path(body["file_path"]),
        outcome=body["outcome"],
        group=body["group"],
    )


@app.post("/api/analysis/logistic-regression")
async def run_logistic_regression(request: Request):
    body = await request.json()
    return logistic_regression(
        Path(body["file_path"]),
        outcome=body["outcome"],
        predictors=body["predictors"],
        covariates=body.get("covariates"),
    )


@app.post("/api/analysis/linear-regression")
async def run_linear_regression(request: Request):
    body = await request.json()
    return linear_regression(
        Path(body["file_path"]),
        outcome=body["outcome"],
        predictors=body["predictors"],
        covariates=body.get("covariates"),
    )


@app.post("/api/analysis/mann-whitney")
async def run_mann_whitney(request: Request):
    body = await request.json()
    return mann_whitney_u(
        Path(body["file_path"]),
        outcome=body["outcome"],
        group=body["group"],
        group1=body.get("group1"),
        group2=body.get("group2"),
        alternative=body.get("alternative", "two-sided"),
    )


@app.post("/api/analysis/wilcoxon")
async def run_wilcoxon(request: Request):
    body = await request.json()
    return wilcoxon_signed_rank(
        Path(body["file_path"]),
        outcome1=body["outcome1"],
        outcome2=body["outcome2"],
        alternative=body.get("alternative", "two-sided"),
    )


@app.post("/api/analysis/fisher-exact")
async def run_fisher_exact(request: Request):
    body = await request.json()
    return fisher_exact_test(
        Path(body["file_path"]),
        variable1=body["variable1"],
        variable2=body["variable2"],
    )


@app.post("/api/analysis/power")
async def run_power_analysis(request: Request):
    body = await request.json()
    return power_analysis_t_test(
        effect_size=body.get("effect_size"),
        n_per_group=body.get("n_per_group"),
        alpha=body.get("alpha", 0.05),
        power=body.get("power", 0.80),
        alternative=body.get("alternative", "two-sided"),
    )


# ---------------------------------------------------------------------------
# Analysis Plans (Phase 2C / Phase 2J — DB persistence)
# ---------------------------------------------------------------------------


@app.post("/api/analysis-plan/create")
async def create_analysis_plan(request: Request):
    """Create a new analysis plan using the planner agent."""
    body = await request.json()
    project_id = body.get("project_id")
    dataset_id = body.get("dataset_id")
    research_question = body.get("research_question", "")
    study_design = body.get("study_design", "")

    if not project_id:
        raise HTTPException(400, "project_id is required")

    # Get dataset path if dataset_id provided
    file_path = None
    if dataset_id:
        async with get_session() as session:
            result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
        if dataset:
            file_path = dataset.stored_path

    # The planner needs a data dictionary. Callers that cannot build one
    # (the frontend) omit it — the backend then derives it from the dataset.
    data_dictionary = body.get("data_dictionary") or {}
    if not data_dictionary.get("variables"):
        if not file_path:
            raise HTTPException(
                422,
                "Cannot build a data profile: provide data_dictionary "
                "or a dataset_id belonging to this project.",
            )
        dd_result = await orchestrator.agents["data_dictionary"].run({
            "file_path": file_path,
            "project_id": project_id,
            "research_question": research_question,
        })
        if not dd_result.success:
            raise HTTPException(
                500,
                f"Data dictionary generation failed: {dd_result.data.get('error', 'unknown')}",
            )
        data_dictionary = dd_result.data

    # Run the planner agent
    planner = orchestrator.agents["analysis_planner"]
    planner_result = await planner.run({
        "project_id": project_id,
        "dataset_id": dataset_id,
        "file_path": file_path,
        "research_question": research_question,
        "study_design": study_design,
        "data_dictionary": data_dictionary,
        "data_quality": body.get("data_quality", {}),
        "power_assessment": body.get("power_assessment", {}),
    })

    if not planner_result.success:
        error = planner_result.data.get("error", "unknown")
        if error == "Insufficient data profile information in context.":
            error += (
                " No outcome variable could be identified from the research "
                "question — name the outcome column in the research question."
            )
        raise HTTPException(500, f"Planning failed: {error}")

    # Extract plan data from the result
    plan_data = planner_result.data

    # Store the plan in the database
    plan_id = plan_data.get("plan_id", str(uuid.uuid4()))
    plan = {
        "plan_id": plan_id,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "research_question": research_question,
        "study_design": study_design,
        "analyses": plan_data.get("analyses", []),
        "warnings": plan_data.get("warnings", []),
        "rejected_methods": plan_data.get("rejected_methods", []),
    }
    await plan_persistence.save_plan(plan)

    # Audit
    await log_audit(
        action="analysis_plan_created",
        agent="analysis_planner",
        decision=f"Created plan with {plan_data.get('n_analyses', 0)} analyses",
        project_id=project_id,
    )

    return {
        "plan_id": plan_id,
        "n_analyses": plan_data.get("n_analyses", 0),
        "analyses": plan_data.get("analyses", []),
        "warnings": plan_data.get("warnings", []),
        "rejected_methods": plan_data.get("rejected_methods", []),
    }


@app.get("/api/analysis-plan/{plan_id}")
async def get_analysis_plan(plan_id: str):
    """Retrieve an analysis plan."""
    plan = await plan_persistence.load_plan(plan_id)
    if not plan:
        raise HTTPException(404, "Analysis plan not found")
    return plan


@app.post("/api/analysis-plan/{analysis_id}/approve")
async def approve_analysis(analysis_id: str, request: Request):
    """Approve an analysis for execution."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    plan_id = body.get("plan_id")

    if not plan_id:
        raise HTTPException(400, "plan_id is required in request body")

    plan = await plan_persistence.load_plan(plan_id)
    if not plan:
        raise HTTPException(404, "Analysis plan not found")

    # Find the analysis
    analysis = None
    for a in plan["analyses"]:
        if a["analysis_id"] == analysis_id:
            analysis = a
            break

    if not analysis:
        raise HTTPException(404, f"Analysis {analysis_id} not found in plan")

    # Check if validation passed
    if analysis.get("validation") == "block":
        raise HTTPException(
            400,
            f"Cannot approve analysis with BLOCK validation status. "
            f"Validation blockers must be resolved first.",
        )

    # Update status and persist
    await plan_persistence.update_analysis(plan_id, analysis_id, {
        "approval": "approved",
        "lifecycle": "approved",
    })

    await log_audit(
        action="analysis_approved",
        agent="researcher",
        decision=f"Approved analysis {analysis_id} ({analysis.get('method', 'unknown')})",
        researcher_decision="approve",
    )

    return {"status": "approved", "analysis_id": analysis_id}


@app.post("/api/analysis-plan/{analysis_id}/reject")
async def reject_analysis(analysis_id: str, request: Request):
    """Reject an analysis."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    plan_id = body.get("plan_id")
    reason = body.get("reason", "")

    if not plan_id:
        raise HTTPException(400, "plan_id is required in request body")

    plan = await plan_persistence.load_plan(plan_id)
    if not plan:
        raise HTTPException(404, "Analysis plan not found")

    analysis = None
    for a in plan["analyses"]:
        if a["analysis_id"] == analysis_id:
            analysis = a
            break

    if not analysis:
        raise HTTPException(404, f"Analysis {analysis_id} not found in plan")

    # Update status and persist
    await plan_persistence.update_analysis(plan_id, analysis_id, {
        "approval": "rejected",
        "lifecycle": "draft",
    })

    await log_audit(
        action="analysis_rejected",
        agent="researcher",
        decision=f"Rejected analysis {analysis_id}: {reason}",
        researcher_decision="reject",
    )

    return {"status": "rejected", "analysis_id": analysis_id, "reason": reason}


@app.post("/api/analysis-plan/{analysis_id}/lock")
async def lock_analysis(analysis_id: str, request: Request):
    """Lock an approved analysis for execution."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    plan_id = body.get("plan_id")

    if not plan_id:
        raise HTTPException(400, "plan_id is required in request body")

    plan = await plan_persistence.load_plan(plan_id)
    if not plan:
        raise HTTPException(404, "Analysis plan not found")

    analysis = None
    for a in plan["analyses"]:
        if a["analysis_id"] == analysis_id:
            analysis = a
            break

    if not analysis:
        raise HTTPException(404, f"Analysis {analysis_id} not found in plan")

    if analysis.get("approval") != "approved":
        raise HTTPException(
            400,
            f"Cannot lock analysis with approval status '{analysis.get('approval')}'. "
            f"Must be approved first.",
        )

    # Update lifecycle and persist
    await plan_persistence.update_analysis(plan_id, analysis_id, {
        "lifecycle": "locked",
    })

    await log_audit(
        action="analysis_locked",
        agent="researcher",
        decision=f"Locked analysis {analysis_id}",
    )

    return {"status": "locked", "analysis_id": analysis_id}


@app.post("/api/analysis-plan/{analysis_id}/execute")
async def execute_analysis(
    analysis_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Execute a locked analysis through the statistical engine."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    plan_id = body.get("plan_id")

    if not plan_id:
        raise HTTPException(400, "plan_id is required in request body")

    plan = await plan_persistence.load_plan(plan_id)
    if not plan:
        raise HTTPException(404, "Analysis plan not found")

    # Check analysis quota
    async with get_session() as session:
        from app.models import AnalysisRunRecord
        result = await session.execute(
            sa.select(sa.func.count(AnalysisRunRecord.id)).where(
                AnalysisRunRecord.project_id == plan.get("project_id")
            )
        )
        run_count = result.scalar()
        if run_count >= current_user.max_analyses:
            raise HTTPException(429, f"Analysis limit reached ({current_user.max_analyses} analyses)")

    analysis = None
    for a in plan["analyses"]:
        if a["analysis_id"] == analysis_id:
            analysis = a
            break

    if not analysis:
        raise HTTPException(404, f"Analysis {analysis_id} not found in plan")

    # Guard: must be locked
    if analysis.get("lifecycle") != "locked":
        raise HTTPException(
            400,
            f"Cannot execute analysis with lifecycle '{analysis.get('lifecycle')}'. "
            f"Must be locked first.",
        )

    # Get dataset path
    dataset_id = plan.get("dataset_id")
    dataset = None
    file_path = None
    if dataset_id:
        async with get_session() as session:
            result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
        if dataset:
            file_path = dataset.stored_path

    if not file_path or dataset is None:
        raise HTTPException(400, "No dataset found for this plan")

    # ---------------------------------------------------------------
    # Rebuild the formal specification from the approved plan entry.
    # Execution runs through the executor's guarded spec path: registry
    # membership, lifecycle/approval enforcement, dataset identity
    # verification, and required-variable checks all execute BEFORE any
    # computation. Method-specific parameter mapping (group/x/y/...)
    # comes from the executor's _spec_to_context.
    # ---------------------------------------------------------------
    from app.data.dataset_identity import compute_dataset_identity
    from app.data.run_store import create_analysis_run, complete_analysis_run, _json_safe
    from app.statistics.decision_trace import DecisionTrace

    try:
        identity = compute_dataset_identity(file_path)
    except Exception as e:
        raise HTTPException(422, f"Cannot read dataset for execution: {type(e).__name__}: {e}")

    spec_dict = {
        "analysis_method": analysis.get("method", ""),
        "primary_outcome": analysis.get("outcome"),
        "predictors": analysis.get("predictors", []) or [],
        "covariates": analysis.get("covariates", []) or [],
        "subgroup_variables": analysis.get("subgroup_variables", []) or [],
        "project_id": plan.get("project_id"),
        "dataset_id": dataset_id,
        "research_question": plan.get("research_question"),
        "study_design": plan.get("study_design"),
        "method_category": analysis.get("analysis_class", "primary"),
        "rationale": analysis.get("rationale"),
        "approval_status": analysis.get("approval", ""),
        "validation_status": analysis.get("validation", "pending"),
        # Binding: the file must still hash to the identity recorded at
        # upload/approval time — otherwise execution is BLOCKED (INV 2).
        "dataset_content_hash": dataset.content_hash,
    }

    # Run record: this execution becomes a first-class, fingerprinted
    # provenance object (dataset identity → run → result → outputs).
    run_id = await create_analysis_run(
        project_id=plan.get("project_id", "unknown"),
        analysis_type=analysis.get("method", ""),
        specification={
            "analysis_method": spec_dict["analysis_method"],
            "primary_outcome": spec_dict["primary_outcome"],
            "predictors": spec_dict["predictors"],
            "covariates": spec_dict["covariates"],
            "alpha": 0.05,
        },
        dataset_id=dataset_id,
        dataset_content_hash=dataset.content_hash,
        dataset_schema_hash=identity["schema_hash"],
        file_path=file_path,
    )

    trace = DecisionTrace(run_id, plan.get("project_id", "unknown"), analysis.get("method", ""))
    trace.add_data_ingestion(
        file_path=file_path,
        content_hash=dataset.content_hash,
        schema_hash=identity["schema_hash"],
        n_rows=identity["row_count"],
        n_cols=identity["column_count"],
        columns=identity["column_names"],
    )

    # Execute via the executor agent (guarded spec path)
    executor = orchestrator.agents["statistical_executor"]
    exec_result = await executor.run({
        "specification": spec_dict,
        "lifecycle": analysis.get("lifecycle", ""),
        "file_path": file_path,
    })

    trace.add_execution(
        result_summary={"success": exec_result.success, "n_fields": len(exec_result.data)},
        warnings=exec_result.warnings,
    )

    await complete_analysis_run(
        run_id,
        result=exec_result.data,
        success=exec_result.success,
        error_message=exec_result.data.get("error") if not exec_result.success else None,
        validation_status=analysis.get("validation"),
        approval_status=analysis.get("approval"),
        lifecycle="executed" if exec_result.success else analysis.get("lifecycle"),
        guardrails_passed=exec_result.success,
        warnings=exec_result.warnings,
        decision_trace=trace.to_list(),
    )

    if exec_result.success:
        # Persist lifecycle change
        await plan_persistence.update_analysis(plan_id, analysis_id, {
            "lifecycle": "executed",
        })

    await log_audit(
        action="analysis_executed" if exec_result.success else "analysis_execution_failed",
        agent="statistical_executor",
        decision=f"Execution {'succeeded' if exec_result.success else 'failed'} for {analysis_id}",
        project_id=plan.get("project_id"),
        output_summary=str(exec_result.data)[:500] if exec_result.success else str(exec_result.data)[:200],
    )

    return {
        "success": exec_result.success,
        "analysis_id": analysis_id,
        "run_id": run_id,
        "data": _json_safe(exec_result.data),
        "warnings": exec_result.warnings,
    }


@app.get("/api/analysis-plan/{analysis_id}/decision-trace")
async def get_decision_trace(analysis_id: str):
    """Retrieve the decision trace for an analysis."""
    found = await plan_persistence.find_analysis(analysis_id)
    if found:
        plan, a = found
        return {
            "analysis_id": analysis_id,
            "analysis_name": a.get("analysis_name", ""),
            "method": a.get("method", ""),
            "lifecycle": a.get("lifecycle", ""),
            "validation": a.get("validation", ""),
            "approval": a.get("approval", ""),
            "prespecified": a.get("prespecified", "unclear"),
        }
    raise HTTPException(404, f"Analysis {analysis_id} not found")


@app.get("/api/methods")
async def list_methods():
    """List all registered statistical methods."""
    from app.statistics.registry import list_methods as _list, REGISTRY
    methods = []
    for mid in _list():
        entry = REGISTRY[mid]
        methods.append({
            "method_id": entry.method_id,
            "name": entry.name,
            "description": entry.description,
            "outcome_types": entry.outcome_types,
            "predictor_types": entry.predictor_types,
            "min_observations": entry.min_observations,
            "effect_measure": entry.effect_measure,
        })
    return {"methods": methods, "count": len(methods)}


# ---------------------------------------------------------------------------
# Data Quality Assessment (Phase 2D)
# ---------------------------------------------------------------------------

@app.post("/api/data-quality/{dataset_id}/assess")
async def assess_data_quality(dataset_id: str, request: Request):
    """Run comprehensive data quality assessment on a dataset."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}

    async with get_session() as session:
        result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    from app.agents.data_quality import DataQualityAgent
    agent = DataQualityAgent()
    outcome = body.get("outcome")
    predictors = body.get("predictors", [])

    result = await agent.run({
        "file_path": dataset.stored_path,
        "dataset_id": dataset_id,
        "outcome": outcome,
        "predictors": predictors,
    })

    if not result.success:
        raise HTTPException(500, f"Assessment failed: {result.data.get('error', 'unknown')}")

    # Persist profile
    from app.data.data_quality_store import save_quality_profile
    from app.data.dataset_identity import compute_dataset_identity
    from app.statistics.data_quality import DataQualityProfile
    profile = DataQualityProfile(**result.data.get("profile", {}))
    identity = compute_dataset_identity(dataset.stored_path)
    profile_id = await save_quality_profile(
        profile, dataset_id,
        dataset_content_hash=identity["content_hash"],
        dataset_schema_hash=identity["schema_hash"],
    )

    await log_audit(
        action="data_quality_assessed",
        agent="data_quality",
        decision=f"Assessed dataset {dataset_id}: {result.data.get('n_issues', 0)} issues found",
        project_id=dataset.project_id,
    )

    return {
        "profile_id": profile_id,
        "dataset_id": dataset_id,
        "n_issues": result.data.get("n_issues", 0),
        "n_critical": result.data.get("n_critical", 0),
        "n_warning": result.data.get("n_warning", 0),
        "missingness": result.data.get("missingness"),
        "missing_data_strategies": result.data.get("missing_data_strategies", []),
        "outlier_summary": result.data.get("outlier_summary", []),
        "outlier_strategies": result.data.get("outlier_strategies", []),
        "sensitivity_variants": result.data.get("sensitivity_variants", []),
        "transformation_candidates": result.data.get("transformation_candidates", []),
    }


@app.get("/api/data-quality/{dataset_id}")
async def get_data_quality(dataset_id: str):
    """Get the most recent data quality profile for a dataset."""
    from app.data.data_quality_store import get_quality_profile_by_dataset
    profile = await get_quality_profile_by_dataset(dataset_id)
    if not profile:
        raise HTTPException(404, "No data quality profile found for this dataset")
    return profile


@app.get("/api/data-quality/{dataset_id}/missingness")
async def get_missingness(dataset_id: str):
    """Get missingness analysis for a dataset."""
    from app.data.data_quality_store import get_quality_profile_by_dataset
    profile = await get_quality_profile_by_dataset(dataset_id)
    if not profile:
        raise HTTPException(404, "No data quality profile found")
    return {
        "dataset_id": dataset_id,
        "missingness": profile.get("profile", {}).get("missingness"),
        "missing_data_strategies": profile.get("profile", {}).get("missing_data_strategies", []),
    }


@app.get("/api/data-quality/{dataset_id}/outliers")
async def get_outliers(dataset_id: str):
    """Get outlier analysis for a dataset."""
    from app.data.data_quality_store import get_quality_profile_by_dataset
    profile = await get_quality_profile_by_dataset(dataset_id)
    if not profile:
        raise HTTPException(404, "No data quality profile found")
    return {
        "dataset_id": dataset_id,
        "outlier_summary": profile.get("profile", {}).get("outlier_summary", []),
        "outlier_strategies": profile.get("profile", {}).get("outlier_strategies", []),
        "sensitivity_variants": profile.get("profile", {}).get("sensitivity_variants", []),
    }


# ---------------------------------------------------------------------------
# Transformation Proposals (Phase 2D)
# ---------------------------------------------------------------------------

@app.post("/api/transformations/propose")
async def propose_transformation(request: Request):
    """Propose a data transformation requiring researcher approval."""
    body = await request.json()
    dataset_id = body.get("dataset_id")
    operation = body.get("operation")
    variable = body.get("variable")
    parameters = body.get("parameters", {})
    reason = body.get("reason")

    if not dataset_id or not operation or not variable:
        raise HTTPException(400, "dataset_id, operation, and variable are required")

    # Compute dataset identity for binding
    async with get_session() as session:
        result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    from app.data.data_quality_store import create_transformation_proposal
    from app.data.dataset_identity import compute_dataset_identity
    from app.statistics.data_quality import HandlingStrategy, HandlingAction

    identity = compute_dataset_identity(dataset.stored_path)

    strategy = HandlingStrategy(
        strategy_id=f"manual_{operation}_{variable}",
        action=HandlingAction.IMPUTE if "impute" in operation else HandlingAction.TRANSFORM,
        target_variable=variable,
        details=reason or f"Proposed {operation} on {variable}",
        requires_approval=True,
    )

    proposal_id = await create_transformation_proposal(
        dataset_id=dataset_id,
        source_content_hash=identity["content_hash"],
        source_schema_hash=identity["schema_hash"],
        operation=operation,
        variable=variable,
        strategy=strategy,
        parameters=parameters,
        reason=reason,
    )

    await log_audit(
        action="transformation_proposed",
        agent="researcher",
        decision=f"Proposed {operation} on {variable} for dataset {dataset_id}",
        project_id=dataset.project_id,
    )

    return {"proposal_id": proposal_id, "status": "detected"}


@app.post("/api/transformations/{proposal_id}/approve")
async def approve_transformation(proposal_id: str):
    """Approve a transformation proposal."""
    from app.data.data_quality_store import approve_transformation as _approve
    result = await _approve(proposal_id)
    if not result:
        raise HTTPException(404, "Proposal not found")
    if "error" in result:
        raise HTTPException(400, result["error"])

    await log_audit(
        action="transformation_approved",
        agent="researcher",
        decision=f"Approved transformation proposal {proposal_id}",
        researcher_decision="approve",
    )

    return {"proposal_id": proposal_id, "status": "approved"}


@app.post("/api/transformations/{proposal_id}/reject")
async def reject_transformation(proposal_id: str, request: Request):
    """Reject a transformation proposal."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    reason = body.get("reason", "")

    from app.data.data_quality_store import reject_transformation as _reject
    result = await _reject(proposal_id, reason)
    if not result:
        raise HTTPException(404, "Proposal not found")
    if "error" in result:
        raise HTTPException(400, result["error"])

    await log_audit(
        action="transformation_rejected",
        agent="researcher",
        decision=f"Rejected transformation proposal {proposal_id}: {reason}",
        researcher_decision="reject",
    )

    return {"proposal_id": proposal_id, "status": "rejected", "reason": reason}


@app.post("/api/transformations/{proposal_id}/apply")
async def apply_transformation(proposal_id: str):
    """Apply an approved transformation to create a derived dataset."""
    # Get proposal to find dataset
    from app.data.data_quality_store import get_transformation_proposal, apply_transformation_proposal
    proposal = await get_transformation_proposal(proposal_id)
    if not proposal:
        raise HTTPException(404, "Proposal not found")

    dataset_id = proposal["dataset_id"]
    async with get_session() as session:
        result = await session.execute(sa.select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    result = await apply_transformation_proposal(proposal_id, dataset.stored_path)
    if not result:
        raise HTTPException(500, "Failed to apply transformation")
    if "error" in result:
        raise HTTPException(400, result["error"])

    await log_audit(
        action="transformation_applied",
        agent="data_quality_engine",
        decision=f"Applied transformation {proposal_id}, derived dataset created",
        project_id=dataset.project_id,
    )

    return result


@app.get("/api/transformations/{proposal_id}")
async def get_transformation(proposal_id: str):
    """Get transformation proposal details."""
    from app.data.data_quality_store import get_transformation_proposal
    proposal = await get_transformation_proposal(proposal_id)
    if not proposal:
        raise HTTPException(404, "Proposal not found")
    return proposal


@app.get("/api/datasets/{dataset_id}/lineage")
async def get_dataset_lineage(dataset_id: str):
    """Get transformation lineage for a dataset."""
    from app.data.data_quality_store import get_dataset_lineage
    lineage = await get_dataset_lineage(dataset_id)
    return {"dataset_id": dataset_id, "lineage": lineage}


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

@app.get("/api/audit")
async def audit_log(project_id: str | None = None, limit: int = 100):
    return await get_audit_log(project_id=project_id, limit=limit)


# ---------------------------------------------------------------------------
# Analysis Runs (Phase 2E)
# ---------------------------------------------------------------------------

@app.get("/api/runs")
async def list_runs(
    project_id: str | None = None,
    analysis_type: str | None = None,
    limit: int = 100,
):
    """List analysis runs with optional filters."""
    from app.data.run_store import list_analysis_runs
    return await list_analysis_runs(
        project_id=project_id,
        analysis_type=analysis_type,
        limit=limit,
    )


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get a specific analysis run."""
    from app.data.run_store import get_analysis_run
    run = await get_analysis_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/api/runs/fingerprint/{fingerprint}")
async def find_runs_by_fingerprint(fingerprint: str):
    """Find all runs with the same input fingerprint."""
    from app.data.run_store import find_runs_by_fingerprint
    runs = await find_runs_by_fingerprint(fingerprint)
    return {"fingerprint": fingerprint, "runs": runs}


@app.post("/api/runs/verify")
async def verify_reproducibility(request: Request):
    """Verify that an analysis is reproducible."""
    body = await request.json()
    specification = body.get("specification", {})
    dataset_content_hash = body.get("dataset_content_hash", "")
    dataset_schema_hash = body.get("dataset_schema_hash", "")
    current_result = body.get("current_result")

    from app.statistics.reproducibility import verify_reproducibility
    return await verify_reproducibility(
        specification, dataset_content_hash, dataset_schema_hash, current_result
    )


@app.post("/api/runs/compare")
async def compare_runs(request: Request):
    """Compare two analysis runs."""
    body = await request.json()
    run_id_a = body.get("run_id_a")
    run_id_b = body.get("run_id_b")

    if not run_id_a or not run_id_b:
        raise HTTPException(400, "run_id_a and run_id_b are required")

    from app.statistics.results_comparator import compare_runs
    try:
        return await compare_runs(run_id_a, run_id_b)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@app.post("/api/auth/register")
async def register(request: Request):
    """Register a new user with an invitation token."""
    body = await request.json()
    token = body.get("token")
    password = body.get("password")
    display_name = body.get("display_name")
    
    if not token or not password:
        raise HTTPException(400, "token and password are required")
    
    async with get_session() as session:
        user = await redeem_invitation(session, token, password, display_name)
        if not user:
            raise HTTPException(400, "Invalid or expired invitation token")
        
        access_token = create_access_token(user.id, user.role)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "role": user.role,
            },
        }


@app.post("/api/auth/login")
async def login(request: Request):
    """Login with email and password."""
    body = await request.json()
    email = body.get("email")
    password = body.get("password")
    
    if not email or not password:
        raise HTTPException(400, "email and password are required")
    
    async with get_session() as session:
        user = await authenticate_user(session, email, password)
        if not user:
            raise HTTPException(401, "Invalid email or password")
        
        access_token = create_access_token(user.id, user.role)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "role": user.role,
            },
        }


@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the current user's profile."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "display_name": current_user.display_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }


@app.post("/api/admin/invitations")
async def create_invitation_endpoint(request: Request, current_user: User = Depends(get_current_admin_user)):
    """Admin: Create a new invitation with optional quota and expiration settings."""
    body = await request.json()
    email = body.get("email")
    role = body.get("role", "researcher")
    expires_hours = body.get("expires_hours", 72)
    max_projects = body.get("max_projects", 5)
    max_analyses = body.get("max_analyses", 50)
    max_upload_mb = body.get("max_upload_mb", 100)
    access_expires_days = body.get("access_expires_days")  # None = no expiration
    
    if not email:
        raise HTTPException(400, "email is required")
    
    async with get_session() as session:
        invitation = await create_invitation(session, email, role, expires_hours)
        # Store quota settings in invitation (will be applied when redeemed)
        # For now, return them so they can be applied manually or via a separate endpoint
        return {
            "id": invitation.id,
            "email": invitation.email,
            "token": invitation.token,
            "role": invitation.role,
            "expires_at": invitation.expires_at.isoformat(),
            "quota": {
                "max_projects": max_projects,
                "max_analyses": max_analyses,
                "max_upload_mb": max_upload_mb,
                "access_expires_days": access_expires_days,
            },
        }


@app.get("/api/admin/users")
async def list_users_endpoint(current_user: User = Depends(get_current_admin_user)):
    """Admin: List all users."""
    async with get_session() as session:
        users = await list_users(session)
        return [
            {
                "id": u.id,
                "email": u.email,
                "display_name": u.display_name,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ]


@app.post("/api/admin/users/{user_id}/deactivate")
async def deactivate_user_endpoint(user_id: str, current_user: User = Depends(get_current_admin_user)):
    """Admin: Deactivate a user."""
    async with get_session() as session:
        success = await deactivate_user(session, user_id)
        if not success:
            raise HTTPException(404, "User not found")
        return {"status": "deactivated"}


@app.post("/api/admin/users/{user_id}/activate")
async def activate_user_endpoint(user_id: str, current_user: User = Depends(get_current_admin_user)):
    """Admin: Activate a user."""
    async with get_session() as session:
        success = await activate_user(session, user_id)
        if not success:
            raise HTTPException(404, "User not found")
        return {"status": "activated"}


@app.put("/api/admin/users/{user_id}/quota")
async def update_user_quota(user_id: str, request: Request, current_user: User = Depends(get_current_admin_user)):
    """Admin: Update user quota and access expiration."""
    from app.security.auth import get_user_by_id
    body = await request.json()
    
    async with get_session() as session:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(404, "User not found")
        
        if "max_projects" in body:
            user.max_projects = body["max_projects"]
        if "max_analyses" in body:
            user.max_analyses = body["max_analyses"]
        if "max_upload_mb" in body:
            user.max_upload_mb = body["max_upload_mb"]
        if "access_expires_days" in body:
            if body["access_expires_days"] is None:
                user.access_expires_at = None
            else:
                from datetime import timedelta
                user.access_expires_at = datetime.now(timezone.utc) + timedelta(days=body["access_expires_days"])
        
        await session.commit()
        await session.refresh(user)
        
        return {
            "id": user.id,
            "email": user.email,
            "max_projects": user.max_projects,
            "max_analyses": user.max_analyses,
            "max_upload_mb": user.max_upload_mb,
            "access_expires_at": user.access_expires_at.isoformat() if user.access_expires_at else None,
        }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    from app.core.ollama_adapter import get_model_provider
    provider = get_model_provider()
    provider_ok = await provider.is_available()
    return {
        "status": "ok",
        "version": "0.2.0",
        "model_provider": provider.provider_name,
        "provider_available": provider_ok,
        "provider_url": settings.OLLAMA_BASE_URL,
        "provider_model": settings.OLLAMA_MODEL,
    }
