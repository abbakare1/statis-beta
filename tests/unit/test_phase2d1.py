"""Phase 2D.1 Tests — Hardened data-quality persistence, proposal lifecycle, and integrity.

Covers:
- DB-backed quality profiles (not in-memory dicts)
- DB-backed proposals with full lifecycle state machine
- Dataset-binding integrity (stale proposal detection)
- Transactional application (rollback on failure)
- Idempotency (repeated apply returns existing result)
- Anti-p-hacking verdict stored on proposals
- Audit trail completeness
"""

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import pytest

from app.statistics.data_quality import (
    DataQualityProfile,
    HandlingStrategy,
    HandlingAction,
    TransformationCandidate,
    TransformationType,
    TransformationSafety,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.normal(50, 15, 100),
        "bmi": np.random.lognormal(3, 0.5, 100),
        "bp": np.random.normal(120, 20, 100),
        "group": np.random.choice(["A", "B", "C"], 100),
        "outcome": np.random.choice([0, 1], 100),
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_with_missing(tmp_path):
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.normal(50, 15, 100),
        "bmi": [np.nan if i % 10 == 0 else np.random.lognormal(3, 0.5) for i in range(100)],
        "bp": np.random.normal(120, 20, 100),
        "group": np.random.choice(["A", "B", "C"], 100),
        "outcome": np.random.choice([0, 1], 100),
    })
    path = tmp_path / "test_missing.csv"
    df.to_csv(path, index=False)
    return path


def _uid() -> str:
    return f"ds_{uuid.uuid4().hex[:12]}"


def _make_strategy(variable: str = "bmi", operation: str = "log") -> HandlingStrategy:
    return HandlingStrategy(
        strategy_id=f"test_{operation}_{variable}",
        action=HandlingAction.IMPUTE if "impute" in operation else HandlingAction.TRANSFORM,
        target_variable=variable,
        details=f"Test {operation} on {variable}",
        requires_approval=True,
    )


async def _create_and_approve(sample_csv_path, *, ds_id=None, operation="log", variable="bmi"):
    """Helper: create proposal → proposed → review_required → approved. Returns (pid, identity)."""
    from app.data.data_quality_store import (
        create_transformation_proposal, transition_proposal,
    )
    from app.data.dataset_identity import compute_dataset_identity

    identity = compute_dataset_identity(str(sample_csv_path))
    if ds_id is None:
        ds_id = _uid()
    pid = await create_transformation_proposal(
        dataset_id=ds_id, operation=operation, variable=variable,
        source_content_hash=identity["content_hash"],
        source_schema_hash=identity["schema_hash"],
        strategy=_make_strategy(variable, operation),
    )
    await transition_proposal(pid, "proposed")
    await transition_proposal(pid, "review_required")
    await transition_proposal(pid, "approved", reviewer="test")
    return pid, identity


# ===========================================================================
# QUALITY PROFILE PERSISTENCE TESTS
# ===========================================================================

class TestQualityProfilePersistence:
    @pytest.mark.asyncio
    async def test_profile_saved_to_db(self, sample_csv):
        from app.data.data_quality_store import save_quality_profile, get_quality_profile
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        profile = DataQualityProfile(dataset_id=ds_id, row_count=100, column_count=6)
        profile_id = await save_quality_profile(
            profile, ds_id,
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"],
        )
        assert profile_id is not None
        retrieved = await get_quality_profile(profile_id)
        assert retrieved is not None
        assert retrieved["dataset_id"] == ds_id
        assert retrieved["profile"]["row_count"] == 100

    @pytest.mark.asyncio
    async def test_profile_by_dataset_returns_most_recent(self, sample_csv):
        from app.data.data_quality_store import save_quality_profile, get_quality_profile_by_dataset
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        p1 = DataQualityProfile(dataset_id=ds_id, row_count=50)
        p2 = DataQualityProfile(dataset_id=ds_id, row_count=200)
        await save_quality_profile(p1, ds_id,
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"])
        await save_quality_profile(p2, ds_id,
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"])

        result = await get_quality_profile_by_dataset(ds_id)
        assert result is not None
        assert result["profile"]["row_count"] == 200

    @pytest.mark.asyncio
    async def test_nonexistent_profile_returns_none(self):
        from app.data.data_quality_store import get_quality_profile
        result = await get_quality_profile("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_profile_contains_identity_hashes(self, sample_csv):
        from app.data.data_quality_store import save_quality_profile, get_quality_profile
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        profile = DataQualityProfile(dataset_id=ds_id)
        pid = await save_quality_profile(profile, ds_id,
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"])

        result = await get_quality_profile(pid)
        assert result["dataset_content_hash"] == identity["content_hash"]
        assert result["dataset_schema_hash"] == identity["schema_hash"]


# ===========================================================================
# PROPOSAL LIFECYCLE STATE MACHINE TESTS
# ===========================================================================

class TestProposalLifecycle:
    @pytest.mark.asyncio
    async def test_create_starts_in_detected(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, get_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        proposal = await get_transformation_proposal(pid)
        assert proposal["status"] == "detected"

    @pytest.mark.asyncio
    async def test_valid_lifecycle_transitions(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, transition_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        result = await transition_proposal(pid, "proposed")
        assert result["status"] == "proposed"

        result = await transition_proposal(pid, "review_required")
        assert result["status"] == "review_required"

        result = await transition_proposal(pid, "approved", reviewer="professor")
        assert result["status"] == "approved"
        assert result["reviewer"] == "professor"
        assert result["reviewed_at"] is not None

    @pytest.mark.asyncio
    async def test_reject_from_proposed(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, transition_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        await transition_proposal(pid, "proposed")
        result = await transition_proposal(pid, "rejected", reason="Not needed")
        assert result["status"] == "rejected"
        assert result["rejection_reason"] == "Not needed"

    @pytest.mark.asyncio
    async def test_reject_from_review_required(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, transition_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        await transition_proposal(pid, "proposed")
        await transition_proposal(pid, "review_required")
        result = await transition_proposal(pid, "rejected", reason="Does not meet criteria")
        assert result["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, transition_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        with pytest.raises(ValueError, match="Invalid state transition"):
            await transition_proposal(pid, "approved")

    @pytest.mark.asyncio
    async def test_rejected_can_re_propose(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, transition_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        await transition_proposal(pid, "proposed")
        await transition_proposal(pid, "rejected", reason="No")
        result = await transition_proposal(pid, "proposed")
        assert result["status"] == "proposed"

    @pytest.mark.asyncio
    async def test_approved_can_be_marked_stale(self, sample_csv):
        from app.data.data_quality_store import mark_proposal_stale

        pid, _ = await _create_and_approve(sample_csv)
        result = await mark_proposal_stale(pid, reason="Dataset changed")
        assert result["status"] == "stale"

    @pytest.mark.asyncio
    async def test_audit_events_recorded(self, sample_csv):
        from app.data.data_quality_store import get_transformation_proposal

        pid, _ = await _create_and_approve(sample_csv)
        proposal = await get_transformation_proposal(pid)
        events = proposal["audit_events"]
        assert len(events) >= 3
        assert events[0]["from"] == "none"
        assert events[0]["to"] == "detected"
        assert events[-1]["from"] == "review_required"
        assert events[-1]["to"] == "approved"


# ===========================================================================
# DATASET-BINDING INTEGRITY TESTS
# ===========================================================================

class TestDatasetBindingIntegrity:
    @pytest.mark.asyncio
    async def test_proposal_binds_to_dataset_identity(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, get_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        proposal = await get_transformation_proposal(pid)
        assert proposal["source_content_hash"] == identity["content_hash"]
        assert proposal["source_schema_hash"] == identity["schema_hash"]

    @pytest.mark.asyncio
    async def test_integrity_check_passes_for_unchanged_dataset(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, verify_proposal_integrity,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        is_valid, reason = await verify_proposal_integrity(pid, str(sample_csv))
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_integrity_check_fails_for_changed_dataset(self, sample_csv, tmp_path):
        from app.data.data_quality_store import (
            create_transformation_proposal, verify_proposal_integrity,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        df = pd.read_csv(sample_csv)
        df.loc[0, "bmi"] = 999.0
        modified_path = tmp_path / "modified.csv"
        df.to_csv(modified_path, index=False)

        is_valid, reason = await verify_proposal_integrity(pid, str(modified_path))
        assert is_valid is False
        assert "changed" in reason.lower()

    @pytest.mark.asyncio
    async def test_apply_fails_on_stale_dataset(self, sample_csv, tmp_path):
        from app.data.data_quality_store import apply_transformation_proposal

        pid, _ = await _create_and_approve(sample_csv)

        df = pd.read_csv(sample_csv)
        df.loc[0, "bmi"] = 999.0
        modified_path = tmp_path / "modified.csv"
        df.to_csv(modified_path, index=False)

        result = await apply_transformation_proposal(pid, str(modified_path))
        assert "error" in result
        assert "integrity" in result["error"].lower()


# ===========================================================================
# IDEMPOTENCY TESTS
# ===========================================================================

class TestIdempotency:
    @pytest.mark.asyncio
    async def test_same_fingerprint_returns_existing_proposal(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid1 = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        pid2 = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        assert pid1 == pid2

    @pytest.mark.asyncio
    async def test_different_fingerprint_returns_new_proposal(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid1 = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy("bmi", "log"),
        )
        pid2 = await create_transformation_proposal(
            dataset_id=ds_id, operation="sqrt", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy("bmi", "sqrt"),
        )
        assert pid1 != pid2

    @pytest.mark.asyncio
    async def test_approved_proposal_not_reused(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid1 = await _create_and_approve(sample_csv, ds_id=ds_id)

        pid2 = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        assert pid1 != pid2

    @pytest.mark.asyncio
    async def test_rejected_proposal_not_reused(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, transition_proposal,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid1 = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        await transition_proposal(pid1, "proposed")
        await transition_proposal(pid1, "rejected", reason="No")

        pid2 = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        assert pid1 != pid2

    @pytest.mark.asyncio
    async def test_apply_idempotent_returns_existing_result(self, sample_csv):
        from app.data.data_quality_store import apply_transformation_proposal

        pid, _ = await _create_and_approve(sample_csv)

        result1 = await apply_transformation_proposal(pid, str(sample_csv))
        assert result1["success"] is True
        assert result1["idempotent"] is False

        result2 = await apply_transformation_proposal(pid, str(sample_csv))
        assert result2["success"] is True
        assert result2["idempotent"] is True


# ===========================================================================
# TRANSACTIONAL APPLICATION TESTS
# ===========================================================================

class TestTransactionalApplication:
    @pytest.mark.asyncio
    async def test_full_lifecycle_apply_creates_derived(self, sample_csv):
        from app.data.data_quality_store import get_transformation_proposal, apply_transformation_proposal

        pid, _ = await _create_and_approve(sample_csv)

        result = await apply_transformation_proposal(pid, str(sample_csv))
        assert result["success"] is True
        assert "derived_path" in result
        assert Path(result["derived_path"]).exists()
        assert result["impact_summary"]["n_rows_before"] == 100
        assert result["impact_summary"]["n_rows_after"] == 100

        proposal = await get_transformation_proposal(pid)
        assert proposal["status"] == "applied"
        assert proposal["applied_at"] is not None
        assert proposal["derived_dataset_id"] is not None

    @pytest.mark.asyncio
    async def test_apply_records_anti_p_hacking_verdict(self, sample_csv):
        from app.data.data_quality_store import get_transformation_proposal, apply_transformation_proposal

        pid, _ = await _create_and_approve(sample_csv)

        result = await apply_transformation_proposal(pid, str(sample_csv))
        assert "anti_p_hacking_verdict" in result
        assert result["anti_p_hacking_verdict"]["overall_verdict"] in ("safe", "suspicious")

        proposal = await get_transformation_proposal(pid)
        assert proposal["anti_p_hacking_verdict"] is not None

    @pytest.mark.asyncio
    async def test_apply_fails_for_nonexistent_proposal(self):
        from app.data.data_quality_store import apply_transformation_proposal
        result = await apply_transformation_proposal("nonexistent", "/nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_apply_fails_for_wrong_status(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, apply_transformation_proposal,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        result = await apply_transformation_proposal(pid, str(sample_csv))
        assert "error" in result


# ===========================================================================
# PROPOSAL LISTING AND RETRIEVAL TESTS
# ===========================================================================

class TestProposalRetrieval:
    @pytest.mark.asyncio
    async def test_get_proposal(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, get_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        proposal = await get_transformation_proposal(pid)
        assert proposal is not None
        assert proposal["proposal_id"] == pid
        assert proposal["operation"] == "log"
        assert proposal["variable"] == "bmi"

    @pytest.mark.asyncio
    async def test_list_proposals_by_dataset(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal, list_transformation_proposals
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy("bmi", "log"),
        )
        await create_transformation_proposal(
            dataset_id=ds_id, operation="sqrt", variable="age",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy("age", "sqrt"),
        )

        ds_proposals = await list_transformation_proposals(ds_id)
        assert len(ds_proposals) == 2
        assert all(p["dataset_id"] == ds_id for p in ds_proposals)


# ===========================================================================
# LINEAGE COMPLETENESS TESTS
# ===========================================================================

class TestLineageCompleteness:
    @pytest.mark.asyncio
    async def test_apply_creates_transformation_record(self, sample_csv):
        from app.data.data_quality_store import (
            apply_transformation_proposal, get_dataset_lineage,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        content_prefix = identity["content_hash"][:16]

        pid, _ = await _create_and_approve(sample_csv)
        await apply_transformation_proposal(pid, str(sample_csv))

        lineage = await get_dataset_lineage(content_prefix)
        assert len(lineage) >= 1
        assert lineage[0]["operation"] == "log"


# ===========================================================================
# FULL END-TO-END SCENARIO TESTS
# ===========================================================================

class TestPhase2D1_EndToEnd:
    @pytest.mark.asyncio
    async def test_complete_data_quality_workflow(self, sample_csv_with_missing):
        """Full workflow: assess → detect → propose → approve → apply → verify."""
        from app.data.data_quality_store import (
            save_quality_profile, create_transformation_proposal,
            transition_proposal, apply_transformation_proposal,
            get_transformation_proposal, get_dataset_lineage,
        )
        from app.data.dataset_identity import compute_dataset_identity
        from app.statistics.missing_data import analyze_missingness, generate_missing_data_strategies
        from app.statistics.data_quality import HandlingAction

        identity = compute_dataset_identity(str(sample_csv_with_missing))
        ds_id = _uid()

        df = pd.read_csv(sample_csv_with_missing)
        missingness = analyze_missingness(df, outcome="outcome", predictors=["bmi", "bp"])
        strategies = generate_missing_data_strategies(missingness, outcome="outcome", predictors=["bmi"])
        profile = DataQualityProfile(
            dataset_id=ds_id, row_count=len(df), column_count=len(df.columns),
            missingness=missingness, missing_data_strategies=strategies,
        )
        profile_id = await save_quality_profile(
            profile, ds_id,
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"],
        )
        assert profile_id is not None

        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="mean_impute", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=HandlingStrategy(
                strategy_id="mean_impute_bmi", action=HandlingAction.IMPUTE,
                target_variable="bmi", details="Mean imputation for bmi",
                requires_approval=True,
            ),
        )

        await transition_proposal(pid, "proposed")
        await transition_proposal(pid, "review_required")
        await transition_proposal(pid, "approved", reviewer="dr_smith")

        result = await apply_transformation_proposal(pid, str(sample_csv_with_missing))
        assert result["success"] is True
        assert result["idempotent"] is False

        derived_path = Path(result["derived_path"])
        assert derived_path.exists()
        derived_df = pd.read_csv(derived_path)
        assert derived_df["bmi"].isna().sum() == 0

        proposal = await get_transformation_proposal(pid)
        assert proposal["status"] == "applied"
        assert proposal["impact_summary"]["n_rows_before"] == proposal["impact_summary"]["n_rows_after"]

        content_prefix = identity["content_hash"][:16]
        lineage = await get_dataset_lineage(content_prefix)
        assert len(lineage) >= 1

    @pytest.mark.asyncio
    async def test_stale_proposal_detection(self, sample_csv, tmp_path):
        from app.data.data_quality_store import apply_transformation_proposal

        pid, _ = await _create_and_approve(sample_csv)

        df = pd.read_csv(sample_csv)
        df.loc[0, "bmi"] = 999.0
        changed_path = tmp_path / "changed.csv"
        df.to_csv(changed_path, index=False)

        result = await apply_transformation_proposal(pid, str(changed_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejection_blocks_application(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, transition_proposal,
            apply_transformation_proposal,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        pid = await create_transformation_proposal(
            dataset_id=ds_id, operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=_make_strategy(),
        )
        await transition_proposal(pid, "proposed")
        await transition_proposal(pid, "rejected", reason="Methodological concern")

        result = await apply_transformation_proposal(pid, str(sample_csv))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_multiple_proposals_for_same_dataset(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, list_transformation_proposals,
        )
        from app.data.dataset_identity import compute_dataset_identity

        identity = compute_dataset_identity(str(sample_csv))
        ds_id = _uid()
        ops = [("log", "bmi"), ("sqrt", "age"), ("mean_impute", "bp")]
        for op, var in ops:
            await create_transformation_proposal(
                dataset_id=ds_id, operation=op, variable=var,
                source_content_hash=identity["content_hash"],
                source_schema_hash=identity["schema_hash"],
                strategy=_make_strategy(var, op),
            )

        proposals = await list_transformation_proposals(ds_id)
        assert len(proposals) == 3
