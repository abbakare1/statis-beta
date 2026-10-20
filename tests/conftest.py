"""Pytest configuration for STATIS tests."""

import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path for all tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Test isolation -------------------------------------------------------
# Tests must never touch the real application database or write artifacts
# into the project's data/outputs directories. Redirect all runtime paths to
# a per-session temp directory BEFORE config.settings is first imported.
_TEST_RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="statis_pytest_"))
os.environ["STATIS_DATA_DIR"] = str(_TEST_RUNTIME_DIR / "data")
os.environ["STATIS_UPLOAD_DIR"] = str(_TEST_RUNTIME_DIR / "data" / "uploads")
os.environ["STATIS_WORKING_DIR"] = str(_TEST_RUNTIME_DIR / "data" / "working")
os.environ["STATIS_OUTPUT_DIR"] = str(_TEST_RUNTIME_DIR / "outputs")
os.environ["STATIS_DB_PATH"] = str(_TEST_RUNTIME_DIR / "database" / "statis.db")
os.environ["STATIS_LOG_DIR"] = str(_TEST_RUNTIME_DIR / "logs")
os.environ["STATIS_MEMORY_DIR"] = str(_TEST_RUNTIME_DIR / "memory")

# Canonical test data paths (READ-ONLY — never modify these directly)
CANONICAL_DATA_DIR = ROOT / "data" / "uploads"
TEST1_PATH = CANONICAL_DATA_DIR / "228683e7-01c9-4050-afde-4d17cdb99e3f_STATIS_Test1_Continuous.csv"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    """Initialize the database once for the entire test session."""
    from app.core.database import init_db, close_db
    import asyncio

    async def _setup():
        await init_db()

    async def _teardown():
        await close_db()

    asyncio.run(_setup())
    yield
    asyncio.run(_teardown())


@pytest.fixture(scope="session", autouse=True)
def _canonical_fixture_hash_before():
    """Record canonical fixture hashes at session start."""
    hashes = {}
    if TEST1_PATH.exists():
        hashes["test1_content"] = _hash_file(TEST1_PATH)
    return hashes


@pytest.fixture(scope="session", autouse=True)
def _verify_canonical_fixture_hash_after(_canonical_fixture_hash_before):
    """Verify canonical fixture hashes at session end — no test may have modified them."""
    yield
    errors = []
    if "test1_content" in _canonical_fixture_hash_before and TEST1_PATH.exists():
        current = _hash_file(TEST1_PATH)
        if current != _canonical_fixture_hash_before["test1_content"]:
            errors.append(
                f"TEST1_PATH was modified during test session! "
                f"Before: {_canonical_fixture_hash_before['test1_content'][:16]}..., "
                f"After: {current[:16]}..."
            )
    if errors:
        pytest.fail("\n".join(errors))


@pytest.fixture
def isolated_copy(tmp_path):
    """Factory fixture: create an isolated copy of a file in a temp directory.

    Usage:
        def test_something(isolated_copy):
            copy_path = isolated_copy(TEST1_PATH)
            # work with copy_path — original is untouched
    """
    copies = []

    def _copy(source: Path) -> Path:
        dest = tmp_path / source.name
        shutil.copy2(source, dest)
        copies.append(dest)
        return dest

    yield _copy

    # Cleanup happens automatically via tmp_path


@pytest.fixture
def canonical_test1_path():
    """Provide the canonical Test 1 path (READ-ONLY)."""
    return TEST1_PATH
