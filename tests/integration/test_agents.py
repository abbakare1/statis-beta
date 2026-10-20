"""Integration tests for the data agents."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import numpy as np
import pandas as pd
import pytest


# Initialize DB before any agent runs
@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    """Create test database tables."""
    from app.core.database import init_db, close_db
    asyncio.run(init_db())
    yield
    asyncio.run(close_db())


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def sample_csv(tmp_path):
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "patient_id": range(1, n + 1),
        "age": np.random.randint(18, 85, n),
        "sex": np.random.choice(["M", "F"], n),
        "bmi": np.random.normal(25, 5, n),
        "bp_systolic": np.random.normal(130, 15, n),
        "outcome": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "treatment": np.random.choice(["Drug", "Placebo"], n),
    })
    df.loc[5, "bmi"] = np.nan
    df.loc[10, "bp_systolic"] = np.nan
    df.loc[15, "age"] = np.nan
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def xlsx_file(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(1, 51),
        "score": np.random.normal(50, 10, 50),
        "group": np.random.choice(["A", "B", "C"], 50),
    })
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)
    return path


class TestDataIngestion:
    def test_csv_ingestion(self, sample_csv):
        from app.agents.data_ingestion import DataIngestionAgent
        agent = DataIngestionAgent()
        result = run_async(agent.run({
            "file_path": str(sample_csv),
            "project_id": "test-project",
            "version": 1,
        }))
        assert result.success
        assert result.data["shape"]["rows"] == 100
        assert result.data["shape"]["columns"] == 7
        assert result.data["content_hash"]
        assert len(result.data["columns"]) == 7

    def test_xlsx_ingestion(self, xlsx_file):
        from app.agents.data_ingestion import DataIngestionAgent
        agent = DataIngestionAgent()
        result = run_async(agent.run({
            "file_path": str(xlsx_file),
            "project_id": "test-project",
            "version": 1,
        }))
        assert result.success
        assert result.data["shape"]["rows"] == 50

    def test_nonexistent_file(self):
        from app.agents.data_ingestion import DataIngestionAgent
        agent = DataIngestionAgent()
        result = run_async(agent.run({"file_path": "/nonexistent/file.csv", "project_id": "test"}))
        assert not result.success


class TestDataDictionary:
    def test_variable_detection(self, sample_csv):
        from app.agents.data_dictionary import DataDictionaryAgent
        agent = DataDictionaryAgent()
        result = run_async(agent.run({
            "file_path": str(sample_csv),
            "project_id": "test-project",
        }))
        assert result.success
        assert result.data["n_variables"] == 7
        id_vars = result.data["identifier_candidates"]
        assert "patient_id" in id_vars

    def test_outcome_detection(self, sample_csv):
        from app.agents.data_dictionary import DataDictionaryAgent
        agent = DataDictionaryAgent()
        result = run_async(agent.run({
            "file_path": str(sample_csv),
            "project_id": "test-project",
        }))
        assert "outcome" in result.data["outcome_candidates"]


class TestDataQuality:
    def test_missing_detection(self, sample_csv):
        from app.agents.data_quality import DataQualityAgent
        agent = DataQualityAgent()
        result = run_async(agent.run({
            "file_path": str(sample_csv),
            "project_id": "test-project",
        }))
        assert result.success
        assert result.data["total_missing_cells"] > 0
        assert len(result.data["issues"]) > 0
        missing_issues = [i for i in result.data["issues"] if i["issue_type"] == "missing"]
        assert len(missing_issues) > 0

    def test_no_duplicates(self, sample_csv):
        from app.agents.data_quality import DataQualityAgent
        agent = DataQualityAgent()
        result = run_async(agent.run({"file_path": str(sample_csv), "project_id": "test"}))
        assert result.success
        assert result.data["duplicate_rows"] == 0
