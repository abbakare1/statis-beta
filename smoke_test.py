"""Smoke test — verify core components work end-to-end."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
s = get_settings()
print(f"BASE_DIR: {s.BASE_DIR}")
print(f"DB_PATH: {s.DB_PATH}")
print(f"HOST: {s.HOST}:{s.PORT}")
print(f"OLLAMA: {s.OLLAMA_BASE_URL} ({s.OLLAMA_MODEL})")

from app.core.database import init_db
import asyncio
asyncio.run(init_db())
print("Database initialized OK")

from app.statistics.engine import descriptive_statistics, independent_t_test
import pandas as pd
import numpy as np

df = pd.DataFrame({"x": np.random.normal(0, 1, 100), "y": np.random.normal(5, 2, 100)})
test_path = ROOT / "data" / "test_smoke.csv"
df.to_csv(test_path, index=False)
result = descriptive_statistics(test_path)
print(f"Descriptive stats: {len(result['columns'])} columns, {result['n_rows']} rows")

df2 = pd.DataFrame({"val": np.random.normal(0, 1, 200), "grp": np.random.choice(["A", "B"], 200)})
test_path2 = ROOT / "data" / "test_ttest.csv"
df2.to_csv(test_path2, index=False)
t_result = independent_t_test(test_path2, outcome="val", group="grp")
print(f"t-test: t={t_result['t_statistic']:.4f}, p={t_result['p_value']:.4f}")

# Verify agents import
from app.agents.data_ingestion import DataIngestionAgent
from app.agents.data_dictionary import DataDictionaryAgent
from app.agents.data_quality import DataQualityAgent
from app.agents.research_question import ResearchQuestionAgent
from app.agents.study_design import StudyDesignAgent
from app.agents.power_sample_size import PowerSampleSizeAgent
from app.agents.analysis_planner import AnalysisPlannerAgent
from app.agents.statistical_executor import StatisticalExecutorAgent
from app.orchestration.orchestrator import Orchestrator
print("All agents import OK")

# Verify API app
from app.api.main import app
print(f"FastAPI app: {app.title} v{app.version}")
print()
print("ALL SMOKE TESTS PASSED")
