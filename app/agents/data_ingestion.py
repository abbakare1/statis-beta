"""Data Ingestion Agent — load CSV/XLSX/TSV, detect types, create working copy."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from app.agents.base import BaseAgent, AgentResult
from app.data.hashing import file_hash
from config.settings import get_settings


class DataIngestionAgent(BaseAgent):
    name = "data_ingestion"
    requires_llm = False
    description = "Load dataset, detect types, create versioned working copy."

    async def _execute(self, context: dict) -> AgentResult:
        project_id = context.get("project_id")
        file_path = Path(context["file_path"])
        original_filename = context.get("original_filename", file_path.name)

        settings = get_settings()

        if not file_path.exists():
            return AgentResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
                warnings=[f"File not found: {file_path}"],
            )

        suffix = file_path.suffix.lower()
        if suffix not in settings.ALLOWED_EXTENSIONS:
            return AgentResult(
                success=False,
                data={"error": f"Unsupported file type: {suffix}"},
                warnings=[f"Unsupported file type: {suffix}. Allowed: {settings.ALLOWED_EXTENSIONS}"],
            )

        # Compute content hash
        content_hash = file_hash(file_path)

        # Determine working directory
        working_dir = settings.WORKING_DIR / project_id if project_id else settings.WORKING_DIR
        working_dir.mkdir(parents=True, exist_ok=True)

        # Create versioned working copy
        version = context.get("version", 1)
        working_filename = f"v{version}_{file_path.name}"
        working_path = working_dir / working_filename
        shutil.copy2(file_path, working_path)

        # Load the dataset
        try:
            if suffix == ".csv":
                df = pd.read_csv(working_path)
            elif suffix == ".tsv":
                df = pd.read_csv(working_path, sep="\t")
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(working_path)
            elif suffix == ".parquet":
                df = pd.read_parquet(working_path)
            else:
                return AgentResult(
                    success=False,
                    data={"error": f"Cannot read file type: {suffix}"},
                )
        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": f"Failed to parse file: {e}"},
                warnings=[f"Parse error: {e}"],
            )

        # Build column metadata
        columns_meta = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            columns_meta.append({
                "name": col,
                "dtype": dtype_str,
                "n_unique": int(df[col].nunique()),
                "n_missing": int(df[col].isna().sum()),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_categorical": isinstance(df[col].dtype, pd.CategoricalDtype) or (
                    df[col].dtype == object and df[col].nunique() < 50
                ),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(df[col]),
            })

        # Basic shape info
        shape = {"rows": df.shape[0], "columns": df.shape[1]}
        memory_bytes = int(df.memory_usage(deep=True).sum())

        return AgentResult(
            success=True,
            data={
                "dataset_id": context.get("dataset_id"),
                "file_path": str(working_path),
                "original_filename": original_filename,
                "content_hash": content_hash,
                "version": version,
                "shape": shape,
                "columns": columns_meta,
                "memory_bytes": memory_bytes,
                "dtypes_summary": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
                "has_index_candidate": _detect_index_candidate(df),
            },
            warnings=[],
            confidence="high",
        )


def _detect_index_candidate(df: pd.DataFrame) -> list[str]:
    """Detect columns that could serve as row identifiers."""
    candidates = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].isna().sum() == 0:
            candidates.append(col)
    return candidates
