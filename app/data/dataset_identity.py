"""Dataset Identity — canonical hashing for tabular data.

Provides deterministic, reproducible SHA-256 hashing for tabular datasets
that is invariant to row order, column order, and serialization format.

Canonical strategy:
1. Convert all columns to string representation
2. Sort columns alphabetically
3. Sort rows lexicographically by all column values
4. Serialize to canonical CSV (no index, sorted columns, sorted rows)
5. SHA-256 hash the canonical bytes

This ensures:
- Same data → same hash regardless of row/column order
- Same data → same hash regardless of pandas DataFrame internal ordering
- Different data → different hash
- Schema changes → different schema_hash
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def compute_dataset_identity(
    file_path: Path | str,
    schema_hash: str | None = None,
) -> dict[str, Any]:
    """Compute full dataset identity from a file.

    Returns dict with:
    - content_hash: SHA-256 of canonical data representation
    - schema_hash: SHA-256 of column names + dtypes
    - row_count: number of rows
    - column_count: number of columns
    - column_names: sorted list of column names
    """
    df = _load(file_path)
    content_hash = canonical_content_hash(df)
    computed_schema_hash = canonical_schema_hash(df)

    return {
        "content_hash": content_hash,
        "schema_hash": computed_schema_hash,
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_names": sorted(df.columns.tolist()),
    }


def canonical_content_hash(df: pd.DataFrame) -> str:
    """Compute canonical SHA-256 hash of DataFrame content.

    Invariant to row order and column order.
    """
    canonical_bytes = _canonical_serialize(df)
    return hashlib.sha256(canonical_bytes).hexdigest()


def canonical_schema_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of DataFrame schema (column names + dtypes).

    Column order does not affect this hash.
    """
    schema = {
        "columns": sorted(df.columns.tolist()),
        "dtypes": {col: str(df[col].dtype) for col in sorted(df.columns.tolist())},
    }
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_dataset_identity(
    file_path: Path | str,
    expected_content_hash: str,
    expected_schema_hash: str | None = None,
    expected_row_count: int | None = None,
) -> dict[str, Any]:
    """Verify that a dataset matches its expected identity.

    Returns dict with:
    - valid: bool
    - content_match: bool
    - schema_match: bool (if schema_hash provided)
    - row_count_match: bool (if row_count provided)
    - expected: dict of expected values
    - observed: dict of observed values
    - mismatches: list of mismatch descriptions
    """
    identity = compute_dataset_identity(file_path)
    mismatches = []

    content_match = identity["content_hash"] == expected_content_hash
    if not content_match:
        mismatches.append(
            f"CONTENT_HASH_MISMATCH: expected={expected_content_hash[:16]}..., "
            f"observed={identity['content_hash'][:16]}..."
        )

    schema_match = True
    if expected_schema_hash is not None:
        schema_match = identity["schema_hash"] == expected_schema_hash
        if not schema_match:
            mismatches.append(
                f"SCHEMA_HASH_MISMATCH: expected={expected_schema_hash[:16]}..., "
                f"observed={identity['schema_hash'][:16]}..."
            )

    row_count_match = True
    if expected_row_count is not None:
        row_count_match = identity["row_count"] == expected_row_count
        if not row_count_match:
            mismatches.append(
                f"ROW_COUNT_MISMATCH: expected={expected_row_count}, "
                f"observed={identity['row_count']}"
            )

    return {
        "valid": content_match and schema_match and row_count_match,
        "content_match": content_match,
        "schema_match": schema_match,
        "row_count_match": row_count_match,
        "expected": {
            "content_hash": expected_content_hash,
            "schema_hash": expected_schema_hash,
            "row_count": expected_row_count,
        },
        "observed": {
            "content_hash": identity["content_hash"],
            "schema_hash": identity["schema_hash"],
            "row_count": identity["row_count"],
            "column_count": identity["column_count"],
        },
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(file_path: Path | str) -> pd.DataFrame:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    elif suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    elif suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def _canonical_serialize(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to canonical bytes.

    Steps:
    1. Convert all columns to string
    2. Sort columns alphabetically
    3. Sort rows lexicographically
    4. Serialize to CSV bytes (no index)
    """
    # Convert to string representation for consistent handling
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str)

    # Sort columns alphabetically
    df_str = df_str[sorted(df_str.columns.tolist())]

    # Sort rows lexicographically by all columns
    df_str = df_str.sort_values(by=list(df_str.columns)).reset_index(drop=True)

    # Serialize to canonical CSV bytes
    return df_str.to_csv(index=False).encode("utf-8")
