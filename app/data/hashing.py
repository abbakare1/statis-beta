"""Data hashing and versioning utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute content hash of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
