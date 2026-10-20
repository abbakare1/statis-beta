"""Tables Module — Phase 2G.

Deterministic table rendering for publication outputs. Consumes persisted
StatisticalResult objects; never recomputes statistics.
"""

from app.tables.renderer import TableRenderer

__all__ = ["TableRenderer"]
