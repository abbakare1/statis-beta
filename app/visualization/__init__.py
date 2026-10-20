"""Visualization Module — Phase 2G.

Deterministic figure rendering using matplotlib. Consumes persisted
StatisticalResult objects; never recomputes statistics.
"""

from app.visualization.renderer import FigureRenderer

__all__ = ["FigureRenderer"]
