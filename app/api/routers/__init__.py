"""API Routers — modular route registration for the STATIS FastAPI app."""

from app.api.routers.outputs import (
    figure_router,
    table_router,
    report_router,
    repro_router,
)

__all__ = [
    "figure_router",
    "table_router",
    "report_router",
    "repro_router",
]
