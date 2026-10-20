"""Output Lifecycle — state machine for publication output transitions.

Phase 2H: Enforces valid lifecycle transitions for figures, tables, reports,
and reproducibility packages. Every transition is validated; invalid
transitions raise OutputLifecycleError.
"""

from __future__ import annotations

from app.outputs.specification import OutputLifecycle


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class OutputLifecycleError(Exception):
    """Raised when an invalid lifecycle transition is attempted."""

    def __init__(self, current: str, target: str, reason: str = ""):
        self.current = current
        self.target = target
        self.reason = reason
        msg = f"Invalid lifecycle transition: {current!r} -> {target!r}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Allowed transitions
# ---------------------------------------------------------------------------

# Maps current state → set of allowed next states.
# Failure states are terminal except VALIDATION_FAILED which allows
# re-validation (→ VALIDATING).
ALLOWED_TRANSITIONS: dict[OutputLifecycle, set[OutputLifecycle]] = {
    OutputLifecycle.DRAFT: {
        OutputLifecycle.VALIDATING,
    },
    OutputLifecycle.VALIDATING: {
        OutputLifecycle.VALIDATED,
        OutputLifecycle.VALIDATION_FAILED,
    },
    OutputLifecycle.VALIDATED: {
        OutputLifecycle.RENDERING,
    },
    OutputLifecycle.RENDERING: {
        OutputLifecycle.GENERATED,
        OutputLifecycle.RENDERING_FAILED,
    },
    OutputLifecycle.GENERATED: {
        OutputLifecycle.REVIEW_REQUIRED,
        OutputLifecycle.RENDERING,  # re-render allowed from generated
    },
    OutputLifecycle.REVIEW_REQUIRED: {
        OutputLifecycle.APPROVED,
        OutputLifecycle.RENDERING,  # re-render before approval
    },
    OutputLifecycle.APPROVED: {
        OutputLifecycle.ARCHIVED,
    },
    OutputLifecycle.ARCHIVED: set(),  # terminal
    OutputLifecycle.VALIDATION_FAILED: {
        OutputLifecycle.VALIDATING,  # revalidation allowed
    },
    OutputLifecycle.RENDERING_FAILED: {
        OutputLifecycle.VALIDATING,  # must revalidate before re-render
    },
}


# States from which no further modification is permitted.
IMMUTABLE_STATES: set[OutputLifecycle] = {
    OutputLifecycle.APPROVED,
    OutputLifecycle.ARCHIVED,
}


def can_transition(current: str | OutputLifecycle, target: str | OutputLifecycle) -> bool:
    """Return True if the transition from current to target is allowed."""
    cur = OutputLifecycle(current) if isinstance(current, str) else current
    tgt = OutputLifecycle(target) if isinstance(target, str) else target
    return tgt in ALLOWED_TRANSITIONS.get(cur, set())


def validate_transition(current: str | OutputLifecycle, target: str | OutputLifecycle) -> None:
    """Raise OutputLifecycleError if the transition is not allowed."""
    cur = OutputLifecycle(current) if isinstance(current, str) else current
    tgt = OutputLifecycle(target) if isinstance(target, str) else target

    if cur not in ALLOWED_TRANSITIONS:
        raise OutputLifecycleError(cur.value, tgt.value, f"Unknown current state {cur.value!r}")

    if tgt not in ALLOWED_TRANSITIONS[cur]:
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS[cur])
        reason = f"Allowed from {cur.value!r}: {allowed}" if allowed else "No transitions allowed from this state"
        raise OutputLifecycleError(cur.value, tgt.value, reason)


def is_immutable(state: str | OutputLifecycle) -> bool:
    """Return True if the state is immutable (approved or archived)."""
    s = OutputLifecycle(state) if isinstance(state, str) else state
    return s in IMMUTABLE_STATES


def requires_revalidation_before_render(state: str | OutputLifecycle) -> bool:
    """Return True if rendering from this state requires revalidation first."""
    s = OutputLifecycle(state) if isinstance(state, str) else state
    return s in (
        OutputLifecycle.VALIDATION_FAILED,
        OutputLifecycle.RENDERING_FAILED,
    )


def get_allowed_transitions(state: str | OutputLifecycle) -> list[str]:
    """Return sorted list of allowed next states for a given state."""
    s = OutputLifecycle(state) if isinstance(state, str) else state
    return sorted(t.value for t in ALLOWED_TRANSITIONS.get(s, set()))
