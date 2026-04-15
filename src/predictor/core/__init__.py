"""Domain-agnostic vector prediction engine.

Anything in this package must be sport-neutral. Sport-specific adapters
live in `predictor.nba` and `predictor.ufc`.
"""

from predictor.core.domain import Event, EventKind, Outcome
from predictor.core.vector import Vector
from predictor.core.vector_space import PredictionSummary, VectorSpace

__all__ = [
    "Event",
    "EventKind",
    "Outcome",
    "Vector",
    "VectorSpace",
    "PredictionSummary",
]
