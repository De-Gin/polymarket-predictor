"""Domain-agnostic vector prediction engine.

Anything in this package must be sport-neutral. Sport-specific adapters
live in `predictor.nba` and `predictor.ufc`.
"""

from src.predictor.core.domain import Event, EventKind, Outcome
from src.predictor.core.vector import Vector
from src.predictor.core.vector_space import PredictionSummary, VectorSpace

__all__ = [
    "Event",
    "EventKind",
    "Outcome",
    "Vector",
    "VectorSpace",
    "PredictionSummary",
]
