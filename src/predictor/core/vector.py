"""A single directional signal in prediction space.

Design notes (changed from the earlier Java port):
- One signed-magnitude field instead of the redundant direction/magnitude split.
- `tanh` saturation instead of clipping, so strong signals don't all read the same.
- `group` tag so a VectorSpace can downweight correlated signals at aggregation time.
- Frozen dataclass — vectors are values, not bags of mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.predictor.core.math_utils import exp_freshness, tanh_saturate


@dataclass(frozen=True, slots=True)
class Vector:
    """One piece of evidence about a binary-outcome event.

    Convention:
        - `signed_magnitude` > 0 favors outcome B
        - `signed_magnitude` < 0 favors outcome A
        - All three of signed_magnitude, reliability, freshness live in [-1,1] or [0,1].

    The product `signed_magnitude * reliability * freshness` is the "effective force"
    that contributes to the net prediction signal.
    """

    source_id: str
    source_name: str
    category: str  # human-readable tag ("elo_diff", "rest_advantage", ...)
    group: str  # correlation group — vectors in the same group are co-aggregated
    data_date: date

    signed_magnitude: float  # [-1, +1], positive favors outcome B
    reliability: float  # [0, 1]
    freshness: float  # [0, 1] — set via `with_freshness` typically

    # Debug trail — keep the raw numbers that produced this vector
    raw_score: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not -1.0 - 1e-9 <= self.signed_magnitude <= 1.0 + 1e-9:
            raise ValueError(
                f"signed_magnitude {self.signed_magnitude} out of [-1, 1] "
                f"for {self.source_id}"
            )
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                f"reliability {self.reliability} out of [0, 1] for {self.source_id}"
            )
        if not 0.0 <= self.freshness <= 1.0:
            raise ValueError(
                f"freshness {self.freshness} out of [0, 1] for {self.source_id}"
            )

    def effective_force(self) -> float:
        """The actual contribution this vector makes to the net prediction signal."""
        return self.signed_magnitude * self.reliability * self.freshness

    def with_freshness(self, reference_date: date, half_life_days: float) -> Vector:
        """Return a copy with freshness recomputed relative to `reference_date`."""
        f = exp_freshness(self.data_date, reference_date, half_life_days)
        return replace_vector(self, freshness=f)

    @staticmethod
    def from_signed_score(
        source_id: str,
        source_name: str,
        category: str,
        group: str,
        data_date: date,
        raw_score: float,
        scale: float,
        reliability: float,
        notes: str = "",
    ) -> Vector:
        """Build a vector from an unbounded signed score.

        `raw_score * scale` is passed through tanh to get [-1, 1]. Positive score
        must mean "favors outcome B" — the caller owns that semantic mapping.
        """
        sm = tanh_saturate(raw_score * scale)
        return Vector(
            source_id=source_id,
            source_name=source_name,
            category=category,
            group=group,
            data_date=data_date,
            signed_magnitude=sm,
            reliability=reliability,
            freshness=1.0,  # caller applies with_freshness() after
            raw_score=raw_score,
            notes=notes,
        )


def replace_vector(v: Vector, **changes: object) -> Vector:
    """Like dataclasses.replace but typed for Vector."""
    from dataclasses import replace

    return replace(v, **changes)


# Re-export for convenience; callers can `from src.predictor.core.vector import replace_vector`.
__all__ = ["Vector", "replace_vector"]


# Suppress unused-import warning for `field` — kept for future additions.
_ = field
