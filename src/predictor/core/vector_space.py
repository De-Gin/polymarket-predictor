"""Aggregates vectors into a single probability estimate with explainability."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from src.predictor.core.math_utils import sigmoid, tanh_saturate
from src.predictor.core.vector import Vector


@dataclass(frozen=True, slots=True)
class PredictionSummary:
    """Immutable snapshot of a prediction: what we think and why."""

    net_force: float
    probability_b: float
    agreement: float  # how aligned vectors are — NOT true calibrated confidence
    vector_count: int
    force_by_group: dict[str, float]
    force_by_category: dict[str, float]
    top_influencers: list[Vector]
    conflicting_groups: list[str]
    correlation_aware: bool

    @property
    def probability_a(self) -> float:
        return 1.0 - self.probability_b

    def readable(self, outcome_a_name: str, outcome_b_name: str) -> str:
        lines = ["=== PREDICTION ==="]
        lines.append(f"  {outcome_a_name}: {self.probability_a * 100:.1f}%")
        lines.append(f"  {outcome_b_name}: {self.probability_b * 100:.1f}%")
        lines.append(
            f"  net force: {self.net_force:+.3f}   agreement: {self.agreement:.2f}   "
            f"vectors: {self.vector_count}"
        )
        if self.conflicting_groups:
            lines.append(f"  conflicting groups: {', '.join(self.conflicting_groups)}")
        lines.append("  top factors:")
        for v in self.top_influencers:
            favors = outcome_b_name if v.effective_force() > 0 else outcome_a_name
            lines.append(
                f"    {v.source_name}  →  favors {favors}  "
                f"(force {v.effective_force():+.3f}, group={v.group})"
            )
        return "\n".join(lines)


@dataclass
class VectorSpace:
    """Holds all vectors for a single event and computes the aggregate signal.

    Parameters
    ----------
    event_id:
        Opaque identifier — a vector space always belongs to one event.
    correlation_aware:
        If True (default), vectors in the same `group` are summed and then passed
        through tanh before being added to the net force. This prevents N highly
        correlated signals from double-counting. If False, falls back to naive
        sum (matches the original Java engine).
    sensitivity:
        Input scaling for the sigmoid. Tune via backtest calibration.
    """

    event_id: str
    correlation_aware: bool = True
    sensitivity: float = 1.0
    vectors: list[Vector] = field(default_factory=list)

    def add(self, v: Vector) -> None:
        self.vectors.append(v)

    def extend(self, vs: Iterable[Vector]) -> None:
        self.vectors.extend(vs)

    # -- aggregation ----------------------------------------------------------

    def force_by_group(self) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for v in self.vectors:
            acc[v.group] += v.effective_force()
        return dict(acc)

    def force_by_category(self) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for v in self.vectors:
            acc[v.category] += v.effective_force()
        return dict(acc)

    def net_force(self) -> float:
        """Aggregate signal.

        When `correlation_aware=True`, signals are first summed within each
        correlation group and each group's total is passed through tanh. So a
        group with 5 redundant vectors contributes at most ±1 — not ±5. The
        inter-group sum is unbounded (sigmoid handles it downstream).
        """
        if not self.correlation_aware:
            return sum(v.effective_force() for v in self.vectors)
        return sum(tanh_saturate(g) for g in self.force_by_group().values())

    def probability_b(self) -> float:
        return sigmoid(self.net_force(), self.sensitivity)

    def probability_a(self) -> float:
        return 1.0 - self.probability_b()

    # -- explainability -------------------------------------------------------

    def agreement(self) -> float:
        """How aligned the vectors are.

        NOTE: this is an *agreement* score (all forces same sign), not a proper
        calibrated confidence. True confidence must come from out-of-sample
        Brier calibration — see `predictor.eval.calibration`. Keep using this
        only for quick explainability.
        """
        forces = [v.effective_force() for v in self.vectors if abs(v.effective_force()) > 1e-6]
        if not forces:
            return 0.0

        pos_mag = sum(abs(f) for f in forces if f > 0)
        neg_mag = sum(abs(f) for f in forces if f < 0)
        total_mag = pos_mag + neg_mag
        if total_mag == 0:
            return 0.0
        dominant = max(pos_mag, neg_mag)
        mag_ratio = dominant / total_mag

        pos_n = sum(1 for f in forces if f > 0)
        neg_n = sum(1 for f in forces if f < 0)
        count_ratio = max(pos_n, neg_n) / (pos_n + neg_n)

        return (mag_ratio + count_ratio) / 2.0

    def top_influencers(self, n: int = 5) -> list[Vector]:
        return sorted(self.vectors, key=lambda v: -abs(v.effective_force()))[:n]

    def conflicting_groups(self) -> list[str]:
        """Groups whose net force points opposite the overall net force."""
        overall_sign = 1 if self.net_force() >= 0 else -1
        out = []
        for g, f in self.force_by_group().items():
            if abs(f) < 0.01:
                continue
            if (f > 0) != (overall_sign > 0):
                out.append(g)
        return out

    def summarize(self) -> PredictionSummary:
        return PredictionSummary(
            net_force=self.net_force(),
            probability_b=self.probability_b(),
            agreement=self.agreement(),
            vector_count=len(self.vectors),
            force_by_group=self.force_by_group(),
            force_by_category=self.force_by_category(),
            top_influencers=self.top_influencers(),
            conflicting_groups=self.conflicting_groups(),
            correlation_aware=self.correlation_aware,
        )
