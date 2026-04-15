"""Brier score and skill score.

Brier score = mean squared error between predicted probability and actual outcome.
    0.0 = perfect predictions
    0.25 = always predicting 50%
    higher = worse than coin flip

Brier skill score = 1 - (brier / brier_of_reference). Positive = better than baseline.
"""

from __future__ import annotations

from collections.abc import Sequence


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error. `outcomes` must be 0/1."""
    if len(probabilities) != len(outcomes):
        raise ValueError("length mismatch")
    if not probabilities:
        raise ValueError("cannot compute Brier on empty inputs")
    n = len(probabilities)
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / n


def brier_skill_score(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    baseline: float | None = None,
) -> float:
    """Skill vs a reference forecaster.

    If `baseline` is None, uses the base rate of outcomes (most naive predictor:
    always predict the historical win rate).
    """
    if baseline is None:
        baseline = sum(outcomes) / len(outcomes)
    model = brier_score(probabilities, outcomes)
    ref = brier_score([baseline] * len(outcomes), outcomes)
    if ref == 0:
        return 0.0 if model == 0 else float("-inf")
    return 1.0 - model / ref
