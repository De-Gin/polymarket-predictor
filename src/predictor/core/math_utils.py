"""Numerical utilities used throughout the vector engine."""

from __future__ import annotations

import math
from datetime import date


def sigmoid(x: float, sensitivity: float = 1.0) -> float:
    """Logistic function. `sensitivity` scales the input before applying 1/(1+e^-x).

    Higher sensitivity = the probability moves toward 0 or 1 faster for a given
    net force. Typical values: 0.5 (conservative) to 3.0 (decisive). Learn from
    backtest calibration rather than picking arbitrarily.
    """
    # Guard against overflow for large negative inputs.
    z = -x * sensitivity
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def tanh_saturate(x: float) -> float:
    """Map an unbounded signed score into [-1, 1] smoothly.

    Prefer this over `max(-1, min(1, x))` — clipping makes a 3σ shock read
    identically to a 1σ one, which throws away information.
    """
    return math.tanh(x)


def exp_freshness(data_date: date, reference_date: date, half_life_days: float) -> float:
    """Exponential time decay. Returns 1.0 when dates are equal, 0.5 at one half-life.

    Applied symmetrically — data from the future (e.g. during backtesting where
    reference_date lags data_date) decays the same as data from the past.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    days_old = abs((data_date - reference_date).days)
    lam = math.log(2) / half_life_days
    return math.exp(-lam * days_old)
