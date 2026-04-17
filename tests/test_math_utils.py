"""Tests for core numerical utilities."""

from __future__ import annotations

import math
from datetime import date

import pytest

from src.predictor.core.math_utils import exp_freshness, sigmoid, tanh_saturate


class TestSigmoid:
    def test_zero_input_is_half(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_saturates_to_one(self):
        assert sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative_saturates_to_zero(self):
        assert sigmoid(-100.0) == pytest.approx(0.0)

    def test_symmetric(self):
        assert sigmoid(1.5) + sigmoid(-1.5) == pytest.approx(1.0)

    def test_overflow_guard(self):
        # Must not raise even at extreme values
        assert sigmoid(1e6, sensitivity=10.0) == pytest.approx(1.0)
        assert sigmoid(-1e6, sensitivity=10.0) == pytest.approx(0.0)

    def test_sensitivity_steepens(self):
        # Same input, higher sensitivity → closer to 0 or 1
        low = sigmoid(0.5, sensitivity=1.0)
        high = sigmoid(0.5, sensitivity=3.0)
        assert 0.5 < low < high < 1.0


class TestTanhSaturate:
    def test_zero_stays_zero(self):
        assert tanh_saturate(0.0) == 0.0

    def test_saturates_in_bounds(self):
        assert -1.0 < tanh_saturate(-10.0) <= -0.999
        assert 0.999 <= tanh_saturate(10.0) < 1.0

    def test_preserves_sign(self):
        assert tanh_saturate(0.5) > 0
        assert tanh_saturate(-0.5) < 0

    def test_no_clipping_midrange(self):
        # Unlike max(-1, min(1, x)), tanh differentiates mid-range inputs
        assert tanh_saturate(0.5) != tanh_saturate(0.7)
        assert tanh_saturate(2.0) != tanh_saturate(3.0)


class TestExpFreshness:
    def test_zero_days_old_is_one(self):
        d = date(2025, 1, 1)
        assert exp_freshness(d, d, half_life_days=30) == pytest.approx(1.0)

    def test_half_life(self):
        d1 = date(2025, 1, 1)
        d2 = date(2025, 1, 31)
        assert exp_freshness(d1, d2, half_life_days=30) == pytest.approx(0.5, rel=1e-3)

    def test_symmetric_in_time(self):
        d1 = date(2025, 1, 1)
        d2 = date(2025, 2, 1)
        forward = exp_freshness(d1, d2, half_life_days=30)
        backward = exp_freshness(d2, d1, half_life_days=30)
        assert forward == backward

    def test_rejects_nonpositive_half_life(self):
        d = date(2025, 1, 1)
        with pytest.raises(ValueError):
            exp_freshness(d, d, half_life_days=0)
        with pytest.raises(ValueError):
            exp_freshness(d, d, half_life_days=-1)

    def test_decays_monotonically(self):
        from datetime import timedelta

        d = date(2025, 1, 1)
        values = [exp_freshness(d, d + timedelta(days=n), 30) for n in range(0, 180, 10)]
        assert all(a >= b for a, b in zip(values, values[1:]))
        assert values[-1] < 0.1
        assert not any(math.isnan(v) for v in values)
