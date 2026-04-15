"""Tests for the Vector primitive."""

from __future__ import annotations

from datetime import date

import pytest

from predictor.core.vector import Vector


def make_vector(**overrides) -> Vector:
    defaults = dict(
        source_id="src1",
        source_name="Source 1",
        category="test",
        group="grp_a",
        data_date=date(2025, 1, 1),
        signed_magnitude=0.5,
        reliability=0.8,
        freshness=1.0,
    )
    defaults.update(overrides)
    return Vector(**defaults)


class TestVectorConstruction:
    def test_happy_path(self):
        v = make_vector()
        assert v.signed_magnitude == 0.5
        assert v.reliability == 0.8
        assert v.freshness == 1.0

    def test_rejects_signed_magnitude_out_of_range(self):
        with pytest.raises(ValueError):
            make_vector(signed_magnitude=1.5)
        with pytest.raises(ValueError):
            make_vector(signed_magnitude=-1.5)

    def test_rejects_reliability_out_of_range(self):
        with pytest.raises(ValueError):
            make_vector(reliability=1.5)
        with pytest.raises(ValueError):
            make_vector(reliability=-0.1)

    def test_rejects_freshness_out_of_range(self):
        with pytest.raises(ValueError):
            make_vector(freshness=1.5)
        with pytest.raises(ValueError):
            make_vector(freshness=-0.1)

    def test_is_immutable(self):
        v = make_vector()
        with pytest.raises(Exception):  # FrozenInstanceError
            v.signed_magnitude = 0.1  # type: ignore[misc]


class TestEffectiveForce:
    def test_product_of_three_components(self):
        v = make_vector(signed_magnitude=0.5, reliability=0.8, freshness=0.5)
        assert v.effective_force() == pytest.approx(0.5 * 0.8 * 0.5)

    def test_zero_reliability_nullifies(self):
        v = make_vector(signed_magnitude=0.9, reliability=0.0, freshness=1.0)
        assert v.effective_force() == 0.0

    def test_zero_freshness_nullifies(self):
        v = make_vector(signed_magnitude=0.9, reliability=1.0, freshness=0.0)
        assert v.effective_force() == 0.0

    def test_preserves_sign(self):
        pos = make_vector(signed_magnitude=0.5)
        neg = make_vector(signed_magnitude=-0.5)
        assert pos.effective_force() > 0
        assert neg.effective_force() < 0


class TestFromSignedScore:
    def test_positive_score_favors_b(self):
        v = Vector.from_signed_score(
            source_id="s", source_name="S", category="c", group="g",
            data_date=date(2025, 1, 1),
            raw_score=2.0, scale=1.0, reliability=1.0,
        )
        assert v.signed_magnitude > 0

    def test_negative_score_favors_a(self):
        v = Vector.from_signed_score(
            source_id="s", source_name="S", category="c", group="g",
            data_date=date(2025, 1, 1),
            raw_score=-2.0, scale=1.0, reliability=1.0,
        )
        assert v.signed_magnitude < 0

    def test_saturates_in_bounds(self):
        v = Vector.from_signed_score(
            source_id="s", source_name="S", category="c", group="g",
            data_date=date(2025, 1, 1),
            raw_score=1000.0, scale=1.0, reliability=1.0,
        )
        assert -1.0 <= v.signed_magnitude <= 1.0
        assert v.signed_magnitude > 0.99

    def test_preserves_raw_score_for_debug(self):
        v = Vector.from_signed_score(
            source_id="s", source_name="S", category="c", group="g",
            data_date=date(2025, 1, 1),
            raw_score=3.14, scale=2.0, reliability=0.5,
        )
        assert v.raw_score == 3.14


class TestWithFreshness:
    def test_same_date_full_freshness(self):
        v = make_vector(data_date=date(2025, 1, 1), freshness=0.5)
        v2 = v.with_freshness(reference_date=date(2025, 1, 1), half_life_days=10)
        assert v2.freshness == pytest.approx(1.0)

    def test_half_life_gives_half(self):
        v = make_vector(data_date=date(2025, 1, 1), freshness=1.0)
        v2 = v.with_freshness(reference_date=date(2025, 1, 11), half_life_days=10)
        assert v2.freshness == pytest.approx(0.5, rel=1e-3)

    def test_returns_new_instance(self):
        v = make_vector()
        v2 = v.with_freshness(reference_date=date(2025, 1, 2), half_life_days=30)
        assert v is not v2
        # Original unchanged
        assert v.freshness == 1.0
