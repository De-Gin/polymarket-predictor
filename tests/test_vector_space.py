"""Tests for VectorSpace aggregation."""

from __future__ import annotations

from datetime import date

import pytest

from predictor.core.vector import Vector
from predictor.core.vector_space import VectorSpace


def v(
    sm: float,
    group: str = "g1",
    reliability: float = 1.0,
    freshness: float = 1.0,
    category: str = "cat",
    source_id: str | None = None,
) -> Vector:
    return Vector(
        source_id=source_id or f"src_{sm}_{group}",
        source_name="x",
        category=category,
        group=group,
        data_date=date(2025, 1, 1),
        signed_magnitude=sm,
        reliability=reliability,
        freshness=freshness,
    )


class TestNetForce:
    def test_empty_space_is_zero(self):
        vs = VectorSpace(event_id="e1")
        assert vs.net_force() == 0.0

    def test_single_positive_vector(self):
        vs = VectorSpace(event_id="e1", correlation_aware=False)
        vs.add(v(0.4))
        assert vs.net_force() == pytest.approx(0.4)

    def test_sums_opposing_vectors(self):
        vs = VectorSpace(event_id="e1", correlation_aware=False)
        vs.add(v(0.6, group="g1"))
        vs.add(v(-0.3, group="g2"))
        assert vs.net_force() == pytest.approx(0.3)

    def test_correlation_aware_downweights_redundant_signals(self):
        """5 vectors in the same group should contribute much less than 5 in different groups."""
        redundant = VectorSpace(event_id="e1", correlation_aware=True)
        diverse = VectorSpace(event_id="e1", correlation_aware=True)
        for i in range(5):
            redundant.add(v(0.5, group="same"))
            diverse.add(v(0.5, group=f"g{i}"))
        # redundant: tanh(5 * 0.5) = tanh(2.5) ≈ 0.987
        # diverse: 5 * tanh(0.5) ≈ 5 * 0.462 = 2.31
        assert redundant.net_force() < 1.0
        assert diverse.net_force() > 2.0
        assert diverse.net_force() > redundant.net_force() * 2


class TestProbability:
    def test_zero_force_is_half(self):
        vs = VectorSpace(event_id="e1")
        assert vs.probability_b() == pytest.approx(0.5)

    def test_positive_force_favors_b(self):
        vs = VectorSpace(event_id="e1", correlation_aware=False)
        vs.add(v(0.8))
        assert vs.probability_b() > 0.5
        assert vs.probability_a() < 0.5

    def test_complementary(self):
        vs = VectorSpace(event_id="e1", correlation_aware=False)
        vs.add(v(0.3))
        vs.add(v(-0.1, group="g2"))
        assert vs.probability_a() + vs.probability_b() == pytest.approx(1.0)


class TestAgreement:
    def test_all_same_sign_high(self):
        vs = VectorSpace(event_id="e1")
        vs.add(v(0.4, group="g1"))
        vs.add(v(0.5, group="g2"))
        vs.add(v(0.6, group="g3"))
        assert vs.agreement() == pytest.approx(1.0)

    def test_split_vectors_low(self):
        vs = VectorSpace(event_id="e1")
        vs.add(v(0.5, group="g1"))
        vs.add(v(-0.5, group="g2"))
        # 50/50 split → agreement 0.5
        assert vs.agreement() == pytest.approx(0.5)

    def test_empty_is_zero(self):
        vs = VectorSpace(event_id="e1")
        assert vs.agreement() == 0.0


class TestExplainability:
    def test_force_by_group(self):
        vs = VectorSpace(event_id="e1")
        vs.add(v(0.3, group="g1"))
        vs.add(v(0.4, group="g1"))
        vs.add(v(-0.2, group="g2"))
        fbg = vs.force_by_group()
        assert fbg["g1"] == pytest.approx(0.7)
        assert fbg["g2"] == pytest.approx(-0.2)

    def test_top_influencers_sorted_by_abs_force(self):
        vs = VectorSpace(event_id="e1")
        vs.add(v(0.1, group="g1", source_id="small"))
        vs.add(v(-0.9, group="g2", source_id="big_neg"))
        vs.add(v(0.4, group="g3", source_id="medium"))
        top = vs.top_influencers(n=2)
        assert [x.source_id for x in top] == ["big_neg", "medium"]

    def test_conflicting_groups(self):
        vs = VectorSpace(event_id="e1", correlation_aware=False)
        vs.add(v(0.8, group="dominant"))  # overall force ≈ +0.6 → favors B
        vs.add(v(-0.2, group="contrarian"))
        conflicts = vs.conflicting_groups()
        assert "contrarian" in conflicts
        assert "dominant" not in conflicts

    def test_summarize_packages_everything(self):
        vs = VectorSpace(event_id="e1", correlation_aware=False)
        vs.add(v(0.4, group="g1"))
        vs.add(v(-0.1, group="g2"))
        s = vs.summarize()
        assert s.vector_count == 2
        assert s.net_force == pytest.approx(0.3)
        assert 0.5 < s.probability_b < 1.0
        assert "g1" in s.force_by_group
        assert len(s.top_influencers) == 2
