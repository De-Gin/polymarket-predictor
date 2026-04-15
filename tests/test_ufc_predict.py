"""Tests for UFC fight prediction orchestration."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from predictor.ufc.elo import UfcEloSystem
from predictor.ufc.features import FighterProfile
from predictor.ufc.predict import predict_fight


class TestPredictFight:
    def test_balanced_no_history_near_half(self):
        sys = UfcEloSystem()
        a = FighterProfile(fighter_id="a", age=30, reach_cm=180)
        b = FighterProfile(fighter_id="b", age=30, reach_cm=180)
        pred = predict_fight(a, b, date(2025, 1, 1), sys)
        assert 0.45 < pred.fighter_a_win_probability < 0.55

    def test_reach_advantage_moves_probability(self):
        sys = UfcEloSystem()
        a = FighterProfile(fighter_id="a", age=30, reach_cm=170)
        b = FighterProfile(fighter_id="b", age=30, reach_cm=195)  # +25cm reach
        pred = predict_fight(a, b, date(2025, 1, 1), sys)
        assert pred.fighter_b_win_probability > pred.fighter_a_win_probability

    def test_age_advantage_helps_younger(self):
        sys = UfcEloSystem()
        a = FighterProfile(fighter_id="a", age=38)
        b = FighterProfile(fighter_id="b", age=26)
        pred = predict_fight(a, b, date(2025, 1, 1), sys)
        assert pred.fighter_b_win_probability > 0.55

    def test_elo_dominance(self):
        sys = UfcEloSystem()
        # Build up A's rating
        d = date(2020, 1, 1)
        for i in range(15):
            sys.record_fight("champ", f"opp_{i}", "W", d + timedelta(days=i * 90))
            sys.record_fight("jobber", f"opp_b_{i}", "L", d + timedelta(days=i * 90))
        a = FighterProfile(fighter_id="champ", age=30)
        b = FighterProfile(fighter_id="jobber", age=30)
        pred = predict_fight(a, b, date(2025, 1, 1), sys)
        # With correlation_aware=True, Elo + form share the "skill" group and the
        # aggregate is tanh-saturated. A single-group signal can't push past
        # sigmoid(1.0) ≈ 0.73 on its own, which is by design — other vector
        # groups (physicality, striking, grappling) need to stack on top to get
        # extreme probabilities. See README "correlation-aware aggregation".
        assert pred.fighter_a_win_probability > 0.65

    def test_handles_missing_attributes(self):
        sys = UfcEloSystem()
        a = FighterProfile(fighter_id="a")  # all None
        b = FighterProfile(fighter_id="b")
        pred = predict_fight(a, b, date(2025, 1, 1), sys)
        # Still produces a prediction
        assert pred.fighter_a_win_probability + pred.fighter_b_win_probability == pytest.approx(1.0)
        # Only Elo vector should fire
        assert pred.summary.vector_count >= 1

    def test_readable_output(self):
        sys = UfcEloSystem()
        a = FighterProfile(fighter_id="islam", age=31, reach_cm=178)
        b = FighterProfile(fighter_id="volk", age=36, reach_cm=179)
        pred = predict_fight(a, b, date(2025, 1, 1), sys, a_name="Islam", b_name="Volk")
        out = pred.summary.readable("Islam", "Volk")
        assert "Islam" in out
        assert "Volk" in out
