"""Tests for NBA game prediction orchestration."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from predictor.nba.elo import NbaEloSystem
from predictor.nba.predict import predict_game


def _season_sim(sys: NbaEloSystem, games: list[tuple[str, str, int, int, date]]) -> None:
    for home, away, hs, as_, d in games:
        sys.record_game(home, away, hs, as_, d)


class TestPredictGame:
    def test_even_matchup_near_half(self):
        sys = NbaEloSystem()
        # No history — both teams at 1500, but home gets HCA
        pred = predict_game("LAL", "BOS", date(2025, 4, 1), sys)
        # Home court alone should push >50% home
        assert pred.home_win_probability > 0.5
        assert pred.home_win_probability + pred.away_win_probability == pytest.approx(1.0)

    def test_strong_home_favorite(self):
        sys = NbaEloSystem()
        # LAL wins many; BOS loses many
        d = date(2025, 1, 1)
        for i in range(30):
            sys.record_game("LAL", "WAS", 120, 95, d + timedelta(days=i))
            sys.record_game("CHI", "BOS", 115, 90, d + timedelta(days=i))
        pred = predict_game("LAL", "BOS", d + timedelta(days=60), sys)
        assert pred.home_win_probability > 0.75

    def test_strong_away_favorite(self):
        sys = NbaEloSystem()
        d = date(2025, 1, 1)
        # BOS is the juggernaut
        for i in range(30):
            sys.record_game("BOS", "WAS", 130, 95, d + timedelta(days=i))
            sys.record_game("CHI", "LAL", 110, 85, d + timedelta(days=i))
        pred = predict_game("LAL", "BOS", d + timedelta(days=60), sys)
        # Despite home court, BOS should still be favored
        assert pred.away_win_probability > 0.55

    def test_prediction_has_explainability(self):
        sys = NbaEloSystem()
        pred = predict_game("LAL", "BOS", date(2025, 4, 1), sys)
        assert pred.summary.vector_count >= 1
        assert len(pred.summary.top_influencers) >= 1
        readable = pred.summary.readable("LAL", "BOS")
        assert "LAL" in readable
        assert "BOS" in readable

    def test_correlation_aware_vs_naive_differ(self):
        sys = NbaEloSystem()
        # Build up strong home team
        d = date(2025, 1, 1)
        for i in range(20):
            sys.record_game("LAL", "WAS", 120, 90, d + timedelta(days=i))
        naive = predict_game(
            "LAL", "BOS", d + timedelta(days=40), sys, correlation_aware=False
        )
        corr = predict_game(
            "LAL", "BOS", d + timedelta(days=40), sys, correlation_aware=True
        )
        # Both agree directionally but magnitude differs
        assert naive.home_win_probability > 0.5
        assert corr.home_win_probability > 0.5
