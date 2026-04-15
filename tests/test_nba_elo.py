"""Tests for the NBA Elo system."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from predictor.nba.elo import (
    HOME_COURT_ELO,
    INITIAL_RATING,
    LEAGUE_MEAN,
    NbaEloSystem,
    expected_score,
    update_ratings,
)


class TestExpectedScore:
    def test_equal_ratings_neutral_is_half(self):
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_favored(self):
        assert expected_score(1600, 1500) > 0.5

    def test_home_advantage_helps(self):
        neutral = expected_score(1500, 1500, 0)
        home = expected_score(1500, 1500, HOME_COURT_ELO)
        assert home > neutral

    def test_400_elo_diff_is_about_10_to_1(self):
        # Classic Elo property
        p = expected_score(1900, 1500)
        assert 0.9 < p < 0.92


class TestUpdateRatings:
    def test_winner_gains_loser_loses(self):
        nh, na = update_ratings(1500, 1500, home_won=True, margin=10)
        assert nh > 1500
        assert na < 1500
        # Conservation
        assert nh - 1500 == pytest.approx(-(na - 1500))

    def test_upset_shifts_ratings_more(self):
        # Favored home team wins — small rating change
        fav_win_h, fav_win_a = update_ratings(1700, 1400, home_won=True, margin=10)
        # Upset — underdog wins
        ups_win_h, ups_win_a = update_ratings(1700, 1400, home_won=False, margin=10)
        # Rating movement
        assert abs(ups_win_a - 1400) > abs(fav_win_a - 1400)

    def test_blowout_moves_more_than_squeaker(self):
        close_h, _ = update_ratings(1500, 1500, home_won=True, margin=2)
        blowout_h, _ = update_ratings(1500, 1500, home_won=True, margin=30)
        assert blowout_h > close_h


class TestEloSystem:
    def test_new_team_has_initial_rating(self):
        sys = NbaEloSystem()
        assert sys.rating("LAL") == INITIAL_RATING

    def test_record_game_updates_both(self):
        sys = NbaEloSystem()
        sys.record_game("LAL", "BOS", 120, 100, date(2025, 1, 1))
        assert sys.rating("LAL") > INITIAL_RATING
        assert sys.rating("BOS") < INITIAL_RATING

    def test_rejects_tie(self):
        sys = NbaEloSystem()
        with pytest.raises(ValueError):
            sys.record_game("LAL", "BOS", 100, 100, date(2025, 1, 1))

    def test_tracks_games_played(self):
        sys = NbaEloSystem()
        for d in range(1, 6):
            sys.record_game("LAL", "BOS", 110, 100, date(2025, 1, d))
        assert sys.get_state("LAL").games_played == 5
        assert sys.get_state("BOS").games_played == 5

    def test_recent_results_bounded(self):
        sys = NbaEloSystem()
        for d in range(1, 25):
            sys.record_game("LAL", "BOS", 110, 100, date(2025, 1, 1) + timedelta(days=d))
        # We keep only 20 most recent
        assert len(sys.get_state("LAL").recent_results) == 20
        # And the form is all wins (1.0) for LAL
        assert sys.get_state("LAL").form_score(window=10) == 0.5  # 1.0 - 0.5

    def test_start_new_season_regresses_toward_mean(self):
        sys = NbaEloSystem()
        sys.get_state("LAL").rating = 1800
        sys.get_state("BOS").rating = 1300
        sys.start_new_season()
        lal = sys.rating("LAL")
        bos = sys.rating("BOS")
        # Both moved toward league mean
        assert 1500 < lal < 1800
        assert 1300 < bos < LEAGUE_MEAN + 1  # BOS was below mean
        # Symmetry: equal distance kept (75%)
        assert lal - LEAGUE_MEAN == pytest.approx((1800 - LEAGUE_MEAN) * 0.75)

    def test_days_rest(self):
        sys = NbaEloSystem()
        sys.record_game("LAL", "BOS", 110, 100, date(2025, 1, 1))
        assert sys.get_state("LAL").days_rest(date(2025, 1, 5)) == 4
