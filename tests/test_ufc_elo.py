"""Tests for UFC Elo system."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from predictor.ufc.elo import (
    INITIAL_RATING,
    UfcEloSystem,
    expected_score,
    update_ratings,
)


class TestExpectedScore:
    def test_equal_ratings(self):
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_favored(self):
        assert expected_score(1700, 1500) > 0.5


class TestUpdateRatings:
    def test_winner_gains(self):
        na, nb = update_ratings(1500, 1500, "W")
        assert na > 1500
        assert nb < 1500

    def test_draw_no_net_change_equal_rated(self):
        na, nb = update_ratings(1500, 1500, "D")
        assert na == pytest.approx(1500)
        assert nb == pytest.approx(1500)

    def test_draw_shifts_underdog_toward_favorite(self):
        # Higher-rated fighter draws with lower-rated — underdog gains
        na, nb = update_ratings(1700, 1400, "D")
        assert na < 1700
        assert nb > 1400

    def test_title_fight_larger_movement(self):
        na_regular, _ = update_ratings(1500, 1500, "W", title_fight=False)
        na_title, _ = update_ratings(1500, 1500, "W", title_fight=True)
        assert na_title - 1500 > na_regular - 1500

    def test_invalid_result_raises(self):
        with pytest.raises(ValueError):
            update_ratings(1500, 1500, "X")


class TestUfcEloSystem:
    def test_new_fighter_initial_rating(self):
        sys = UfcEloSystem()
        assert sys.rating("mcgregor") == INITIAL_RATING

    def test_record_fight_updates_both(self):
        sys = UfcEloSystem()
        sys.record_fight("mcgregor", "poirier", "W", date(2021, 1, 23))
        assert sys.rating("mcgregor") > INITIAL_RATING
        assert sys.rating("poirier") < INITIAL_RATING

    def test_form_score_tracks_results(self):
        sys = UfcEloSystem()
        d = date(2023, 1, 1)
        for i in range(3):
            sys.record_fight("jones", "opp", "W", d + timedelta(days=i * 90))
        assert sys.get_state("jones").form_score(window=3) == 0.5  # 3 wins, centered

    def test_form_score_mixed(self):
        sys = UfcEloSystem()
        d = date(2023, 1, 1)
        sys.record_fight("f1", "o", "W", d)
        sys.record_fight("f1", "o", "L", d + timedelta(days=180))
        sys.record_fight("f1", "o", "W", d + timedelta(days=360))
        # 2W 1L → form = 2/3 - 0.5 = 0.1666
        assert sys.get_state("f1").form_score(window=3) == pytest.approx(2 / 3 - 0.5)

    def test_days_since_last_fight(self):
        sys = UfcEloSystem()
        sys.record_fight("khabib", "gaethje", "W", date(2020, 10, 24))
        assert sys.get_state("khabib").days_since_last_fight(date(2020, 11, 24)) == 31
