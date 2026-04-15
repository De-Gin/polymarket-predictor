"""Elo rating system for NBA teams.

Tuned against 538-style parameters — the specific numbers here are a reasonable
starting point, not gospel. Calibrate via backtest before trusting them.

Constants worth understanding:
    K_FACTOR          How reactive the rating is to a single game. 20 is typical.
    HOME_COURT_ELO    How many Elo points home advantage is worth. Historically ~100.
                      COVID-era games dropped this to ~50. Use 80 as a compromise.
    MOV_MULT_A/B      Margin-of-victory multiplier tunables. Scales K by log-margin so
                      blowouts count more than squeakers, with diminishing returns.
    CARRY_OVER        Fraction of a team's end-of-season rating carried to the next
                      season. 0.75 means regress 25% toward 1505 (the league mean).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

INITIAL_RATING = 1500.0
LEAGUE_MEAN = 1505.0
K_FACTOR = 20.0
HOME_COURT_ELO = 80.0
MOV_MULT_A = 2.2
MOV_MULT_B = 0.001
CARRY_OVER = 0.75


@dataclass
class TeamState:
    """Snapshot of a single team's rating + recent context."""

    team_id: str  # abbreviation, e.g. "LAL"
    rating: float = INITIAL_RATING
    last_game_date: date | None = None
    recent_results: list[int] = field(default_factory=list)  # 1=W, 0=L, most recent first
    games_played: int = 0

    def form_score(self, window: int = 10) -> float:
        """Win rate over the last `window` games, centered on 0.5.

        Returns a value in [-0.5, +0.5]. Positive = winning more than half.
        """
        if not self.recent_results:
            return 0.0
        w = self.recent_results[:window]
        return sum(w) / len(w) - 0.5

    def days_rest(self, as_of: date) -> int | None:
        if self.last_game_date is None:
            return None
        return (as_of - self.last_game_date).days


def expected_score(rating_a: float, rating_b: float, home_advantage_for_a: float = 0.0) -> float:
    """Pre-game probability that team A beats team B.

    `home_advantage_for_a` is the Elo adjustment; pass HOME_COURT_ELO if A is home,
    -HOME_COURT_ELO if B is home, 0 for neutral venue.
    """
    diff = (rating_a + home_advantage_for_a) - rating_b
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """538-style margin-of-victory scaling."""
    return math.log(max(margin, 1) + 1) * (MOV_MULT_A / (elo_diff_winner * MOV_MULT_B + MOV_MULT_A))


def update_ratings(
    rating_home: float,
    rating_away: float,
    home_won: bool,
    margin: int,
    neutral_site: bool = False,
) -> tuple[float, float]:
    """Returns (new_home_rating, new_away_rating) after a single game."""
    hca = 0.0 if neutral_site else HOME_COURT_ELO
    exp_home = expected_score(rating_home, rating_away, hca)
    actual_home = 1.0 if home_won else 0.0

    # MoV multiplier is applied from the winner's perspective
    if home_won:
        elo_diff_winner = (rating_home + hca) - rating_away
    else:
        elo_diff_winner = rating_away - (rating_home + hca)
    mov = _mov_multiplier(margin, elo_diff_winner)

    delta = K_FACTOR * mov * (actual_home - exp_home)
    return rating_home + delta, rating_away - delta


class NbaEloSystem:
    """Stateful Elo system. Feed it games in chronological order.

    Call `record_game` for each played game; query `get_state(team_id)` anytime
    to get the current rating. Use `start_new_season` to apply carryover regression.
    """

    def __init__(self) -> None:
        self._states: dict[str, TeamState] = {}

    # -- queries --------------------------------------------------------------

    def get_state(self, team_id: str) -> TeamState:
        if team_id not in self._states:
            self._states[team_id] = TeamState(team_id=team_id)
        return self._states[team_id]

    def rating(self, team_id: str) -> float:
        return self.get_state(team_id).rating

    def pregame_win_probability(
        self, home_id: str, away_id: str, neutral_site: bool = False
    ) -> float:
        hca = 0.0 if neutral_site else HOME_COURT_ELO
        return expected_score(self.rating(home_id), self.rating(away_id), hca)

    # -- mutations ------------------------------------------------------------

    def record_game(
        self,
        home_id: str,
        away_id: str,
        home_score: int,
        away_score: int,
        game_date: date,
        neutral_site: bool = False,
    ) -> None:
        """Update both teams' ratings based on the game outcome."""
        if home_score == away_score:
            raise ValueError(f"Tie not allowed in NBA ({home_score}-{away_score})")
        home = self.get_state(home_id)
        away = self.get_state(away_id)

        home_won = home_score > away_score
        margin = abs(home_score - away_score)
        new_home, new_away = update_ratings(
            home.rating, away.rating, home_won, margin, neutral_site=neutral_site
        )

        home.rating = new_home
        away.rating = new_away
        home.last_game_date = game_date
        away.last_game_date = game_date
        home.games_played += 1
        away.games_played += 1
        home.recent_results.insert(0, 1 if home_won else 0)
        away.recent_results.insert(0, 0 if home_won else 1)
        # Keep memory bounded
        home.recent_results = home.recent_results[:20]
        away.recent_results = away.recent_results[:20]

    def start_new_season(self) -> None:
        """Regress all ratings toward the league mean by (1 - CARRY_OVER)."""
        for state in self._states.values():
            state.rating = LEAGUE_MEAN + (state.rating - LEAGUE_MEAN) * CARRY_OVER
            state.recent_results = []
            state.games_played = 0
            state.last_game_date = None
