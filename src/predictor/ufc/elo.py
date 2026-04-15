"""Elo rating system for UFC fighters.

Differences from NBA Elo:
- No home court advantage.
- Fights are less frequent; we don't regress ratings between seasons.
- Method of victory (KO, submission, decision) is NOT used for rating updates
  here — a win is a win. MMA is high-variance enough that overweighting KOs
  has historically hurt calibration. Revisit if backtest shows otherwise.
- Title fight bonus: we inflate K slightly for championship bouts, since those
  matchups carry more information about a fighter's true level.
- Draws possible (rare): update is symmetric toward expected score, no net change
  if both fighters equally rated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

INITIAL_RATING = 1500.0
K_FACTOR = 24.0
K_TITLE_BONUS = 6.0  # +6 K for title fights


@dataclass
class FighterState:
    fighter_id: str
    rating: float = INITIAL_RATING
    last_fight_date: date | None = None
    recent_results: list[str] = field(default_factory=list)  # "W", "L", or "D", most recent first
    total_fights: int = 0

    def form_score(self, window: int = 3) -> float:
        """Win rate over the last `window` fights, centered on 0.5. Returns [-0.5, +0.5]."""
        if not self.recent_results:
            return 0.0
        w = self.recent_results[:window]
        wins = sum(1 for r in w if r == "W")
        return wins / len(w) - 0.5

    def days_since_last_fight(self, as_of: date) -> int | None:
        if self.last_fight_date is None:
            return None
        return (as_of - self.last_fight_date).days


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    rating_a: float,
    rating_b: float,
    result: str,
    title_fight: bool = False,
) -> tuple[float, float]:
    """Update both fighters' ratings after a bout.

    `result` is from A's perspective: "W", "L", or "D".
    """
    if result not in {"W", "L", "D"}:
        raise ValueError(f"result must be W/L/D, got {result!r}")
    k = K_FACTOR + (K_TITLE_BONUS if title_fight else 0.0)
    exp_a = expected_score(rating_a, rating_b)
    actual_a = {"W": 1.0, "L": 0.0, "D": 0.5}[result]
    delta = k * (actual_a - exp_a)
    return rating_a + delta, rating_b - delta


class UfcEloSystem:
    """Stateful Elo system for UFC fighters."""

    def __init__(self) -> None:
        self._states: dict[str, FighterState] = {}

    def get_state(self, fighter_id: str) -> FighterState:
        if fighter_id not in self._states:
            self._states[fighter_id] = FighterState(fighter_id=fighter_id)
        return self._states[fighter_id]

    def rating(self, fighter_id: str) -> float:
        return self.get_state(fighter_id).rating

    def pregame_win_probability(self, fighter_a_id: str, fighter_b_id: str) -> float:
        return expected_score(self.rating(fighter_a_id), self.rating(fighter_b_id))

    def record_fight(
        self,
        fighter_a_id: str,
        fighter_b_id: str,
        result: str,  # W/L/D from A's perspective
        fight_date: date,
        title_fight: bool = False,
    ) -> None:
        a = self.get_state(fighter_a_id)
        b = self.get_state(fighter_b_id)
        new_a, new_b = update_ratings(a.rating, b.rating, result, title_fight=title_fight)
        a.rating, b.rating = new_a, new_b

        a.last_fight_date = fight_date
        b.last_fight_date = fight_date
        a.total_fights += 1
        b.total_fights += 1

        a_result = result
        b_result = {"W": "L", "L": "W", "D": "D"}[result]
        a.recent_results.insert(0, a_result)
        b.recent_results.insert(0, b_result)
        a.recent_results = a.recent_results[:10]
        b.recent_results = b.recent_results[:10]
