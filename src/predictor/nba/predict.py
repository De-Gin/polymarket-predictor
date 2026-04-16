"""High-level NBA game prediction orchestration.

Ties Elo, features, and VectorSpace together into a one-call `predict_game`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.predictor.core.domain import Event, EventKind, Outcome
from src.predictor.core.vector_space import PredictionSummary, VectorSpace
from predictor.nba.elo import NbaEloSystem
from predictor.nba.features import GameContext, extract_game_vectors


@dataclass
class GamePrediction:
    event: Event
    summary: PredictionSummary

    @property
    def home_win_probability(self) -> float:
        # Home team is outcome A (is_b=False)
        return self.summary.probability_a

    @property
    def away_win_probability(self) -> float:
        return self.summary.probability_b


def predict_game(
    home_team: str,
    away_team: str,
    game_date: date,
    elo_system: NbaEloSystem,
    sensitivity: float = 1.0,
    correlation_aware: bool = True,
    neutral_site: bool = False,
) -> GamePrediction:
    """Predict a single NBA game using the current Elo system state.

    IMPORTANT: `elo_system` must already contain all games played BEFORE
    `game_date`. To backtest, replay games chronologically and call
    `predict_game` with the pre-game Elo state, then record the game's result.
    """
    event = Event(
        id=f"{game_date.isoformat()}_{away_team}@{home_team}",
        kind=EventKind.NBA_GAME,
        event_date=game_date,
        outcome_a=Outcome(id=home_team, name=home_team, is_b=False),
        outcome_b=Outcome(id=away_team, name=away_team, is_b=True),
    )

    ctx = GameContext(
        game_date=game_date,
        home_team=home_team,
        away_team=away_team,
        neutral_site=neutral_site,
    )
    vectors = extract_game_vectors(ctx, elo_system)

    space = VectorSpace(
        event_id=event.id,
        correlation_aware=correlation_aware,
        sensitivity=sensitivity,
    )
    space.extend(vectors)

    return GamePrediction(event=event, summary=space.summarize())
