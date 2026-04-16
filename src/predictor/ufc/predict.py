"""High-level UFC fight prediction orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.predictor.core.domain import Event, EventKind, Outcome
from src.predictor.core.vector_space import PredictionSummary, VectorSpace
from predictor.ufc.elo import UfcEloSystem
from predictor.ufc.features import FightContext, FighterProfile, extract_fight_vectors


@dataclass
class FightPrediction:
    event: Event
    summary: PredictionSummary

    @property
    def fighter_a_win_probability(self) -> float:
        return self.summary.probability_a

    @property
    def fighter_b_win_probability(self) -> float:
        return self.summary.probability_b


def predict_fight(
    fighter_a: FighterProfile,
    fighter_b: FighterProfile,
    fight_date: date,
    elo_system: UfcEloSystem,
    title_fight: bool = False,
    sensitivity: float = 1.0,
    correlation_aware: bool = True,
    a_name: str | None = None,
    b_name: str | None = None,
) -> FightPrediction:
    """Predict a single UFC fight.

    `elo_system` must already contain all fights prior to `fight_date`.
    """
    event = Event(
        id=f"{fight_date.isoformat()}_{fighter_a.fighter_id}_vs_{fighter_b.fighter_id}",
        kind=EventKind.UFC_FIGHT,
        event_date=fight_date,
        outcome_a=Outcome(id=fighter_a.fighter_id, name=a_name or fighter_a.fighter_id, is_b=False),
        outcome_b=Outcome(id=fighter_b.fighter_id, name=b_name or fighter_b.fighter_id, is_b=True),
    )
    ctx = FightContext(
        fight_date=fight_date, fighter_a=fighter_a, fighter_b=fighter_b, title_fight=title_fight
    )
    vectors = extract_fight_vectors(ctx, elo_system)

    space = VectorSpace(
        event_id=event.id,
        correlation_aware=correlation_aware,
        sensitivity=sensitivity,
    )
    space.extend(vectors)
    return FightPrediction(event=event, summary=space.summarize())
