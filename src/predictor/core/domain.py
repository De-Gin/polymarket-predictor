"""Domain primitives for a binary-outcome prediction event.

Intentionally generic — an `Event` is anything with two outcomes (A vs B) and
a scheduled date. NBA games, UFC fights, coin flips all fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class EventKind(str, Enum):
    """Supported event types. Extend as you add domains."""

    NBA_GAME = "nba_game"
    UFC_FIGHT = "ufc_fight"


@dataclass(frozen=True, slots=True)
class Outcome:
    """One of two outcomes in a binary prediction event.

    Convention:
        - `is_b = False` → outcome A (the "reference" side, e.g. home team, favorite)
        - `is_b = True`  → outcome B (the "challenger" side, e.g. away team, underdog)

    Positive net force favors outcome B.
    """

    id: str  # stable identifier (e.g. team abbreviation, fighter id)
    name: str
    is_b: bool


@dataclass(frozen=True, slots=True)
class Event:
    """A binary-outcome prediction event.

    `reference_date` is used for freshness decay. Typically the event date itself,
    but during backtesting you'd pass a pre-event date to simulate a real prediction.
    """

    id: str
    kind: EventKind
    event_date: date
    outcome_a: Outcome
    outcome_b: Outcome
    # Optional resolution
    winner_id: str | None = None

    def __post_init__(self) -> None:
        if self.outcome_a.is_b or not self.outcome_b.is_b:
            raise ValueError(
                f"Event {self.id}: outcome_a must have is_b=False, "
                f"outcome_b must have is_b=True"
            )

    @property
    def resolved(self) -> bool:
        return self.winner_id is not None

    @property
    def winner_is_b(self) -> bool | None:
        if self.winner_id is None:
            return None
        if self.winner_id == self.outcome_a.id:
            return False
        if self.winner_id == self.outcome_b.id:
            return True
        raise ValueError(
            f"winner_id {self.winner_id!r} matches neither outcome "
            f"({self.outcome_a.id!r}, {self.outcome_b.id!r})"
        )

    def with_result(self, winner_id: str) -> Event:
        """Return a copy of this event with the winner set."""
        return Event(
            id=self.id,
            kind=self.kind,
            event_date=self.event_date,
            outcome_a=self.outcome_a,
            outcome_b=self.outcome_b,
            winner_id=winner_id,
        )
