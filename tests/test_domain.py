"""Tests for Event/Outcome domain primitives."""

from __future__ import annotations

from datetime import date

import pytest

from predictor.core.domain import Event, EventKind, Outcome


def make_event(**overrides) -> Event:
    defaults = dict(
        id="game_1",
        kind=EventKind.NBA_GAME,
        event_date=date(2025, 4, 1),
        outcome_a=Outcome(id="LAL", name="Lakers", is_b=False),
        outcome_b=Outcome(id="BOS", name="Celtics", is_b=True),
    )
    defaults.update(overrides)
    return Event(**defaults)


class TestEventValidation:
    def test_happy_path(self):
        e = make_event()
        assert not e.resolved
        assert e.winner_is_b is None

    def test_rejects_wrong_outcome_slots(self):
        with pytest.raises(ValueError):
            make_event(
                outcome_a=Outcome(id="A", name="A", is_b=True),
                outcome_b=Outcome(id="B", name="B", is_b=True),
            )


class TestResolution:
    def test_with_result_a_wins(self):
        e = make_event().with_result("LAL")
        assert e.resolved
        assert e.winner_is_b is False

    def test_with_result_b_wins(self):
        e = make_event().with_result("BOS")
        assert e.resolved
        assert e.winner_is_b is True

    def test_unknown_winner_id_raises(self):
        e = make_event().with_result("NYK")
        with pytest.raises(ValueError):
            _ = e.winner_is_b

    def test_immutable(self):
        e = make_event()
        with pytest.raises(Exception):  # FrozenInstanceError
            e.winner_id = "LAL"  # type: ignore[misc]
