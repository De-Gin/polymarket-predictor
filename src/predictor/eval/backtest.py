"""Chronological backtest harness for NBA seasons and UFC card history.

The harness replays events in order. Before each event, we predict with the
current rating state; then we record the actual outcome, updating ratings
going forward. This avoids leak — the model never sees a game's outcome
before predicting it.

Returns a `BacktestResult` with probability/outcome arrays so you can feed
them to `brier_score` and `calibration_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from predictor.nba.elo import NbaEloSystem
from predictor.nba.predict import predict_game
from predictor.ufc.elo import UfcEloSystem
from predictor.ufc.features import FighterProfile
from predictor.ufc.predict import predict_fight


@dataclass
class BacktestRow:
    event_id: str
    event_date: date
    outcome_a_name: str
    outcome_b_name: str
    model_probability_b: float  # probability B wins
    actual_outcome_b: int  # 1 if B won, 0 otherwise


@dataclass
class BacktestResult:
    rows: list[BacktestRow] = field(default_factory=list)

    @property
    def probabilities(self) -> list[float]:
        return [r.model_probability_b for r in self.rows]

    @property
    def outcomes(self) -> list[int]:
        return [r.actual_outcome_b for r in self.rows]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "event_id": r.event_id,
                    "event_date": r.event_date,
                    "a": r.outcome_a_name,
                    "b": r.outcome_b_name,
                    "p_b": r.model_probability_b,
                    "outcome_b": r.actual_outcome_b,
                }
                for r in self.rows
            ]
        )


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------


def backtest_nba_season(
    games: pd.DataFrame,
    sensitivity: float = 1.0,
    correlation_aware: bool = True,
    skip_warmup_games: int = 200,
) -> BacktestResult:
    """Run a chronological backtest over one NBA season's games.

    Parameters
    ----------
    games:
        DataFrame from `nba.data.merge_to_games` with columns
        GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_PTS, AWAY_PTS, HOME_WON.
    skip_warmup_games:
        Don't record predictions for the first N games — Elo needs a burn-in
        period before predictions are meaningful. Default 200 (~6 weeks of NBA).
    """
    sys = NbaEloSystem()
    result = BacktestResult()

    for i, game in enumerate(games.itertuples(index=False)):
        home, away = game.HOME_TEAM, game.AWAY_TEAM
        game_date = _as_date(game.GAME_DATE)

        if i >= skip_warmup_games:
            pred = predict_game(
                home_team=home,
                away_team=away,
                game_date=game_date,
                elo_system=sys,
                sensitivity=sensitivity,
                correlation_aware=correlation_aware,
            )
            outcome_b = 0 if game.HOME_WON else 1  # B is the away team
            result.rows.append(
                BacktestRow(
                    event_id=str(game.GAME_ID),
                    event_date=game_date,
                    outcome_a_name=home,
                    outcome_b_name=away,
                    model_probability_b=pred.away_win_probability,
                    actual_outcome_b=outcome_b,
                )
            )

        sys.record_game(
            home_id=home,
            away_id=away,
            home_score=int(game.HOME_PTS),
            away_score=int(game.AWAY_PTS),
            game_date=game_date,
        )

    return result


# ---------------------------------------------------------------------------
# UFC
# ---------------------------------------------------------------------------


def backtest_ufc(
    fights: pd.DataFrame,
    sensitivity: float = 1.0,
    correlation_aware: bool = True,
    skip_warmup_fights: int = 200,
) -> BacktestResult:
    """Run a chronological backtest over UFC fights.

    Expects a DataFrame produced by `ufc.data.load_fights`. See that module
    for the expected schema.
    """
    sys = UfcEloSystem()
    result = BacktestResult()

    attr_cols_a = {
        "age": "a_age",
        "height_cm": "a_height_cm",
        "reach_cm": "a_reach_cm",
        "stance": "a_stance",
        "slpm": "a_slpm",
        "str_acc": "a_str_acc",
        "sapm": "a_sapm",
        "str_def": "a_str_def",
        "td_avg": "a_td_avg",
        "td_acc": "a_td_acc",
        "td_def": "a_td_def",
        "sub_avg": "a_sub_avg",
    }
    attr_cols_b = {k: v.replace("a_", "b_", 1) for k, v in attr_cols_a.items()}

    for i, row in enumerate(fights.itertuples(index=False)):
        fight_date = _as_date(row.fight_date)
        a_id = row.fighter_a_id
        b_id = row.fighter_b_id
        result_val = row.result  # "W" if A won, "L" if B won, "D" draw

        if i >= skip_warmup_fights and result_val != "D":
            a_profile = _profile_from_row(row, a_id, attr_cols_a)
            b_profile = _profile_from_row(row, b_id, attr_cols_b)
            pred = predict_fight(
                fighter_a=a_profile,
                fighter_b=b_profile,
                fight_date=fight_date,
                elo_system=sys,
                title_fight=bool(getattr(row, "title_fight", False)),
                sensitivity=sensitivity,
                correlation_aware=correlation_aware,
                a_name=row.fighter_a_name,
                b_name=row.fighter_b_name,
            )
            outcome_b = 1 if result_val == "L" else 0
            result.rows.append(
                BacktestRow(
                    event_id=f"{fight_date.isoformat()}_{a_id}_vs_{b_id}",
                    event_date=fight_date,
                    outcome_a_name=row.fighter_a_name,
                    outcome_b_name=row.fighter_b_name,
                    model_probability_b=pred.fighter_b_win_probability,
                    actual_outcome_b=outcome_b,
                )
            )

        sys.record_fight(
            fighter_a_id=a_id,
            fighter_b_id=b_id,
            result=result_val,
            fight_date=fight_date,
            title_fight=bool(getattr(row, "title_fight", False)),
        )

    return result


def _profile_from_row(row, fighter_id: str, col_map: dict[str, str]) -> FighterProfile:
    kwargs: dict[str, object] = {"fighter_id": fighter_id}
    for field_name, col in col_map.items():
        value = getattr(row, col, None)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        kwargs[field_name] = value
    return FighterProfile(**kwargs)  # type: ignore[arg-type]


def _as_date(v) -> date:
    if isinstance(v, date):
        return v
    return pd.to_datetime(v).date()
