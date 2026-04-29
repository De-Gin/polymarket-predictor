"""Build a `FighterProfile` from the historical fights CSV by name lookup.

The CSV (from `scripts/adapt_ufc_kaggle.py`) stores, per fight, the attributes
for both corners. To predict a new fight, we take the most-recent row that
contains the fighter (either corner) as of some cutoff date and lift their
attributes.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd

from predictor.ufc.features import FighterProfile


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


_FIELDS = (
    "age",
    "height_cm",
    "reach_cm",
    "stance",
    "slpm",
    "str_acc",
    "sapm",
    "str_def",
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
)


def _row_profile(row: pd.Series, side: str, fighter_id: str) -> FighterProfile:
    kwargs: dict[str, Any] = {"fighter_id": fighter_id}
    for f in _FIELDS:
        col = f"{side}_{f}"
        if col not in row.index:
            continue
        v = row[col]
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        kwargs[f] = v
    return FighterProfile(**kwargs)  # type: ignore[arg-type]


def latest_profile(
    fights: pd.DataFrame, fighter_name: str, as_of: date | None = None
) -> FighterProfile | None:
    """Return the fighter's most recent profile from history up to `as_of`.

    Returns None if the fighter is not found in the CSV at all.
    """
    target = _norm(fighter_name)
    df = fights
    if as_of is not None:
        df = df[df["fight_date"] <= as_of]
    if df.empty:
        return None

    # Match by normalized name against either corner.
    a_matches = df["fighter_a_name"].astype(str).map(_norm) == target
    b_matches = df["fighter_b_name"].astype(str).map(_norm) == target

    # Most recent matching row, preferring whichever corner contains them.
    rows = df[a_matches | b_matches]
    if rows.empty:
        return None

    # We want the latest fight containing this fighter.
    last = rows.sort_values("fight_date").iloc[-1]

    if _norm(str(last["fighter_a_name"])) == target:
        return _row_profile(last, side="a", fighter_id=str(last["fighter_a_id"]))
    return _row_profile(last, side="b", fighter_id=str(last["fighter_b_id"]))


def overlay_windowed(
    profile: FighterProfile | None,
    fighter_name: str,
    as_of: date | None = None,
    n: int = 10,
) -> FighterProfile | None:
    """Enrich a profile with rolling-window stats from the ufcstats parquet.

    No-op if:
      - profile is None,
      - the ufcstats parquet doesn't exist (never scraped/parsed),
      - the fighter isn't in the parquet or has too few recent fights.

    Import is local so modules can be used without the parquet present.
    """
    if profile is None:
        return None
    try:
        from predictor.ufc.windowed_stats import (
            apply_to_profile,
            windowed_stats,
        )
    except Exception:
        return profile
    try:
        ws = windowed_stats(fighter_name, as_of=as_of, n=n)
    except FileNotFoundError:
        return profile
    return apply_to_profile(profile, ws)


def fight_count(fights: pd.DataFrame, fighter_name: str, as_of: date | None = None) -> int:
    """Return the number of fights the fighter appears in, up to `as_of` if given."""
    target = _norm(fighter_name)
    df = fights
    if as_of is not None:
        df = df[df["fight_date"] <= as_of]
    if df.empty:
        return 0
    a = (df["fighter_a_name"].astype(str).map(_norm) == target).sum()
    b = (df["fighter_b_name"].astype(str).map(_norm) == target).sum()
    return int(a + b)
