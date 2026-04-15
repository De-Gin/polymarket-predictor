"""UFC data loading from CSV.

There's no clean public UFC API. The practical options are:
1. Scrape ufcstats.com (brittle; don't recommend for v1).
2. Use a Kaggle dataset like "ufcdataset" or "ultimate-ufc-dataset".
3. Hand-maintain a CSV for recent fights.

This module expects a CSV with the columns below. A `fights_schema()` helper
prints the schema, and `load_fights()` normalizes the columns to expected types.

Expected CSV columns:
    fight_date         (YYYY-MM-DD)
    fighter_a_id       (stable id / slug)
    fighter_a_name
    fighter_b_id
    fighter_b_name
    result             ("W" = A won, "L" = B won, "D" = draw/NC)
    title_fight        (true/false)
    weight_class       (string)
    a_age, a_height_cm, a_reach_cm, a_stance
    b_age, b_height_cm, b_reach_cm, b_stance
    a_slpm, a_str_acc, a_sapm, a_str_def, a_td_avg, a_td_acc, a_td_def, a_sub_avg
    b_slpm, b_str_acc, b_sapm, b_str_def, b_td_avg, b_td_acc, b_td_def, b_sub_avg

Any of the stat columns may be NaN — feature extraction checks for nulls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "fight_date",
    "fighter_a_id",
    "fighter_a_name",
    "fighter_b_id",
    "fighter_b_name",
    "result",
)


def fights_schema() -> str:
    """Return the expected schema as a printable string."""
    return __doc__ or ""


def load_fights(path: Path | str) -> pd.DataFrame:
    """Load and normalize a UFC fights CSV.

    - Parses `fight_date` as a `date`.
    - Ensures required columns are present.
    - Sorts by `fight_date` ascending.
    """
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"UFC csv missing required columns: {sorted(missing)}")
    df["fight_date"] = pd.to_datetime(df["fight_date"]).dt.date
    if "title_fight" not in df.columns:
        df["title_fight"] = False
    df["title_fight"] = df["title_fight"].fillna(False).astype(bool)
    bad_results = set(df["result"].unique()) - {"W", "L", "D"}
    if bad_results:
        raise ValueError(f"'result' must be W/L/D; found: {bad_results}")
    return df.sort_values("fight_date").reset_index(drop=True)
