"""Adapt the Kaggle 'Ultimate UFC Dataset' (ufc-master.csv) into our schema.

Source columns (Kaggle):     R_* / B_* corner stats, `Winner` in {Red, Blue, Draw, No Contest}.
Target columns (our schema): fighter_a_* / fighter_b_*, `result` in {W, L, D}.

Convention: Red corner -> fighter_a, Blue corner -> fighter_b.

Usage:
    python scripts/adapt_ufc_kaggle.py data/ufc/ufc-master.csv data/ufc/fights.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip().lower()).strip("-")
    return s or "unknown"


def adapt(src: Path, dst: Path, seed: int = 42) -> None:
    df = pd.read_csv(src)

    # Drop No Contests — not usable as win/loss signal.
    df = df[df["Winner"].isin(["Red", "Blue", "Draw"])].copy().reset_index(drop=True)

    # UFC matchmakers pre-sort the favorite into the Red corner, so Red wins
    # ~58% of fights. If we always map Red->A, the model inherits that selection
    # bias as a free ~8% offset. Randomize A/B assignment per fight so the base
    # rate is ~50/50 and the model's own discrimination is what's measured.
    rng = np.random.default_rng(seed)
    swap = rng.random(len(df)) < 0.5

    def pick(red_col: str, blue_col: str) -> pd.Series:
        return np.where(swap, df[blue_col], df[red_col])

    out = pd.DataFrame()
    out["fight_date"] = df["date"]

    out["fighter_a_name"] = pick("R_fighter", "B_fighter")
    out["fighter_b_name"] = pick("B_fighter", "R_fighter")
    out["fighter_a_id"] = pd.Series(out["fighter_a_name"]).map(_slug)
    out["fighter_b_id"] = pd.Series(out["fighter_b_name"]).map(_slug)

    # Original result: "Red" = A won pre-swap. After swap, if we swapped, Red is now B.
    red_won = df["Winner"] == "Red"
    draw = df["Winner"] == "Draw"
    # After swap: A won iff (red_won & !swap) | (blue_won & swap)
    a_won = (red_won & ~swap) | ((~red_won & ~draw) & swap)
    out["result"] = np.where(draw, "D", np.where(a_won, "W", "L"))

    out["title_fight"] = df["title_bout"].fillna(False).astype(bool)
    out["weight_class"] = df.get("weight_class")

    # Physicals
    out["a_age"] = pick("R_age", "B_age")
    out["b_age"] = pick("B_age", "R_age")
    out["a_height_cm"] = pick("R_Height_cms", "B_Height_cms")
    out["b_height_cm"] = pick("B_Height_cms", "R_Height_cms")
    out["a_reach_cm"] = pick("R_Reach_cms", "B_Reach_cms")
    out["b_reach_cm"] = pick("B_Reach_cms", "R_Reach_cms")
    out["a_stance"] = pick("R_Stance", "B_Stance")
    out["b_stance"] = pick("B_Stance", "R_Stance")

    # Striking / grappling stats (Kaggle stats are per-fight averages; divide
    # SLpM by 15 for a rough per-minute proxy — diffs remain meaningful).
    # SApM / str_def / td_def aren't in the dataset → left NaN, features.py
    # skips the striking block cleanly.
    out["a_slpm"] = pick("R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed") / 15.0
    out["b_slpm"] = pick("B_avg_SIG_STR_landed", "R_avg_SIG_STR_landed") / 15.0
    out["a_str_acc"] = pick("R_avg_SIG_STR_pct", "B_avg_SIG_STR_pct")
    out["b_str_acc"] = pick("B_avg_SIG_STR_pct", "R_avg_SIG_STR_pct")
    out["a_sapm"] = pd.NA
    out["b_sapm"] = pd.NA
    out["a_str_def"] = pd.NA
    out["b_str_def"] = pd.NA

    out["a_td_avg"] = pick("R_avg_TD_landed", "B_avg_TD_landed")
    out["b_td_avg"] = pick("B_avg_TD_landed", "R_avg_TD_landed")
    out["a_td_acc"] = pick("R_avg_TD_pct", "B_avg_TD_pct")
    out["b_td_acc"] = pick("B_avg_TD_pct", "R_avg_TD_pct")
    out["a_td_def"] = pd.NA
    out["b_td_def"] = pd.NA
    out["a_sub_avg"] = pick("R_avg_SUB_ATT", "B_avg_SUB_ATT")
    out["b_sub_avg"] = pick("B_avg_SUB_ATT", "R_avg_SUB_ATT")

    # Sort ascending so the backtest harness sees history in order.
    out = out.sort_values("fight_date").reset_index(drop=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"Wrote {len(out)} fights to {dst}")
    print(f"Date range: {out['fight_date'].min()} -> {out['fight_date'].max()}")
    print(f"Result distribution: {out['result'].value_counts().to_dict()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: adapt_ufc_kaggle.py <source.csv> <dest.csv>")
        sys.exit(1)
    adapt(Path(sys.argv[1]), Path(sys.argv[2]))
