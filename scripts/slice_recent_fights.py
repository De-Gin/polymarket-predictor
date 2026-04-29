"""Filter the full UFC fights CSV down to fights on/after a cutoff date.

Used for backtest comparisons against the ufcstats windowed-stats overlay,
which only has data from ~April 2023 onward.

Usage:
    python scripts/slice_recent_fights.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = Path("data/ufc/fights.csv")
DST = Path("data/ufc/fights_recent.csv")
CUTOFF = "2023-04-01"


def main() -> int:
    df = pd.read_csv(SRC)
    df["fight_date"] = pd.to_datetime(df["fight_date"])
    recent = df[df["fight_date"] >= CUTOFF]
    recent.to_csv(DST, index=False)
    print(f"{len(df)} -> {len(recent)} rows >= {CUTOFF}")
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
