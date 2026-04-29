"""Per-fighter rolling-window stats from the parsed ufcstats parquet.

Replaces the career-average numbers in the Kaggle CSV. Given a fighter and an
`as_of` date, take their last N UFC fights strictly before that date and
compute per-minute / per-attempt rates. This is what features.py actually wants.

Loaded once, queried many times — the DataFrame is pre-indexed on fighter_hash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from predictor.ufc.features import FighterProfile

_PARQUET = Path("data/ufcstats/fights.parquet")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


@dataclass(frozen=True)
class WindowedStats:
    fighter_hash: str
    fighter_name: str
    n_fights: int
    total_time_sec: int
    # striking
    slpm: float | None       # sig strikes landed per minute
    str_acc: float | None    # sig_str_landed / sig_str_att
    sapm: float | None       # opp sig strikes landed per minute
    str_def: float | None    # 1 - opp_sig_str_landed / opp_sig_str_att
    # grappling
    td_avg: float | None     # takedowns per 15 min
    td_acc: float | None
    td_def: float | None
    sub_avg: float | None    # submission attempts per 15 min


@lru_cache(maxsize=1)
def load_fights(path: str | None = None) -> pd.DataFrame:
    """Cached read of the parsed parquet. Returns a copy-safe indexed frame."""
    p = Path(path) if path else _PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run scripts/parse_ufcstats.py (which depends on scrape_ufcstats.py)"
        )
    df = pd.read_parquet(p)
    df["fight_date"] = pd.to_datetime(df["fight_date"]).dt.date
    # Name lookup column (normalized).
    df["_name_key"] = df["fighter_name"].astype(str).map(_norm)
    return df


@lru_cache(maxsize=1)
def _name_to_hash(path: str | None = None) -> dict[str, str]:
    """Normalized-name -> most-recent fighter_hash mapping."""
    df = load_fights(path)
    # Pick the most-recent hash seen for each normalized name.
    latest = df.sort_values("fight_date").drop_duplicates("_name_key", keep="last")
    return dict(zip(latest["_name_key"], latest["fighter_hash"]))


def resolve_hash(name_or_hash: str, path: str | None = None) -> str | None:
    """Accept a hash or a name; return fighter_hash if known."""
    v = (name_or_hash or "").strip().lower()
    if re.fullmatch(r"[a-f0-9]{16}", v):
        return v
    return _name_to_hash(path).get(_norm(name_or_hash))


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def windowed_stats(
    fighter: str,
    as_of: date | None = None,
    n: int = 10,
    min_fights: int = 3,
    path: str | None = None,
) -> WindowedStats | None:
    """Last-N-fights aggregates for a fighter, strictly before `as_of`.

    Returns None if the fighter is unknown or has fewer than `min_fights`
    recorded fights. Fields may individually be None when denominators are 0
    (e.g. `td_def` is None if the fighter has never defended a takedown).
    """
    h = resolve_hash(fighter, path)
    if h is None:
        return None

    df = load_fights(path)
    sub = df[df["fighter_hash"] == h]
    if as_of is not None:
        sub = sub[sub["fight_date"] < as_of]
    if len(sub) < min_fights:
        return None

    sub = sub.sort_values("fight_date").tail(n)

    time_sec = int(sub["total_fight_time_sec"].sum())
    minutes = time_sec / 60.0
    fifteens = time_sec / 900.0  # per-15-minute unit

    ssl = int(sub["sig_str_landed"].sum())
    ssa = int(sub["sig_str_att"].sum())
    opp_ssl = int(sub["opp_sig_str_landed"].sum())
    opp_ssa = int(sub["opp_sig_str_att"].sum())
    tdl = int(sub["td_landed"].sum())
    tda = int(sub["td_att"].sum())
    opp_tdl = int(sub["opp_td_landed"].sum())
    opp_tda = int(sub["opp_td_att"].sum())
    suba = int(sub["sub_att"].sum())

    return WindowedStats(
        fighter_hash=h,
        fighter_name=str(sub.iloc[-1]["fighter_name"]),
        n_fights=len(sub),
        total_time_sec=time_sec,
        slpm=_safe_div(ssl, minutes),
        str_acc=_safe_div(ssl, ssa),
        sapm=_safe_div(opp_ssl, minutes),
        str_def=(None if opp_ssa <= 0 else 1.0 - (opp_ssl / opp_ssa)),
        td_avg=_safe_div(tdl, fifteens),
        td_acc=_safe_div(tdl, tda),
        td_def=(None if opp_tda <= 0 else 1.0 - (opp_tdl / opp_tda)),
        sub_avg=_safe_div(suba, fifteens),
    )


def to_profile_overrides(ws: WindowedStats) -> dict:
    """Return the kwargs subset of FighterProfile that windowed stats can fill."""
    return {
        "slpm": ws.slpm,
        "str_acc": ws.str_acc,
        "sapm": ws.sapm,
        "str_def": ws.str_def,
        "td_avg": ws.td_avg,
        "td_acc": ws.td_acc,
        "td_def": ws.td_def,
        "sub_avg": ws.sub_avg,
    }


def apply_to_profile(
    profile: FighterProfile,
    ws: WindowedStats | None,
    gap_fill_only: bool = True,
) -> FighterProfile:
    """Merge windowed stats into a profile.

    `gap_fill_only=True` (default): windowed values fill *missing* fields only.
    The Kaggle CSV provides smooth career averages for slpm/str_acc/td_avg/
    td_acc/sub_avg (large-n) but is missing sapm/str_def/td_def (None). With
    this mode, windowed only fills those gaps — the most useful change without
    introducing small-sample noise on already-covered fields.

    `gap_fill_only=False`: windowed overrides Kaggle wherever it has data.
    """
    if ws is None:
        return profile
    candidates = to_profile_overrides(ws)
    if gap_fill_only:
        overrides = {
            k: v
            for k, v in candidates.items()
            if v is not None and getattr(profile, k, None) is None
        }
    else:
        overrides = {k: v for k, v in candidates.items() if v is not None}
    if not overrides:
        return profile
    data = profile.__dict__.copy()
    data.update(overrides)
    return FighterProfile(**data)
