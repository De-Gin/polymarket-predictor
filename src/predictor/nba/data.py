"""NBA data loading via nba_api with on-disk caching.

The nba_api package hits stats.nba.com's public endpoints. Those endpoints are
slow and rate-limited, so we cache every response as parquet in `data/cache/`.

Data returned is a pandas DataFrame with one row per team-game (two rows per
actual game — one for each team). The `merge_to_games` helper collapses that
into one row per game with home/away columns.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Project-relative cache directory
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "nba"


def _cache_path(season: str) -> Path:
    return CACHE_DIR / f"games_{season}.parquet"


def fetch_season_games(
    season: str = "2024-25",
    season_type: str = "Regular Season",
    use_cache: bool = True,
    sleep_seconds: float = 0.6,
) -> pd.DataFrame:
    """Fetch all games for a season. Returns one row per team-game.

    Parameters
    ----------
    season:
        NBA season identifier like "2023-24".
    season_type:
        "Regular Season" or "Playoffs".
    use_cache:
        When True, read from disk cache if present. Set False to force refresh.
    sleep_seconds:
        Courtesy delay before the API call. stats.nba.com rate-limits aggressive clients.

    Returns
    -------
    DataFrame with columns:
        GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, MATCHUP, WL, PTS, ...
    """
    cache = _cache_path(season)
    if use_cache and cache.exists():
        logger.info("Loading cached games for %s from %s", season, cache)
        return pd.read_parquet(cache)

    # Lazy import — nba_api is heavy and not needed if the cache is warm
    from nba_api.stats.endpoints import leaguegamelog  # type: ignore[import-not-found]

    time.sleep(sleep_seconds)
    logger.info("Fetching %s %s from stats.nba.com...", season, season_type)
    resp = leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star=season_type, player_or_team_abbreviation="T"
    )
    df = resp.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    logger.info("Cached %d team-game rows to %s", len(df), cache)
    return df


def merge_to_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Collapse two rows per game (one per team) into one row per game.

    Output columns:
        GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_PTS, AWAY_PTS, HOME_WON
    """
    required = {"GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS", "WL"}
    missing = required - set(team_games.columns)
    if missing:
        raise ValueError(f"team_games missing columns: {missing}")

    # MATCHUP format: "LAL vs. BOS" for home team, "LAL @ BOS" for away team
    team_games = team_games.copy()
    team_games["IS_HOME"] = team_games["MATCHUP"].str.contains(" vs. ", regex=False)

    home = team_games[team_games["IS_HOME"]].rename(
        columns={"TEAM_ABBREVIATION": "HOME_TEAM", "PTS": "HOME_PTS", "WL": "HOME_WL"}
    )
    away = team_games[~team_games["IS_HOME"]].rename(
        columns={"TEAM_ABBREVIATION": "AWAY_TEAM", "PTS": "AWAY_PTS"}
    )

    merged = home.merge(
        away[["GAME_ID", "AWAY_TEAM", "AWAY_PTS"]],
        on="GAME_ID",
        how="inner",
    )
    merged["HOME_WON"] = merged["HOME_WL"] == "W"
    merged = merged[["GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_PTS", "AWAY_PTS", "HOME_WON"]]
    return merged.sort_values("GAME_DATE").reset_index(drop=True)


def games_before(games: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Filter to games played strictly before `cutoff`. Useful for backtesting."""
    return games[games["GAME_DATE"] < cutoff].copy()


def games_on(games: pd.DataFrame, on_date: date) -> pd.DataFrame:
    """Filter to games played on `on_date`."""
    return games[games["GAME_DATE"] == on_date].copy()
