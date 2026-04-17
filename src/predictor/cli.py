"""Command-line interface for the sports predictor.

Usage examples:

    sports-predictor nba backtest --season 2023-24
    sports-predictor nba predict BOS LAL --season 2024-25 --date 2025-04-20
    sports-predictor nba edge BOS LAL --market 0.58 --season 2024-25 --date 2025-04-20
    sports-predictor ufc backtest --csv data/fights.csv

The `predict` and `edge` commands work by re-training Elo on cached season
games up to (but not including) the target date, then predicting. The first
run for a season will hit stats.nba.com and cache a parquet file; subsequent
runs are instant.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from predictor.eval.backtest import backtest_nba_season, backtest_ufc
from predictor.eval.brier import brier_score, brier_skill_score
from predictor.eval.calibration import calibration_report
from predictor.eval.edge import edge_vs_market
from predictor.nba.data import fetch_season_games, games_before, merge_to_games
from predictor.nba.elo import NbaEloSystem
from predictor.nba.predict import predict_game
from predictor.ufc.data import load_fights

app = typer.Typer(
    help="Vector-based prediction engine for NBA + UFC, targeting Polymarket edge.",
    no_args_is_help=True,
    add_completion=False,
)

nba_app = typer.Typer(help="NBA game predictions and backtests.", no_args_is_help=True)
ufc_app = typer.Typer(help="UFC fight backtests.", no_args_is_help=True)
app.add_typer(nba_app, name="nba")
app.add_typer(ufc_app, name="ufc")

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as e:
        raise typer.BadParameter(f"Dates must be YYYY-MM-DD, got {value!r}") from e


def _train_nba_elo_to(as_of: date, season: str) -> NbaEloSystem:
    """Fetch season games and train an Elo system with all games strictly before `as_of`."""
    raw = fetch_season_games(season)
    games = merge_to_games(raw)
    prior = games_before(games, as_of)

    if prior.empty:
        console.print(
            f"[yellow]Warning:[/yellow] no games in season {season} before {as_of}. "
            f"Predictions will use default Elo ratings (1500 for everyone)."
        )

    sys = NbaEloSystem()
    for row in prior.itertuples(index=False):
        sys.record_game(
            home_id=row.HOME_TEAM,
            away_id=row.AWAY_TEAM,
            home_score=int(row.HOME_PTS),
            away_score=int(row.AWAY_PTS),
            game_date=row.GAME_DATE,
        )
    return sys


def _print_prediction(pred, home: str, away: str) -> None:
    """Render a prediction with a rich table + the explainability block."""
    tbl = Table(title=f"NBA Prediction — {away} @ {home}", show_header=True, box=None)
    tbl.add_column("Team", style="bold")
    tbl.add_column("Win probability", justify="right")
    tbl.add_row(home, f"{pred.home_win_probability * 100:.1f}%")
    tbl.add_row(away, f"{pred.away_win_probability * 100:.1f}%")
    console.print(tbl)
    console.print()
    console.print(pred.summary.readable(home, away))


# ---------------------------------------------------------------------------
# NBA commands
# ---------------------------------------------------------------------------


@nba_app.command("predict")
def nba_predict(
    home: str = typer.Argument(..., help="Home team abbreviation, e.g. BOS"),
    away: str = typer.Argument(..., help="Away team abbreviation, e.g. LAL"),
    season: str = typer.Option("2024-25", help="NBA season identifier, e.g. 2023-24"),
    date_str: str = typer.Option(
        None, "--date", help="Game date YYYY-MM-DD (defaults to today)"
    ),
    sensitivity: float = typer.Option(1.0, help="Sigmoid sensitivity (tune via backtest)"),
    naive: bool = typer.Option(
        False, help="Disable correlation-aware aggregation (baseline mode)"
    ),
) -> None:
    """Predict a single NBA game."""
    game_date = _parse_date(date_str) if date_str else date.today()
    home = home.upper()
    away = away.upper()

    elo = _train_nba_elo_to(game_date, season)
    pred = predict_game(
        home_team=home,
        away_team=away,
        game_date=game_date,
        elo_system=elo,
        sensitivity=sensitivity,
        correlation_aware=not naive,
    )
    _print_prediction(pred, home, away)


@nba_app.command("edge")
def nba_edge(
    home: str = typer.Argument(..., help="Home team abbreviation"),
    away: str = typer.Argument(..., help="Away team abbreviation"),
    market: float = typer.Option(
        ..., "--market", help="Polymarket price for the HOME team to win, in [0, 1]"
    ),
    season: str = typer.Option("2024-25", help="NBA season identifier"),
    date_str: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD"),
    sensitivity: float = typer.Option(1.0, help="Sigmoid sensitivity"),
    naive: bool = typer.Option(False, help="Disable correlation-aware aggregation"),
) -> None:
    """Compute edge vs a Polymarket price (expressed as home-team YES share price)."""
    if not 0.0 <= market <= 1.0:
        raise typer.BadParameter("--market must be in [0, 1]")

    game_date = _parse_date(date_str) if date_str else date.today()
    home = home.upper()
    away = away.upper()

    elo = _train_nba_elo_to(game_date, season)
    pred = predict_game(
        home_team=home,
        away_team=away,
        game_date=game_date,
        elo_system=elo,
        sensitivity=sensitivity,
        correlation_aware=not naive,
    )
    _print_prediction(pred, home, away)

    edge = edge_vs_market(
        outcome_name=f"{home} wins",
        model_probability=pred.home_win_probability,
        market_price=market,
    )
    console.print()
    color = "green" if edge.edge > 0 else "red"
    console.print(f"[bold {color}]{edge.readable()}[/bold {color}]")


@nba_app.command("backtest")
def nba_backtest(
    season: str = typer.Option("2023-24", help="NBA season identifier"),
    sensitivity: float = typer.Option(1.0, help="Sigmoid sensitivity"),
    naive: bool = typer.Option(False, help="Disable correlation-aware aggregation"),
    warmup: int = typer.Option(200, help="Games to skip at the start for Elo burn-in"),
) -> None:
    """Replay a full NBA season and score the model."""
    raw = fetch_season_games(season)
    games = merge_to_games(raw)
    console.print(f"Loaded {len(games)} games for {season}.")

    result = backtest_nba_season(
        games,
        sensitivity=sensitivity,
        correlation_aware=not naive,
        skip_warmup_games=warmup,
    )
    _report(result.probabilities, result.outcomes, label=f"NBA {season}")


# ---------------------------------------------------------------------------
# UFC commands
# ---------------------------------------------------------------------------


@ufc_app.command("backtest")
def ufc_backtest(
    csv: Path = typer.Option(..., help="Path to UFC fights CSV (see predictor/ufc/data.py)"),
    sensitivity: float = typer.Option(1.0, help="Sigmoid sensitivity"),
    naive: bool = typer.Option(False, help="Disable correlation-aware aggregation"),
    warmup: int = typer.Option(200, help="Fights to skip at the start for Elo burn-in"),
) -> None:
    """Replay a UFC fight history CSV and score the model."""
    if not csv.exists():
        raise typer.BadParameter(f"CSV not found: {csv}")

    fights = load_fights(csv)
    console.print(f"Loaded {len(fights)} fights from {csv}.")

    result = backtest_ufc(
        fights,
        sensitivity=sensitivity,
        correlation_aware=not naive,
        skip_warmup_fights=warmup,
    )
    _report(result.probabilities, result.outcomes, label=f"UFC ({csv.name})")


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------


def _report(probs: list[float], outcomes: list[int], label: str) -> None:
    if not probs:
        console.print(f"[red]No predictions produced for {label}.[/red]")
        raise typer.Exit(1)

    brier = brier_score(probs, outcomes)
    skill = brier_skill_score(probs, outcomes)
    cal = calibration_report(probs, outcomes)

    tbl = Table(title=f"Backtest — {label}", show_header=True)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Interpretation", style="dim")
    tbl.add_row("Predictions", f"{len(probs):,}", "")
    tbl.add_row("Base rate (outcome B)", f"{sum(outcomes) / len(outcomes):.3f}", "")
    tbl.add_row("Brier score", f"{brier:.4f}", "lower is better, 0.25 = random")
    tbl.add_row("Brier skill score", f"{skill:+.4f}", "positive = better than base rate")
    tbl.add_row("ECE", f"{cal.ece:.4f}", "< 0.05 is well-calibrated")
    console.print(tbl)
    console.print()
    console.print(cal.readable())


# ---------------------------------------------------------------------------
# Entrypoint for `python -m predictor.cli`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
