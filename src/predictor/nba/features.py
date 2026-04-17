"""Convert NBA team state into a list of Vectors for a single game prediction.

Each vector carries a `group` tag so the VectorSpace can downweight correlated
signals. Groups here:

    team_strength  — Elo diff, season record, recent form (highly correlated;
                     under correlation_aware aggregation these collectively
                     contribute at most ±1 unit of force)
    schedule       — rest days, back-to-back flag, travel
    venue          — home court factor
    momentum       — L10 form (separate-ish from season-long Elo)

Signed-magnitude convention: POSITIVE favors the AWAY team (outcome B).
The home team is always outcome A in our Event.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from predictor.core.vector import Vector
from predictor.nba.elo import HOME_COURT_ELO, NbaEloSystem


@dataclass
class GameContext:
    """Everything we need to know pre-game to build vectors."""

    game_date: date
    home_team: str
    away_team: str
    neutral_site: bool = False


def extract_game_vectors(
    ctx: GameContext,
    elo_system: NbaEloSystem,
    reference_date: date | None = None,
) -> list[Vector]:
    """Build the feature vectors for one game.

    Parameters
    ----------
    ctx:
        Matchup + date.
    elo_system:
        Elo system already updated with all games prior to `ctx.game_date`.
    reference_date:
        Used for freshness decay. Defaults to `ctx.game_date` (signals are "as of now").
    """
    ref = reference_date or ctx.game_date
    home_state = elo_system.get_state(ctx.home_team)
    away_state = elo_system.get_state(ctx.away_team)

    vectors: list[Vector] = []

    # --- Elo differential (the dominant signal) ------------------------------
    # Convert Elo diff to a [-1, 1] score. 400 Elo ≈ 10:1 odds — cap effective
    # signal around ±200 Elo for a single vector.
    elo_diff = away_state.rating - home_state.rating  # positive favors away
    vectors.append(
        Vector.from_signed_score(
            source_id="elo_diff",
            source_name=f"Elo diff ({away_state.rating:.0f} away vs {home_state.rating:.0f} home)",
            category="elo_rating",
            group="team_strength",
            data_date=ref,
            raw_score=elo_diff,
            scale=1 / 200.0,
            reliability=0.90,
            notes=f"away_elo={away_state.rating:.1f} home_elo={home_state.rating:.1f}",
        )
    )

    # --- Recent form (L10 win rate centered on 0.5) --------------------------
    away_form = away_state.form_score(window=10)
    home_form = home_state.form_score(window=10)
    form_diff = away_form - home_form  # in [-1, +1]
    if len(away_state.recent_results) >= 3 and len(home_state.recent_results) >= 3:
        vectors.append(
            Vector.from_signed_score(
                source_id="recent_form",
                source_name=f"L10 form (away {away_form:+.2f} vs home {home_form:+.2f})",
                category="l10_form",
                group="momentum",
                data_date=ref,
                raw_score=form_diff,
                scale=1.0,
                reliability=0.55,
                notes=f"away_l10={away_state.recent_results[:10]} home_l10={home_state.recent_results[:10]}",
            )
        )

    # --- Home court advantage ------------------------------------------------
    if not ctx.neutral_site:
        # Home advantage favors A (home) → negative signed magnitude
        vectors.append(
            Vector.from_signed_score(
                source_id="home_court",
                source_name="Home court advantage",
                category="venue",
                group="venue",
                data_date=ref,
                raw_score=-HOME_COURT_ELO / 200.0,  # ~-0.4 → tanh(-0.4) ≈ -0.38
                scale=1.0,
                reliability=0.85,
                notes="standard NBA home court Elo adjustment",
            )
        )

    # --- Rest advantage ------------------------------------------------------
    home_rest = home_state.days_rest(ctx.game_date)
    away_rest = away_state.days_rest(ctx.game_date)
    if home_rest is not None and away_rest is not None:
        # Cap rest at 5 days — beyond that, effect plateaus
        home_rest_c = min(home_rest, 5)
        away_rest_c = min(away_rest, 5)
        rest_diff = away_rest_c - home_rest_c  # positive favors away
        if rest_diff != 0:
            vectors.append(
                Vector.from_signed_score(
                    source_id="rest_diff",
                    source_name=f"Rest diff (away {away_rest}d vs home {home_rest}d)",
                    category="rest",
                    group="schedule",
                    data_date=ref,
                    raw_score=rest_diff,
                    scale=0.2,  # 1 day diff → ~0.2 force before reliability
                    reliability=0.45,
                )
            )

    # --- Back-to-back penalty ------------------------------------------------
    # B2B: team is playing on 1 day of rest
    home_b2b = home_rest is not None and home_rest <= 1
    away_b2b = away_rest is not None and away_rest <= 1
    if home_b2b != away_b2b:
        # One team is on B2B and the other isn't
        raw = 0.4 if home_b2b else -0.4  # home_b2b → favors away (positive)
        vectors.append(
            Vector.from_signed_score(
                source_id="b2b",
                source_name="Back-to-back disadvantage",
                category="b2b",
                group="schedule",
                data_date=ref,
                raw_score=raw,
                scale=1.0,
                reliability=0.60,
            )
        )

    return vectors
