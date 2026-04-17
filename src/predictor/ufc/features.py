"""Turn UFC fighter state into Vectors for a fight prediction.

Convention: positive signed_magnitude favors fighter B (outcome B / challenger).

Correlation groups:
    skill         — Elo diff, form (dominant signal)
    physicality   — reach, height, age
    striking      — landed/min, accuracy, defense
    grappling     — takedown avg/acc/def, sub attempts
    schedule      — layoff, cage rust
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from predictor.core.vector import Vector
from predictor.ufc.elo import UfcEloSystem


@dataclass
class FighterProfile:
    """Attributes used for features. Any field may be None (missing)."""

    fighter_id: str
    age: float | None = None
    height_cm: float | None = None
    reach_cm: float | None = None
    stance: str | None = None
    # striking
    slpm: float | None = None  # significant strikes landed per minute
    str_acc: float | None = None  # [0, 1]
    sapm: float | None = None  # strikes absorbed per minute
    str_def: float | None = None  # [0, 1]
    # grappling
    td_avg: float | None = None  # takedowns landed per 15 min
    td_acc: float | None = None  # [0, 1]
    td_def: float | None = None  # [0, 1]
    sub_avg: float | None = None  # submission attempts per 15 min


@dataclass
class FightContext:
    fight_date: date
    fighter_a: FighterProfile
    fighter_b: FighterProfile
    title_fight: bool = False


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if math.isnan(a) or math.isnan(b):
        return None
    return b - a  # diff favoring B (matches our convention)


def extract_fight_vectors(ctx: FightContext, elo_system: UfcEloSystem) -> list[Vector]:
    """Build a list of vectors for one UFC fight prediction."""
    a = ctx.fighter_a
    b = ctx.fighter_b
    ref = ctx.fight_date

    vectors: list[Vector] = []
    a_state = elo_system.get_state(a.fighter_id)
    b_state = elo_system.get_state(b.fighter_id)

    # --- Elo differential (primary) ------------------------------------------
    elo_diff = b_state.rating - a_state.rating
    vectors.append(
        Vector.from_signed_score(
            source_id="elo_diff",
            source_name=f"Elo diff ({b_state.rating:.0f} vs {a_state.rating:.0f})",
            category="elo_rating",
            group="skill",
            data_date=ref,
            raw_score=elo_diff,
            scale=1 / 200.0,
            reliability=0.85,
        )
    )

    # --- Recent form --------------------------------------------------------
    if len(a_state.recent_results) >= 2 and len(b_state.recent_results) >= 2:
        form_diff = b_state.form_score(window=3) - a_state.form_score(window=3)
        vectors.append(
            Vector.from_signed_score(
                source_id="form_l3",
                source_name=f"Last-3 form diff ({form_diff:+.2f})",
                category="recent_form",
                group="skill",  # correlated with Elo — same group
                data_date=ref,
                raw_score=form_diff,
                scale=1.2,
                reliability=0.50,
            )
        )

    # --- Reach differential --------------------------------------------------
    reach_diff = _safe_diff(a.reach_cm, b.reach_cm)
    if reach_diff is not None and reach_diff != 0:
        # ~5 cm reach advantage is meaningful
        vectors.append(
            Vector.from_signed_score(
                source_id="reach_diff",
                source_name=f"Reach diff ({reach_diff:+.1f} cm)",
                category="reach",
                group="physicality",
                data_date=ref,
                raw_score=reach_diff,
                scale=1 / 10.0,
                reliability=0.55,
            )
        )

    # --- Age differential (older = disadvantage, past ~32 especially) --------
    if a.age is not None and b.age is not None:
        # Raw age diff favoring B = a.age - b.age (B is younger → positive)
        age_raw = a.age - b.age
        # Amplify when either fighter is past 32
        past_prime_penalty = max(0, a.age - 32) - max(0, b.age - 32)
        combined = age_raw + past_prime_penalty
        if abs(combined) > 0.1:
            vectors.append(
                Vector.from_signed_score(
                    source_id="age_diff",
                    source_name=f"Age diff (A={a.age:.0f}, B={b.age:.0f})",
                    category="age",
                    group="physicality",
                    data_date=ref,
                    raw_score=combined,
                    scale=1 / 5.0,  # 5-year gap → meaningful
                    reliability=0.55,
                )
            )

    # --- Striking differential ----------------------------------------------
    # Net striking = (slpm * str_acc) - (sapm * (1 - str_def))
    def _net_striking(p: FighterProfile) -> float | None:
        if None in (p.slpm, p.str_acc, p.sapm, p.str_def):
            return None
        return p.slpm * p.str_acc - p.sapm * (1 - p.str_def)  # type: ignore[operator]

    ns_a = _net_striking(a)
    ns_b = _net_striking(b)
    if ns_a is not None and ns_b is not None:
        striking_diff = ns_b - ns_a
        vectors.append(
            Vector.from_signed_score(
                source_id="striking_net",
                source_name=f"Net striking diff ({striking_diff:+.2f})",
                category="striking",
                group="striking",
                data_date=ref,
                raw_score=striking_diff,
                scale=0.6,
                reliability=0.60,
            )
        )

    # --- Grappling differential ---------------------------------------------
    if a.td_avg is not None and b.td_avg is not None:
        # Offense + defense combined
        a_off = a.td_avg * (a.td_acc or 0.3)
        b_off = b.td_avg * (b.td_acc or 0.3)
        a_score = a_off - (b_off * (1 - (a.td_def or 0.5)))
        b_score = b_off - (a_off * (1 - (b.td_def or 0.5)))
        grap_diff = b_score - a_score
        vectors.append(
            Vector.from_signed_score(
                source_id="grappling_net",
                source_name=f"Grappling diff ({grap_diff:+.2f})",
                category="grappling",
                group="grappling",
                data_date=ref,
                raw_score=grap_diff,
                scale=0.4,
                reliability=0.55,
            )
        )

    # --- Layoff / cage rust -------------------------------------------------
    a_layoff = a_state.days_since_last_fight(ctx.fight_date)
    b_layoff = b_state.days_since_last_fight(ctx.fight_date)
    if a_layoff is not None and b_layoff is not None:
        # Too short (< 60 days) = underprepared; too long (> 400 days) = rusty
        def _layoff_penalty(d: int) -> float:
            if d < 60:
                return -0.3  # underprepared
            if d > 400:
                return -0.2 * min(3, (d - 400) / 200.0)  # rust grows with time
            return 0.0

        a_pen = _layoff_penalty(a_layoff)
        b_pen = _layoff_penalty(b_layoff)
        # If A penalized more than B, favor B (positive)
        combined = a_pen - b_pen
        if abs(combined) > 0.05:
            vectors.append(
                Vector.from_signed_score(
                    source_id="layoff_diff",
                    source_name=f"Layoff (A={a_layoff}d, B={b_layoff}d)",
                    category="layoff",
                    group="schedule",
                    data_date=ref,
                    raw_score=combined,
                    scale=2.0,
                    reliability=0.40,
                )
            )

    return vectors
