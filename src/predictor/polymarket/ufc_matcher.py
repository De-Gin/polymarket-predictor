"""Extract fight moneyline markets from Polymarket UFC events.

A UFC card on Polymarket is one Event with N sub-markets. The moneyline market
has outcomes equal to the two fighters' names (not "Yes"/"No"). All other
sub-markets (method of victory, O/U rounds, etc.) are ignored here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from predictor.polymarket.client import PolymarketEvent, PolymarketMarket

# Outcomes that indicate a prop bet rather than a moneyline.
_NON_MONEYLINE = {
    frozenset(["yes", "no"]),
    frozenset(["over", "under"]),
    frozenset(["draw", "no draw"]),
}


@dataclass(frozen=True)
class FightMarket:
    event_title: str
    event_slug: str
    fight_date: date | None
    fighter_a_name: str
    fighter_b_name: str
    price_a: float  # mid price for fighter A
    price_b: float  # mid price for fighter B
    question: str
    volume: float = 0.0
    liquidity: float = 0.0
    spread: float | None = None  # bid-ask spread on the book (outcome[0])
    ask_a: float | None = None  # price you'd actually PAY to buy fighter A YES
    ask_b: float | None = None  # price you'd actually PAY to buy fighter B YES


def is_moneyline(market: PolymarketMarket) -> bool:
    if len(market.outcomes) != 2:
        return False
    key = frozenset(o.strip().lower() for o in market.outcomes)
    return key not in _NON_MONEYLINE


_VS_SPLIT = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


def _extract_from_question(question: str) -> tuple[str, str] | None:
    """Fallback: parse 'Event prefix: A vs. B (...)' → (A, B)."""
    q = question
    # Strip anything in parentheses at the end.
    q = re.sub(r"\s*\([^)]*\)\s*$", "", q).strip()
    # Drop everything before a colon if present.
    if ":" in q:
        q = q.split(":", 1)[1].strip()
    parts = _VS_SPLIT.split(q, maxsplit=1)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if not a or not b:
        return None
    return a, b


def to_fight_market(event: PolymarketEvent, market: PolymarketMarket) -> FightMarket | None:
    if not is_moneyline(market):
        return None

    # Prefer the outcomes list — it's cleaner than parsing the question.
    a_name, b_name = market.outcomes[0].strip(), market.outcomes[1].strip()
    # Guard: if outcomes look like generic placeholders, try the question.
    if a_name.lower() in {"fighter a", "fighter b"} or not a_name or not b_name:
        parsed = _extract_from_question(market.question)
        if parsed is None:
            return None
        a_name, b_name = parsed

    price_a, price_b = market.prices[0], market.prices[1]
    # Prefer the market-level gameStartTime (actual fight start) over the event
    # end_date (which is often end-of-day on the card).
    start = market.game_start_time or event.end_date
    fight_date = start.date() if start else None

    return FightMarket(
        event_title=event.title,
        event_slug=event.slug,
        fight_date=fight_date,
        fighter_a_name=a_name,
        fighter_b_name=b_name,
        price_a=price_a,
        price_b=price_b,
        question=market.question,
        volume=market.volume,
        liquidity=market.liquidity,
        spread=market.spread,
        ask_a=market.ask_for_index(0),
        ask_b=market.ask_for_index(1),
    )


def extract_fight_markets(events: list[PolymarketEvent]) -> list[FightMarket]:
    """Find all moneyline fight markets across a list of UFC events."""
    out: list[FightMarket] = []
    for e in events:
        for m in e.markets:
            fm = to_fight_market(e, m)
            if fm is not None:
                out.append(fm)
    return out
