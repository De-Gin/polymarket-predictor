"""Polymarket Gamma API client.

Gamma is the public read-only API for market metadata and prices. The CLOB
(central limit order book) endpoint is separate and needed for placing orders,
which we don't do here.

Base: https://gamma-api.polymarket.com/markets

We keep the surface minimal: fetch active markets, parse the subset of fields
we need, return dataclasses. Errors surface as requests exceptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class PolymarketMarket:
    id: str
    question: str
    slug: str
    end_date: datetime | None
    outcomes: list[str]  # e.g. ["Yes", "No"] or ["Fighter A", "Fighter B"]
    prices: list[float]  # aligned with outcomes; each in [0, 1] — mid prices
    volume: float
    liquidity: float
    category: str | None
    tags: list[str]
    # Orderbook — both sides refer to outcome[0]. outcome[1] is the complement:
    #   outcome[1].bid = 1 - outcome[0].ask, outcome[1].ask = 1 - outcome[0].bid.
    best_bid: float | None  # highest buy price for outcome[0]
    best_ask: float | None  # lowest sell price for outcome[0]
    spread: float | None  # bestAsk - bestBid for outcome[0]
    last_trade_price: float | None
    game_start_time: datetime | None  # actual event start (preferable to end_date)

    def price_for(self, outcome: str) -> float | None:
        for o, p in zip(self.outcomes, self.prices):
            if o.strip().lower() == outcome.strip().lower():
                return p
        return None

    def ask_for_index(self, idx: int) -> float | None:
        """Return the ask (execution cost to BUY YES) for outcome at `idx`."""
        if self.best_bid is None or self.best_ask is None:
            return None
        if idx == 0:
            return self.best_ask
        if idx == 1:
            return round(1.0 - self.best_bid, 6)
        return None


def _parse_json_field(raw) -> list:
    """Gamma returns outcomes/prices as JSON-encoded strings inside the JSON object."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _parse_date(raw) -> datetime | None:
    if not raw:
        return None
    try:
        # Gamma uses ISO 8601 with Z suffix.
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _opt_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_market(m: dict) -> PolymarketMarket | None:
    outcomes = _parse_json_field(m.get("outcomes"))
    prices_raw = _parse_json_field(m.get("outcomePrices"))
    if not outcomes or not prices_raw or len(outcomes) != len(prices_raw):
        return None
    try:
        prices = [float(p) for p in prices_raw]
    except (TypeError, ValueError):
        return None

    tags_raw = m.get("tags") or []
    tags: list[str] = []
    for t in tags_raw:
        if isinstance(t, dict):
            label = t.get("label") or t.get("slug")
            if label:
                tags.append(str(label))
        elif isinstance(t, str):
            tags.append(t)

    # Prefer numeric fields where Gamma exposes both string + numeric variants.
    volume = _opt_float(m.get("volumeNum")) or _opt_float(m.get("volume")) or 0.0
    liquidity = _opt_float(m.get("liquidityNum")) or _opt_float(m.get("liquidity")) or 0.0

    return PolymarketMarket(
        id=str(m.get("id", "")),
        question=str(m.get("question", "")),
        slug=str(m.get("slug", "")),
        end_date=_parse_date(m.get("endDate")),
        outcomes=[str(o) for o in outcomes],
        prices=prices,
        volume=volume,
        liquidity=liquidity,
        category=m.get("category"),
        tags=tags,
        best_bid=_opt_float(m.get("bestBid")),
        best_ask=_opt_float(m.get("bestAsk")),
        spread=_opt_float(m.get("spread")),
        last_trade_price=_opt_float(m.get("lastTradePrice")),
        game_start_time=_parse_date(m.get("gameStartTime")),
    )


@dataclass(frozen=True)
class PolymarketEvent:
    id: str
    title: str
    slug: str
    end_date: datetime | None
    markets: list[PolymarketMarket]


def fetch_events_by_tag(
    tag_slug: str, limit: int = 500, timeout: float = 20.0
) -> list[PolymarketEvent]:
    """Fetch events for a tag (e.g. 'ufc', 'nba'). Sports markets are grouped here."""
    resp = requests.get(
        f"{GAMMA_BASE}/events",
        params={
            "active": "true",
            "closed": "false",
            "limit": limit,
            "tag_slug": tag_slug,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    out: list[PolymarketEvent] = []
    for e in data:
        markets: list[PolymarketMarket] = []
        for m in e.get("markets") or []:
            parsed = _parse_market(m)
            if parsed is not None:
                markets.append(parsed)
        out.append(
            PolymarketEvent(
                id=str(e.get("id", "")),
                title=str(e.get("title") or "").strip(),
                slug=str(e.get("slug") or ""),
                end_date=_parse_date(e.get("endDate")),
                markets=markets,
            )
        )
    return out


def fetch_active_markets(
    limit: int = 500,
    search: str | None = None,
    timeout: float = 20.0,
) -> list[PolymarketMarket]:
    """Fetch active, open markets. `search` does a server-side keyword filter."""
    params: dict[str, object] = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "endDate",
        "ascending": "true",
    }
    if search:
        params["search"] = search

    resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    out: list[PolymarketMarket] = []
    for m in data:
        parsed = _parse_market(m)
        if parsed is not None:
            out.append(parsed)
    return out


def filter_by_keywords(
    markets: Iterable[PolymarketMarket], keywords: Iterable[str]
) -> list[PolymarketMarket]:
    """Return markets whose question/category/tags contain any of the keywords (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    out: list[PolymarketMarket] = []
    for m in markets:
        haystack = " ".join(
            [
                m.question.lower(),
                (m.category or "").lower(),
                " ".join(t.lower() for t in m.tags),
            ]
        )
        if any(k in haystack for k in kws):
            out.append(m)
    return out
