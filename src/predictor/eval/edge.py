"""Compare model probabilities to market-implied probabilities (e.g. Polymarket).

Market price of a binary contract at $P → implied probability = P (in [0, 1]).
Edge = model_probability - market_probability. Positive edge → model thinks
the market underrates this outcome.

`kelly_fraction` computes the Kelly-optimal stake size given the model's
probability and the market's decimal odds. Use fractional Kelly (e.g. 0.25x)
in practice; full Kelly assumes your probability is exactly correct, which it
never is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolymarketEdge:
    outcome_name: str
    model_probability: float
    market_probability: float
    edge: float  # model - market
    expected_value_per_dollar: float  # EV of a $1 bet at market price
    kelly: float  # Kelly-optimal fraction of bankroll [0, 1]

    def readable(self) -> str:
        direction = "BUY" if self.edge > 0 else "PASS"
        return (
            f"{direction}  {self.outcome_name}: "
            f"model={self.model_probability * 100:.1f}%  "
            f"market={self.market_probability * 100:.1f}%  "
            f"edge={self.edge * 100:+.1f}pp  "
            f"EV=${self.expected_value_per_dollar:+.3f}/$  "
            f"Kelly={self.kelly * 100:.1f}%"
        )


def edge_vs_market(
    outcome_name: str,
    model_probability: float,
    market_price: float,
) -> PolymarketEdge:
    """Compute edge and EV.

    `market_price` is the current cost of a $1-payout YES share — i.e. the
    market's implied probability. On Polymarket, this is literally the
    outcome's current price divided by $1.
    """
    _validate_prob("model_probability", model_probability)
    _validate_prob("market_price", market_price)

    edge = model_probability - market_price
    # EV of buying a $1 YES at price `market_price`: you pay market_price,
    # receive $1 with model_probability, $0 otherwise.
    ev = model_probability * (1 - market_price) - (1 - model_probability) * market_price

    return PolymarketEdge(
        outcome_name=outcome_name,
        model_probability=model_probability,
        market_probability=market_price,
        edge=edge,
        expected_value_per_dollar=ev,
        kelly=kelly_fraction(model_probability, market_price),
    )


def kelly_fraction(model_probability: float, market_price: float) -> float:
    """Kelly-optimal fraction of bankroll to stake on YES at `market_price`.

    Formula for binary contract priced at p, model probability q:
        kelly = (q * (1 - p) - (1 - q) * p) / (1 - p)
              = (q - p) / (1 - p)
    Returns 0 when edge is negative (don't bet).
    """
    _validate_prob("model_probability", model_probability)
    _validate_prob("market_price", market_price)
    if market_price >= 1.0 - 1e-9:
        return 0.0
    k = (model_probability - market_price) / (1.0 - market_price)
    return max(0.0, min(1.0, k))


def _validate_prob(name: str, v: float) -> None:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {v}")
