"""Calibration analysis — does the model's N% actually happen N% of the time?

A well-calibrated 70% prediction should win ~70% of the time across many samples.
We bin predictions by probability and compare average predicted probability in
each bin to the observed hit rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_prediction: float  # mean probability predicted in this bin
    hit_rate: float  # fraction of outcomes that were 1


@dataclass
class CalibrationReport:
    bins: list[CalibrationBin]
    ece: float  # expected calibration error — weighted |predicted - actual|

    def readable(self) -> str:
        lines = [f"Calibration (ECE = {self.ece:.3f})"]
        lines.append(f"{'bin':<14} {'n':>5} {'pred':>7} {'actual':>7} {'diff':>7}")
        for b in self.bins:
            diff = b.avg_prediction - b.hit_rate
            lines.append(
                f"[{b.lower:.2f}, {b.upper:.2f}) "
                f"{b.count:>5} {b.avg_prediction:>7.3f} {b.hit_rate:>7.3f} {diff:>+7.3f}"
            )
        return "\n".join(lines)


def calibration_report(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> CalibrationReport:
    if len(probabilities) != len(outcomes):
        raise ValueError("length mismatch")
    if not probabilities:
        raise ValueError("cannot compute calibration on empty inputs")

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[CalibrationBin] = []
    total = len(probabilities)
    ece_weighted = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        # Last bin is inclusive on upper
        if i == n_bins - 1:
            in_bin = [(p, o) for p, o in zip(probabilities, outcomes) if lo <= p <= hi]
        else:
            in_bin = [(p, o) for p, o in zip(probabilities, outcomes) if lo <= p < hi]
        n = len(in_bin)
        if n == 0:
            bins.append(CalibrationBin(lower=lo, upper=hi, count=0, avg_prediction=0, hit_rate=0))
            continue
        avg_p = sum(p for p, _ in in_bin) / n
        hit = sum(o for _, o in in_bin) / n
        ece_weighted += (n / total) * abs(avg_p - hit)
        bins.append(
            CalibrationBin(lower=lo, upper=hi, count=n, avg_prediction=avg_p, hit_rate=hit)
        )

    return CalibrationReport(bins=bins, ece=ece_weighted)
