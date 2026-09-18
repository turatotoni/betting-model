"""Strip bookmaker margin (overround) out of decimal odds.

Decimal odds imply a probability of `1 / odds` per outcome; summed across
all outcomes of a market this is always > 1.0 -- the excess is the
bookmaker's margin. This module removes it via the basic proportional
method (scale each implied probability down by the same factor so they
sum to 1.0). Works for any number of outcomes: 2-way (Over/Under, Asian
Handicap) or 3-way (1X2).
"""

from __future__ import annotations


def devig(*odds: float) -> tuple[float, ...]:
    """De-vigged probabilities for a set of mutually exclusive decimal odds,
    in the same order as given. Every value must be > 1.0 (a valid decimal
    price)."""
    if any(o <= 1.0 for o in odds):
        raise ValueError(f"decimal odds must be > 1.0, got {odds}")
    implied = [1 / o for o in odds]
    total = sum(implied)
    return tuple(p / total for p in implied)