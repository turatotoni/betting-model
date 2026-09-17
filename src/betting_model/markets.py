"""Turn a Dixon-Coles scoreline probability matrix into market probabilities.

`matrix[i, j]` is always P(home scores i, away scores j).
"""

from __future__ import annotations

import numpy as np


def match_odds(matrix: np.ndarray) -> dict[str, float]:
    """1X2 probabilities."""
    return {
        "home": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away": float(np.triu(matrix, 1).sum()),
    }


def over_under(matrix: np.ndarray, line: float = 2.5) -> dict[str, float]:
    """Total-goals over/under probabilities. `line` is normally a .5 value
    (2.5, 3.5, ...) so there's no push, but any value works."""
    n = matrix.shape[0]
    total_goals = np.add.outer(np.arange(n), np.arange(n))
    return {
        "over": float(matrix[total_goals > line].sum()),
        "under": float(matrix[total_goals < line].sum()),
    }


def btts(matrix: np.ndarray) -> dict[str, float]:
    """Both teams to score."""
    yes = float(matrix[1:, 1:].sum())
    return {"yes": yes, "no": 1 - yes}


def asian_handicap(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Home-team-perspective Asian Handicap probabilities for `line`
    (negative = home team favored, e.g. -1.5 means home must win by 2+).

    Quarter lines (x.25 / x.75) split the stake across the two adjacent
    half/whole lines, exactly as real Asian Handicap markets do -- the
    returned probabilities are the blend of both.
    """
    doubled = line * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return _ah_single_line(matrix, line)

    lo = np.floor(doubled) / 2
    hi = lo + 0.5
    a = _ah_single_line(matrix, lo)
    b = _ah_single_line(matrix, hi)
    return {key: (a[key] + b[key]) / 2 for key in a}


def _ah_single_line(matrix: np.ndarray, line: float) -> dict[str, float]:
    n = matrix.shape[0]
    margin = np.subtract.outer(np.arange(n), np.arange(n)).astype(float)  # home goals - away goals
    adjusted = margin + line
    return {
        "win": float(matrix[adjusted > 0].sum()),
        "push": float(matrix[adjusted == 0].sum()),
        "loss": float(matrix[adjusted < 0].sum()),
    }