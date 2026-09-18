"""Calibration metrics for probabilistic predictions.

Judge the model on calibration, not accuracy -- a model that says "60%"
should be right about 60% of the time, not necessarily right *most* of the
time. Brier score and log-loss are the standard scalar summaries;
`reliability_table` is the check for *where* a model is over/under-
confident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(predicted_prob: np.ndarray, actual_outcome: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1
    outcome. Lower is better: 0 is perfect, 0.25 is what "always guess 50%"
    scores against a fair coin."""
    p = np.asarray(predicted_prob, dtype=float)
    y = np.asarray(actual_outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(predicted_prob: np.ndarray, actual_outcome: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the actual outcome under the
    predicted probability. Lower is better; penalizes confident wrong
    predictions far more harshly than Brier score does."""
    p = np.clip(np.asarray(predicted_prob, dtype=float), eps, 1 - eps)
    y = np.asarray(actual_outcome, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(predicted_prob: np.ndarray, actual_outcome: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into `n_bins` equal-width probability bins and
    compare each bin's mean predicted probability to its actual outcome
    rate. A well-calibrated model has the two columns close together in
    every bin that has enough matches to be meaningful."""
    df = pd.DataFrame({"predicted": predicted_prob, "actual": actual_outcome})
    df["bin"] = pd.cut(df["predicted"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = df.groupby("bin", observed=True).agg(
        n=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        actual_rate=("actual", "mean"),
    )
    return table.reset_index()