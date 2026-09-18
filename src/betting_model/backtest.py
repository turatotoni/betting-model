"""Walk-forward backtest: fit Dixon-Coles strictly on past data, predict
forward matches, and attach de-vigged closing-odds probabilities plus
actual results for later scoring (calibration.py) and edge detection.

Refitting is periodic (`refit_every_days`), not per match -- team
strengths move slowly match to match, and refitting the optimizer
thousands of times per league would be needlessly slow. This stays
leakage-free as long as each fit's `as_of` is <= every match it predicts,
which a period-start refit still guarantees (the model is fit once at the
start of a period, strictly before its own refit date, and is then only
used to predict matches on or after that date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from betting_model.devig import devig
from betting_model.dixon_coles import DEFAULT_LOOKBACK_DAYS, DEFAULT_XI, DixonColesModel, fit_dixon_coles
from betting_model.markets import asian_handicap, btts, match_odds, over_under


def walk_forward(
    matches: pd.DataFrame,
    league: str,
    start_date: pd.Timestamp,
    refit_every_days: int = 14,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    xi: float = DEFAULT_XI,
) -> pd.DataFrame:
    """Predict every `league` match on/after `start_date`, refitting every
    `refit_every_days`. Returns one row per predicted match with model
    probabilities, de-vigged market probabilities (where odds are valid),
    and actual outcomes. A match is skipped if either team has no history
    in the fit window (e.g. just promoted).
    """
    league_data = matches[matches["league"] == league].sort_values("date")
    to_predict = league_data[league_data["date"] >= start_date]
    if to_predict.empty:
        raise ValueError(f"no {league} matches on/after {start_date.date()}")

    model: DixonColesModel | None = None
    next_refit = pd.Timestamp.min
    rows = []

    for _, match in to_predict.iterrows():
        if match["date"] >= next_refit:
            model = fit_dixon_coles(league_data, as_of=match["date"], lookback_days=lookback_days, xi=xi)
            next_refit = match["date"] + pd.Timedelta(days=refit_every_days)

        try:
            matrix = model.score_matrix(match["home_team"], match["away_team"])
        except KeyError:
            continue

        rows.append(_prediction_row(match, matrix))

    return pd.DataFrame(rows)


def _prediction_row(match: pd.Series, matrix: np.ndarray) -> dict:
    model_1x2 = match_odds(matrix)
    model_ou = over_under(matrix, line=2.5)
    model_btts = btts(matrix)

    row = {
        "date": match["date"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "actual_ftr": match["ftr"],
        "actual_total_goals": match["fthg"] + match["ftag"],
        "model_home": model_1x2["home"],
        "model_draw": model_1x2["draw"],
        "model_away": model_1x2["away"],
        "model_over25": model_ou["over"],
        "model_under25": model_ou["under"],
        "model_btts_yes": model_btts["yes"],
    }

    if _valid_odds(match, "b365h", "b365d", "b365a"):
        market_h, market_d, market_a = devig(match["b365h"], match["b365d"], match["b365a"])
        row["market_home"], row["market_draw"], row["market_away"] = market_h, market_d, market_a

    if _valid_odds(match, "b365_over25", "b365_under25"):
        market_over, market_under = devig(match["b365_over25"], match["b365_under25"])
        row["market_over25"], row["market_under25"] = market_over, market_under

    if pd.notna(match["ah_line"]) and _valid_odds(match, "b365_ah_home", "b365_ah_away"):
        ah_line = float(match["ah_line"])
        model_ah = asian_handicap(matrix, line=ah_line)
        market_ah_win, market_ah_loss = devig(match["b365_ah_home"], match["b365_ah_away"])
        row["ah_line"] = ah_line
        row["model_ah_win"], row["model_ah_push"], row["model_ah_loss"] = (
            model_ah["win"], model_ah["push"], model_ah["loss"],
        )
        row["market_ah_win"], row["market_ah_loss"] = market_ah_win, market_ah_loss
        row["actual_ah_margin"] = match["fthg"] - match["ftag"] + ah_line

    return row


def _valid_odds(match: pd.Series, *cols: str) -> bool:
    return all(pd.notna(match[c]) and match[c] > 1.0 for c in cols)