"""Dixon-Coles Poisson model for football match goals.

Dixon, M.J. and Coles, S.G. (1997), "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market". Each team gets an
attack strength and a defense strength; expected goals for a match combine
them with a home-advantage term. Treating home/away goals as independent
Poisson variables underestimates low-scoring draws (0-0, 1-1) in practice,
so a correlation correction (rho) is fit alongside attack/defense to fix
just those four low-scoring cells.

This module only fits the goal model and produces a full scoreline
probability matrix -- turning that into specific market probabilities
(1X2, O/U, Asian Handicap, BTTS) lives in markets.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Both of these are untuned placeholders, not results of any validation --
# tuning them properly is a job for the walk-forward backtest (later phase),
# which can score different (xi, lookback_days) pairs against calibration/
# CLV rather than eyeballing it.
DEFAULT_XI = 0.0018  # time-decay rate per day (~385-day half-life)
DEFAULT_LOOKBACK_DAYS = 4 * 365


@dataclass
class DixonColesModel:
    league: str
    as_of: pd.Timestamp
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    n_matches: int

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        missing = [t for t in (home_team, away_team) if t not in self.attack]
        if missing:
            raise KeyError(f"team(s) not in fitted model (no recent matches in window): {missing}")
        lam = np.exp(self.home_adv + self.attack[home_team] - self.defense[away_team])
        mu = np.exp(self.attack[away_team] - self.defense[home_team])
        return float(lam), float(mu)

    def score_matrix(self, home_team: str, away_team: str, max_goals: int = 10) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..max_goals, rows=home goals."""
        lam, mu = self.expected_goals(home_team, away_team)
        home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
        away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
        matrix = np.outer(home_probs, away_probs)

        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            matrix[x, y] *= _tau(x, y, lam, mu, self.rho)

        return matrix / matrix.sum()


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction, applied only to the four cells
    where x<=1 and y<=1; every other scoreline gets a factor of 1."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    xi: float = DEFAULT_XI,
) -> DixonColesModel:
    """Fit team attack/defense strengths on matches strictly before `as_of`.

    `matches` must already be filtered to a single league -- attack/defense
    strengths from different leagues aren't on the same scale, since
    scoring environments differ (e.g. Bundesliga vs Serie A) -- and must
    have columns: date, league, home_team, away_team, fthg, ftag.

    Team strengths aren't identifiable without one constraint (shifting
    every attack AND every defense by the same constant leaves all
    predictions unchanged), so the first team alphabetically is fixed as
    the reference with attack = 0; every other team's attack is relative
    to it.
    """
    window_start = as_of - pd.Timedelta(days=lookback_days)
    fit_data = matches[(matches["date"] >= window_start) & (matches["date"] < as_of)]
    fit_data = fit_data.dropna(subset=["home_team", "away_team", "fthg", "ftag"])
    if fit_data.empty:
        raise ValueError(f"no matches in [{window_start.date()}, {as_of.date()}) to fit on")

    teams = sorted(set(fit_data["home_team"]) | set(fit_data["away_team"]))
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    days_ago = (as_of - fit_data["date"]).dt.days.to_numpy()
    weights = np.exp(-xi * days_ago)

    home_idx = fit_data["home_team"].map(team_idx).to_numpy()
    away_idx = fit_data["away_team"].map(team_idx).to_numpy()
    home_goals = fit_data["fthg"].to_numpy(dtype=float)
    away_goals = fit_data["ftag"].to_numpy(dtype=float)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        attack = np.concatenate(([0.0], theta[: n - 1]))
        defense = theta[n - 1: 2 * n - 1]
        home_adv, rho = theta[2 * n - 1], theta[2 * n]
        return attack, defense, home_adv, rho

    def neg_log_likelihood(theta: np.ndarray) -> float:
        attack, defense, home_adv, rho = unpack(theta)
        lam = np.exp(home_adv + attack[home_idx] - defense[away_idx])
        mu = np.exp(attack[away_idx] - defense[home_idx])

        ll = poisson.logpmf(home_goals, lam) + poisson.logpmf(away_goals, mu)

        low_score = (home_goals <= 1) & (away_goals <= 1)
        if low_score.any():
            tau_vals = np.array([
                _tau(int(hg), int(ag), l, m, rho)
                for hg, ag, l, m in zip(
                    home_goals[low_score], away_goals[low_score], lam[low_score], mu[low_score]
                )
            ])
            ll[low_score] += np.log(np.clip(tau_vals, 1e-10, None))

        return -np.sum(weights * ll)

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.2], [0.0]])
    bounds = [(-3.0, 3.0)] * (n - 1) + [(-3.0, 3.0)] * n + [(-2.0, 2.0), (-1.0, 1.0)]

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
    attack, defense, home_adv, rho = unpack(result.x)

    return DixonColesModel(
        league=str(fit_data["league"].iloc[0]),
        as_of=as_of,
        teams=teams,
        attack=dict(zip(teams, attack.tolist())),
        defense=dict(zip(teams, defense.tolist())),
        home_adv=float(home_adv),
        rho=float(rho),
        n_matches=len(fit_data),
    )