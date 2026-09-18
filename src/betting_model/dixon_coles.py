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

# Picked by a walk-forward grid search over xi in {0.0005, 0.0018, 0.004}
# and lookback_days in {730, 1460, 2920} across all 6 leagues, 2005-2026,
# scored on Over/Under 2.5 and Asian Handicap Brier score (see
# backtest.py / calibration.py). The grid's spread was small (Brier_ou
# 0.2473-0.2510, Brier_ah 0.2497-0.2518) -- these values win or nearly
# win on all three markets (O/U, AH, 1X2) but do NOT fix the model's
# deeper overconfidence at the probability extremes on O/U and AH (see
# reliability_table output); that looks structural -- point-estimate MLE
# doesn't propagate team-strength uncertainty, and it compounds for
# sum-of-two-teams markets (goals, handicap margin) in a way it doesn't
# for the win/draw/away split. A post-hoc calibration layer is the next
# thing to try for that, not further (xi, lookback_days) tuning.
DEFAULT_XI = 0.0018  # time-decay rate per day (~385-day half-life)
DEFAULT_LOOKBACK_DAYS = 8 * 365


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

    low_score = (home_goals <= 1) & (away_goals <= 1)
    ls_hg, ls_ag = home_goals[low_score], away_goals[low_score]
    is00, is01 = (ls_hg == 0) & (ls_ag == 0), (ls_hg == 0) & (ls_ag == 1)
    is10, is11 = (ls_hg == 1) & (ls_ag == 0), (ls_hg == 1) & (ls_ag == 1)

    def neg_log_likelihood_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        attack, defense, home_adv, rho = unpack(theta)
        z_lam = home_adv + attack[home_idx] - defense[away_idx]
        z_mu = attack[away_idx] - defense[home_idx]
        lam, mu = np.exp(z_lam), np.exp(z_mu)

        ll = poisson.logpmf(home_goals, lam) + poisson.logpmf(away_goals, mu)

        # d(log Poisson(k; lam))/d(z_lam) = k - lam (since lam = exp(z_lam));
        # same shape for mu/z_mu. rho has no gradient contribution outside
        # the four low-score cells below.
        d_zlam = home_goals - lam
        d_zmu = away_goals - mu
        d_rho = np.zeros_like(lam)

        if low_score.any():
            l, m = lam[low_score], mu[low_score]
            tau_vals = np.ones_like(l)
            tau_vals[is00] = 1 - l[is00] * m[is00] * rho
            tau_vals[is01] = 1 + l[is01] * rho
            tau_vals[is10] = 1 + m[is10] * rho
            tau_vals[is11] = 1 - rho
            tau_vals = np.clip(tau_vals, 1e-10, None)
            ll[low_score] += np.log(tau_vals)

            # d(tau)/d(lam), d(tau)/d(mu), d(tau)/d(rho) per case, chained
            # through d(lam)/d(z_lam) = lam (and same for mu) to get
            # d(log tau)/d(z_lam) = [d(tau)/d(lam) * lam] / tau, etc.
            dtau_dlam, dtau_dmu, dtau_drho = np.zeros_like(l), np.zeros_like(l), np.zeros_like(l)
            dtau_dlam[is00], dtau_dmu[is00] = -m[is00] * rho, -l[is00] * rho
            dtau_drho[is00] = -l[is00] * m[is00]
            dtau_dlam[is01], dtau_drho[is01] = rho, l[is01]
            dtau_dmu[is10], dtau_drho[is10] = rho, m[is10]
            dtau_drho[is11] = -1.0

            d_zlam[low_score] += dtau_dlam * l / tau_vals
            d_zmu[low_score] += dtau_dmu * m / tau_vals
            d_rho[low_score] += dtau_drho / tau_vals

        nll = -np.sum(weights * ll)

        w_zlam, w_zmu, w_rho = weights * d_zlam, weights * d_zmu, weights * d_rho
        grad_attack, grad_defense = np.zeros(n), np.zeros(n)
        np.add.at(grad_attack, home_idx, -w_zlam)
        np.add.at(grad_attack, away_idx, -w_zmu)
        np.add.at(grad_defense, away_idx, w_zlam)
        np.add.at(grad_defense, home_idx, w_zmu)
        grad_home_adv, grad_rho = -np.sum(w_zlam), -np.sum(w_rho)

        grad = np.concatenate([grad_attack[1:], grad_defense, [grad_home_adv], [grad_rho]])
        return nll, grad

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.2], [0.0]])
    bounds = [(-3.0, 3.0)] * (n - 1) + [(-3.0, 3.0)] * n + [(-2.0, 2.0), (-1.0, 1.0)]

    result = minimize(neg_log_likelihood_and_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds)
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