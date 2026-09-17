import numpy as np
import pandas as pd
import pytest

from betting_model.dixon_coles import _tau, fit_dixon_coles


def test_tau_values():
    assert _tau(0, 0, 1.0, 1.0, 0.1) == pytest.approx(1 - 1.0 * 1.0 * 0.1)
    assert _tau(0, 1, 1.0, 1.0, 0.1) == pytest.approx(1 + 1.0 * 0.1)
    assert _tau(1, 0, 1.0, 1.0, 0.1) == pytest.approx(1 + 1.0 * 0.1)
    assert _tau(1, 1, 1.0, 1.0, 0.1) == pytest.approx(1 - 0.1)
    assert _tau(2, 2, 1.0, 1.0, 0.1) == 1.0


def _synthetic_league(n_rounds: int = 20) -> pd.DataFrame:
    """Strong always beats Weak at home and away; Mid is in between. Any
    reasonable fit should recover Strong > Mid > Weak in attack strength
    and the reverse ordering in defense (lower defense = leakier)."""
    rng = np.random.default_rng(0)
    teams = ["Strong", "Mid", "Weak"]
    rows = []
    start = pd.Timestamp("2024-01-01")
    strength = {"Strong": 2.2, "Mid": 1.3, "Weak": 0.6}
    for round_i in range(n_rounds):
        date = start + pd.Timedelta(days=round_i * 7)
        for home, away in [("Strong", "Weak"), ("Mid", "Strong"), ("Weak", "Mid")]:
            lam = strength[home] * (1 / strength[away]) * 1.1
            mu = strength[away] * (1 / strength[home])
            rows.append(dict(
                date=date, league="Test League", home_team=home, away_team=away,
                fthg=rng.poisson(max(lam, 0.1)), ftag=rng.poisson(max(mu, 0.1)),
            ))
    return pd.DataFrame(rows)


def test_fit_recovers_relative_team_strength():
    matches = _synthetic_league()
    as_of = matches["date"].max() + pd.Timedelta(days=7)
    model = fit_dixon_coles(matches, as_of=as_of, lookback_days=3650, xi=0.0)

    assert model.n_matches == len(matches)
    assert set(model.teams) == {"Strong", "Mid", "Weak"}
    assert model.attack["Strong"] > model.attack["Mid"] > model.attack["Weak"]
    assert model.defense["Strong"] > model.defense["Weak"]  # Strong concedes less -> higher defense param


def test_score_matrix_is_a_valid_distribution():
    matches = _synthetic_league()
    as_of = matches["date"].max() + pd.Timedelta(days=7)
    model = fit_dixon_coles(matches, as_of=as_of, lookback_days=3650, xi=0.0)

    matrix = model.score_matrix("Strong", "Weak", max_goals=10)
    assert matrix.shape == (11, 11)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_expected_goals_raises_for_unknown_team():
    matches = _synthetic_league()
    as_of = matches["date"].max() + pd.Timedelta(days=7)
    model = fit_dixon_coles(matches, as_of=as_of, lookback_days=3650, xi=0.0)

    with pytest.raises(KeyError):
        model.expected_goals("Strong", "Nonexistent FC")


def test_fit_raises_when_window_is_empty():
    matches = _synthetic_league()
    with pytest.raises(ValueError):
        fit_dixon_coles(matches, as_of=pd.Timestamp("2020-01-01"), lookback_days=30)