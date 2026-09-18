import numpy as np
import pandas as pd
import pytest

from betting_model.backtest import walk_forward


def _synthetic_league(n_rounds: int = 60) -> pd.DataFrame:
    """Four teams, round-robin fixtures every week, with plausible-looking
    (but not realistic) closing odds attached to every match so the
    market-side columns in walk_forward's output get exercised too."""
    rng = np.random.default_rng(1)
    strength = {"Strong": 2.2, "Mid": 1.5, "Weak": 0.9, "Other": 1.2}
    fixtures = [("Strong", "Weak"), ("Mid", "Other"), ("Weak", "Mid"), ("Other", "Strong")]
    start = pd.Timestamp("2020-01-01")

    rows = []
    for round_i in range(n_rounds):
        date = start + pd.Timedelta(days=round_i * 7)
        for home, away in fixtures:
            lam = max(strength[home] / strength[away] * 1.1, 0.1)
            mu = max(strength[away] / strength[home], 0.1)
            hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            rows.append(dict(
                date=date, league="Test League", home_team=home, away_team=away,
                fthg=hg, ftag=ag, ftr=ftr,
                b365h=2.0, b365d=3.4, b365a=3.6,
                b365_over25=1.9, b365_under25=1.95,
                ah_line=-0.5, b365_ah_home=1.95, b365_ah_away=1.9,
            ))
    return pd.DataFrame(rows)


def _midpoint_start(matches: pd.DataFrame) -> pd.Timestamp:
    return matches["date"].iloc[len(matches) // 2]


def test_walk_forward_predicts_only_on_or_after_start_date():
    matches = _synthetic_league()
    start_date = _midpoint_start(matches)
    result = walk_forward(matches, league="Test League", start_date=start_date, lookback_days=3650)

    assert len(result) > 0
    assert (result["date"] >= start_date).all()


def test_walk_forward_model_probabilities_are_valid_distributions():
    matches = _synthetic_league()
    start_date = _midpoint_start(matches)
    result = walk_forward(matches, league="Test League", start_date=start_date, lookback_days=3650)

    totals = result["model_home"] + result["model_draw"] + result["model_away"]
    assert totals.round(6).eq(1.0).all()
    assert (result["model_home"] >= 0).all() and (result["model_away"] >= 0).all()


def test_walk_forward_includes_devigged_market_probabilities():
    matches = _synthetic_league()
    start_date = _midpoint_start(matches)
    result = walk_forward(matches, league="Test League", start_date=start_date, lookback_days=3650)

    assert "market_home" in result.columns
    totals = result["market_home"] + result["market_draw"] + result["market_away"]
    assert totals.round(6).eq(1.0).all()

    ou_totals = result["market_over25"] + result["market_under25"]
    assert ou_totals.round(6).eq(1.0).all()


def test_walk_forward_includes_asian_handicap_columns():
    matches = _synthetic_league()
    start_date = _midpoint_start(matches)
    result = walk_forward(matches, league="Test League", start_date=start_date, lookback_days=3650)

    assert {"model_ah_win", "model_ah_push", "model_ah_loss", "market_ah_win", "market_ah_loss"} <= set(result.columns)
    ah_totals = result["model_ah_win"] + result["model_ah_push"] + result["model_ah_loss"]
    assert ah_totals.round(6).eq(1.0).all()


def test_walk_forward_raises_when_nothing_to_predict():
    matches = _synthetic_league()
    with pytest.raises(ValueError):
        walk_forward(matches, league="Test League", start_date=matches["date"].max() + pd.Timedelta(days=365))


def test_walk_forward_skips_team_with_no_history_in_window():
    matches = _synthetic_league()
    start_date = _midpoint_start(matches)

    # a brand-new team appears only on the very last predicted date, with
    # no prior matches at all -- fit_dixon_coles can't know its strength.
    newcomer_date = matches["date"].max() + pd.Timedelta(days=7)
    newcomer_row = pd.DataFrame([dict(
        date=newcomer_date, league="Test League", home_team="Newcomer FC", away_team="Strong",
        fthg=0, ftag=2, ftr="A",
        b365h=5.0, b365d=4.0, b365a=1.5,
        b365_over25=1.9, b365_under25=1.95,
        ah_line=1.0, b365_ah_home=1.9, b365_ah_away=1.95,
    )])
    matches_with_newcomer = pd.concat([matches, newcomer_row], ignore_index=True)

    result = walk_forward(matches_with_newcomer, league="Test League", start_date=start_date, lookback_days=3650)
    assert "Newcomer FC" not in set(result["home_team"]) | set(result["away_team"])