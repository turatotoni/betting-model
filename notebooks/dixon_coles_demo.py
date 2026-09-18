# %% [markdown]
# # Dixon-Coles model: does it produce sane, competitive predictions?
#
# This is NOT the walk-forward backtest (that's a later phase, and needs
# proper staking/CLV tracking across thousands of matches to mean
# anything). This is a much smaller sanity check: take a handful of the
# most recent real matches in the dataset, fit the model using *only* data
# strictly before each match (so this is genuinely out-of-sample, no
# leakage), and see whether its probabilities are in the same neighbourhood
# as the closing market odds, and whether the "favorite" it picks is
# reasonable.

# %%
import numpy as np
import pandas as pd

from betting_model.devig import devig
from betting_model.dixon_coles import fit_dixon_coles
from betting_model.markets import asian_handicap, match_odds, over_under

pd.set_option("display.width", 120)

df = pd.read_parquet("../data/processed/matches.parquet")


# %% [markdown]
# ## Pick the most recent finished match in each league as a test case
#
# Each one gets its own model, fit on that league's history up to (but not
# including) that match's date.

# %%
recent_matches = df.sort_values("date").groupby("league").tail(1)

rows = []
for _, match in recent_matches.iterrows():
    league_data = df[df["league"] == match["league"]]
    model = fit_dixon_coles(league_data, as_of=match["date"])

    matrix = model.score_matrix(match["home_team"], match["away_team"])
    model_1x2 = match_odds(matrix)
    model_ou = over_under(matrix, line=2.5)

    market_h, market_d, market_a = devig(match["b365h"], match["b365d"], match["b365a"])

    predicted = max(model_1x2, key=model_1x2.get)
    actual = {"H": "home", "D": "draw", "A": "away"}[match["ftr"]]

    rows.append({
        "league": match["league"],
        "date": match["date"].date(),
        "match": f"{match['home_team']} vs {match['away_team']}",
        "score": f"{int(match['fthg'])}-{int(match['ftag'])}",
        "model_H/D/A": f"{model_1x2['home']:.2f}/{model_1x2['draw']:.2f}/{model_1x2['away']:.2f}",
        "market_H/D/A": f"{market_h:.2f}/{market_d:.2f}/{market_a:.2f}",
        "model_picked": predicted,
        "actual": actual,
        "correct": predicted == actual,
        "model_over2.5": round(model_ou["over"], 2),
        "market_over2.5_implied": round(1 / match["b365_over25"], 2) if pd.notna(match["b365_over25"]) else None,
    })

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))

# %% [markdown]
# ## Reading this table
#
# - `model_H/D/A` vs `market_H/D/A`: if these are wildly different, either
#   the model or the de-vig math is broken -- they should be in the same
#   ballpark even though they won't match exactly (that gap, when it's the
#   model looking *better* calibrated, is where a real edge would come
#   from -- but you can't tell that from 6 matches, only from a proper
#   backtest).
# - `correct`: whether the model's most-likely outcome actually happened.
#   Do not over-read this on 6 matches -- even a well-calibrated model that
#   says "55% home win" is *supposed* to be wrong 45% of the time. This is
#   a smoke test, not validation.

# %% [markdown]
# ## One fixture in full detail, including Asian Handicap
#
# Asian Handicap is our primary target market, so it's worth checking the
# quarter-line blending logic against a real match with a real line.

# %%
example = recent_matches.iloc[0]
league_data = df[df["league"] == example["league"]]
model = fit_dixon_coles(league_data, as_of=example["date"])
matrix = model.score_matrix(example["home_team"], example["away_team"])

print(f"{example['home_team']} (home) vs {example['away_team']} (away), {example['date'].date()}")
print(f"Actual score: {int(example['fthg'])}-{int(example['ftag'])}")
print(f"Model expected goals: {model.expected_goals(example['home_team'], example['away_team'])}")
print(f"Bookmaker AH line: {example['ah_line']}, Bet365 odds home={example['b365_ah_home']}, away={example['b365_ah_away']}")

if pd.notna(example["ah_line"]):
    ah = asian_handicap(matrix, line=float(example["ah_line"]))
    print(f"Model AH probabilities at that line: {ah}")
    if pd.notna(example["b365_ah_home"]) and pd.notna(example["b365_ah_away"]):
        market_win, market_loss = devig(example["b365_ah_home"], example["b365_ah_away"])
        print(f"Market-implied (de-vigged, ignoring push): win={market_win:.2f}, loss={market_loss:.2f}")