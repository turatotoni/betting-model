# %% [markdown]
# # Sanity checks on the unified match dataset
#
# Run this before building any model. It answers two questions:
# 1. Does the data look right (goal distributions, home advantage)?
# 2. Where are the softest bookmaker margins -- i.e. where should we expect
#    edge-hunting to be more promising? This is the first real evidence for
#    or against the "Asian Handicap / Over-Under in non-PL leagues" edge
#    hypothesis behind this project.
#
# This file uses `# %%` cell markers -- PyCharm's Scientific Mode and VS
# Code both run these as notebook-style cells.

# %%
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 120)

df = pd.read_parquet("../data/processed/matches.parquet")
print(f"{len(df):,} matches, {df['league'].nunique()} leagues, "
      f"{df['season'].min()}..{df['season'].max()}")
df.groupby("league")["season"].agg(["min", "max", "count"])

# %% [markdown]
# ## 1. Goal distribution sanity check
#
# Football goals are famously close to Poisson-distributed (this is *why*
# Poisson-based models work reasonably well for this sport). If the shape
# looks wildly off, something is wrong with the parsing, not the sport.

# %%
total_goals = df["fthg"] + df["ftag"]
total_goals.value_counts().sort_index().plot(kind="bar", title="Total goals per match")
plt.xlabel("Total goals")
plt.ylabel("Matches")
plt.show()

print(f"Mean total goals/match: {total_goals.mean():.2f}")
print(f"Variance: {total_goals.var():.2f}  (Poisson would have variance == mean)")

# %% [markdown]
# ## 2. Home advantage check
#
# Historically home teams in top European leagues win somewhere around
# 44-46% of matches, draws ~25-27%, away wins ~28-30%. If your numbers are
# way outside that, double check the FTR / team-column mapping.

# %%
df["ftr"].value_counts(normalize=True).rename({"H": "Home win", "D": "Draw", "A": "Away win"})

# %%
df.groupby("league")["ftr"].value_counts(normalize=True).unstack().round(3)

# %% [markdown]
# ## 3. Bookmaker margin (overround) by market and league
#
# `implied_prob = 1 / odds`. Summed across all outcomes of a market, this
# is always > 1 -- the excess over 1.0 is the bookmaker's margin. Lower
# margin markets are, all else equal, easier to find value in, since
# there's less of a hole to climb out of before you're even at breakeven.
#
# We compare 1X2 vs Over/Under 2.5 vs Asian Handicap margins, both
# overall and broken out by league -- this is the first real evidence for
# where this project should focus (see the plan's edge-strategy rationale:
# AH/O-U, non-PL leagues).

# %%
def margin(odds_cols: list[str]) -> pd.Series:
    implied = sum(1 / df[c] for c in odds_cols)
    return implied - 1

df["margin_1x2"] = margin(["b365h", "b365d", "b365a"])
df["margin_ou25"] = margin(["b365_over25", "b365_under25"])
df["margin_ah"] = margin(["b365_ah_home", "b365_ah_away"])

df[["margin_1x2", "margin_ou25", "margin_ah"]].mean().rename("mean bookmaker margin")

# %%
df.groupby("league")[["margin_1x2", "margin_ou25", "margin_ah"]].mean().round(4)

# %% [markdown]
# ## 4. De-vigged probabilities sum to 1.0 (correctness check)
#
# This is a pure sanity check on the margin-removal math itself, not a
# statement about the data: dividing each implied probability by the total
# implied probability must produce a distribution that sums to exactly 1.

# %%
# decimal odds are always > 1; a couple of rows in the source data have a
# bogus 0.0 (e.g. Blackpool vs Derby, 2013-04-27) -- drop those too, not
# just the ones missing odds entirely.
valid_odds = (df["b365h"] > 1) & (df["b365d"] > 1) & (df["b365a"] > 1)
implied_1x2 = pd.DataFrame({
    "H": 1 / df.loc[valid_odds, "b365h"],
    "D": 1 / df.loc[valid_odds, "b365d"],
    "A": 1 / df.loc[valid_odds, "b365a"],
})
devigged = implied_1x2.div(implied_1x2.sum(axis=1), axis=0)
assert devigged.sum(axis=1).round(6).eq(1.0).all(), "de-vig math is broken"
print(f"De-vigged 1X2 probabilities sum to 1.0 across {len(devigged):,} matches with odds -- OK")
