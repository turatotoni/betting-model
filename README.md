# BettingModel

A European football (soccer) betting model, built from scratch toward
actual real-money betting. Full roadmap and reasoning:
`~/.claude/plans/i-want-to-try-encapsulated-canyon.md`.

**Strategy:** rather than chase 1X2 in efficient top-league markets, build
one Dixon-Coles-style Poisson goal model and hunt for edge primarily in
**Asian Handicap and Over/Under**, across the top 5-6 European leagues plus
softer lower-tier leagues. Early data confirms the premise: Bet365's
average margin is ~4.0-4.4% on Asian Handicap vs ~5.6-7.1% on 1X2 across
these leagues (see `notebooks/sanity_checks.py`) -- less margin to
overcome before a bet is even +EV.

## Setup

```bash
python -m venv .venv       # already done if you're reading this in PyCharm
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .           # makes `betting_model` importable as a package
```

## Data

```bash
python -m betting_model.ingest
```

Downloads and unifies match results + closing odds (Bet365 + best-of-market)
for Premier League, Championship, La Liga, Serie A, Bundesliga, and Ligue 1,
2000/01 through the current season, into `data/processed/matches.parquet`.

Source: [xgabora/Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data)
(MIT licensed), which aggregates [football-data.co.uk](https://www.football-data.co.uk/)
(results + odds) and [ClubElo](https://www.clubelo.com/) (Elo ratings).

We'd originally planned to pull season-by-season CSVs directly from
football-data.co.uk, but that site blocks all non-browser HTTP clients
(confirmed with curl and with two independent fetch tools -- every
automated request gets redirected to a dead `127.0.0.1` URL, regardless of
User-Agent or client). The GitHub mirror above is sourced from the same
underlying data and isn't blocked.

Raw download is cached at `data/raw/matches_source.csv`; re-run with
`--refresh` to force a re-download once new seasons/matches are appended
upstream, or `--no-download` to rebuild `matches.parquet` from the cached
raw file only.

## Sanity checks

`notebooks/sanity_checks.py` (run as a PyCharm Scientific-Mode / VS Code
`# %%` cell script, or top-to-bottom as a plain script) validates the data
-- goal distributions, home-advantage rates, bookmaker margins by market
and league -- before any modeling work builds on top of it.

## Tests

```bash
pytest
```
