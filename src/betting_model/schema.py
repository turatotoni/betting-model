"""Unified schema for match data.

Source: xgabora/Club-Football-Match-Data on GitHub (MIT licensed), which
itself aggregates football-data.co.uk (results + odds) and ClubElo (Elo
ratings) into one CSV, updated periodically.
https://github.com/xgabora/Club-Football-Match-Data

We originally planned to pull per-season CSVs directly from
football-data.co.uk, but that site blocks non-browser HTTP clients
entirely (curl, requests, and Anthropic's own fetch infra all get
redirected to a dead localhost URL -- confirmed during development, not
specific to any one machine). This GitHub-hosted mirror is sourced from the
same underlying data, isn't blocked, and is a single file instead of one
CSV per league/season -- simpler to work with.

Division codes match football-data.co.uk's own convention (E0 = Premier
League, SP1 = La Liga, etc).
"""

from __future__ import annotations

# Leagues covered in v1, keyed by division code (matches football-data.co.uk).
LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# unified_field -> source column name in Matches.csv
COLUMN_MAP: dict[str, str] = {
    # core match info
    "date": "MatchDate",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "fthg": "FTHome",  # full-time home goals
    "ftag": "FTAway",  # full-time away goals
    "ftr": "FTResult",  # H/D/A
    "hthg": "HTHome",
    "htag": "HTAway",
    "htr": "HTResult",

    # 1X2 odds -- Bet365 (closing), and max across ~17 tracked bookmakers.
    # No Pinnacle-specific column here; Max* is the closest proxy for "best
    # available price" for line-shopping/CLV purposes later.
    "b365h": "OddHome",
    "b365d": "OddDraw",
    "b365a": "OddAway",
    "maxh": "MaxHome",
    "maxd": "MaxDraw",
    "maxa": "MaxAway",

    # Over/Under 2.5 goals
    "b365_over25": "Over25",
    "b365_under25": "Under25",
    "max_over25": "MaxOver25",
    "max_under25": "MaxUnder25",

    # Asian handicap: line (negative = stronger home team) + Bet365 odds
    "ah_line": "HandiSize",
    "b365_ah_home": "HandiHome",
    "b365_ah_away": "HandiAway",

    # Elo + form -- not strictly needed for the Dixon-Coles model, but
    # useful sanity-check / comparison baselines later.
    "home_elo": "HomeElo",
    "away_elo": "AwayElo",
    "form3_home": "Form3Home",
    "form5_home": "Form5Home",
    "form3_away": "Form3Away",
    "form5_away": "Form5Away",

    # match stats, useful for later feature engineering
    "home_shots": "HomeShots",
    "away_shots": "AwayShots",
    "home_target": "HomeTarget",
    "away_target": "AwayTarget",
    "home_corners": "HomeCorners",
    "away_corners": "AwayCorners",
    "home_yellow": "HomeYellow",
    "away_yellow": "AwayYellow",
    "home_red": "HomeRed",
    "away_red": "AwayRed",
}

UNIFIED_COLUMNS: list[str] = ["league", "season", *COLUMN_MAP.keys()]
