import pandas as pd

from betting_model.ingest import _season_label, normalize

SAMPLE_ROWS = [
    # a match we care about (E0 = Premier League)
    dict(
        Division="E0", MatchDate="2023-08-15", HomeTeam="Arsenal", AwayTeam="Wolves",
        FTHome=2, FTAway=1, FTResult="H", HTHome=1, HTAway=0, HTResult="H",
        OddHome=1.3, OddDraw=5.5, OddAway=9.0, MaxHome=1.35, MaxDraw=5.6, MaxAway=9.2,
        Over25=1.6, Under25=2.35, MaxOver25=1.65, MaxUnder25=2.45,
        HandiSize=-1.5, HandiHome=1.9, HandiAway=1.95,
        HomeElo=1900.0, AwayElo=1600.0, Form3Home=9, Form5Home=15, Form3Away=3, Form5Away=6,
        HomeShots=15, AwayShots=8, HomeTarget=7, AwayTarget=3,
        HomeCorners=6, AwayCorners=3, HomeYellow=1, AwayYellow=2, HomeRed=0, AwayRed=0,
    ),
    # a league we don't track (should be filtered out)
    dict(
        Division="SC0", MatchDate="2023-08-15", HomeTeam="Celtic", AwayTeam="Rangers",
        FTHome=1, FTAway=1, FTResult="D", HTHome=0, HTAway=0, HTResult="D",
        OddHome=2.0, OddDraw=3.4, OddAway=3.8, MaxHome=2.05, MaxDraw=3.45, MaxAway=3.85,
        Over25=1.9, Under25=1.9, MaxOver25=1.95, MaxUnder25=1.95,
        HandiSize=-0.25, HandiHome=1.95, HandiAway=1.9,
        HomeElo=1700.0, AwayElo=1650.0, Form3Home=6, Form5Home=9, Form3Away=6, Form5Away=9,
        HomeShots=10, AwayShots=10, HomeTarget=4, AwayTarget=4,
        HomeCorners=5, AwayCorners=5, HomeYellow=3, AwayYellow=3, HomeRed=0, AwayRed=0,
    ),
    # missing home team -- should be dropped, not crash
    dict(
        Division="E0", MatchDate="2023-08-16", HomeTeam=None, AwayTeam="Fulham",
        FTHome=1, FTAway=0, FTResult="H", HTHome=0, HTAway=0, HTResult="D",
        OddHome=2.0, OddDraw=3.4, OddAway=3.8, MaxHome=2.05, MaxDraw=3.45, MaxAway=3.85,
        Over25=1.9, Under25=1.9, MaxOver25=1.95, MaxUnder25=1.95,
        HandiSize=-0.25, HandiHome=1.95, HandiAway=1.9,
        HomeElo=1700.0, AwayElo=1650.0, Form3Home=6, Form5Home=9, Form3Away=6, Form5Away=9,
        HomeShots=10, AwayShots=10, HomeTarget=4, AwayTarget=4,
        HomeCorners=5, AwayCorners=5, HomeYellow=3, AwayYellow=3, HomeRed=0, AwayRed=0,
    ),
]


def test_season_label():
    assert _season_label(pd.Timestamp("2024-01-15")) == "2023-24"
    assert _season_label(pd.Timestamp("2024-08-01")) == "2024-25"


def test_normalize_filters_and_maps_columns():
    raw = pd.DataFrame(SAMPLE_ROWS)
    df = normalize(raw)

    # SC0 (untracked league) and the null-HomeTeam row are both dropped
    assert len(df) == 1
    row = df.iloc[0]
    assert row["league"] == "Premier League"
    assert row["season"] == "2023-24"
    assert row["home_team"] == "Arsenal"
    assert row["fthg"] == 2
    assert row["ftag"] == 1
    assert row["ftr"] == "H"
    assert row["b365h"] == 1.3
    assert row["maxh"] == 1.35
    assert row["ah_line"] == -1.5
    assert row["home_elo"] == 1900.0
    assert pd.Timestamp(row["date"]) == pd.Timestamp("2023-08-15")
