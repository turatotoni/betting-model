import numpy as np
import pytest

from betting_model.markets import asian_handicap, btts, match_odds, over_under


def _degenerate_matrix(home_goals: int, away_goals: int, size: int = 6) -> np.ndarray:
    """All probability mass on a single scoreline -- makes every derived
    market's expected value trivial to state by hand."""
    matrix = np.zeros((size, size))
    matrix[home_goals, away_goals] = 1.0
    return matrix


def test_match_odds_home_win():
    matrix = _degenerate_matrix(2, 1)
    result = match_odds(matrix)
    assert result == {"home": 1.0, "draw": 0.0, "away": 0.0}


def test_match_odds_draw():
    matrix = _degenerate_matrix(1, 1)
    result = match_odds(matrix)
    assert result == {"home": 0.0, "draw": 1.0, "away": 0.0}


def test_over_under_25():
    over = over_under(_degenerate_matrix(2, 1), line=2.5)  # 3 goals total
    assert over == {"over": 1.0, "under": 0.0}

    under = over_under(_degenerate_matrix(1, 0), line=2.5)  # 1 goal total
    assert under == {"over": 0.0, "under": 1.0}


def test_btts():
    assert btts(_degenerate_matrix(2, 1)) == {"yes": 1.0, "no": 0.0}
    assert btts(_degenerate_matrix(2, 0)) == {"yes": 0.0, "no": 1.0}


def test_asian_handicap_whole_line_push():
    # home wins by exactly 1, handicap -1 -> push
    result = asian_handicap(_degenerate_matrix(2, 1), line=-1.0)
    assert result == {"win": 0.0, "push": 1.0, "loss": 0.0}


def test_asian_handicap_half_line_no_push_possible():
    result = asian_handicap(_degenerate_matrix(2, 1), line=-0.5)  # margin 1 - 0.5 = 0.5 > 0
    assert result == {"win": 1.0, "push": 0.0, "loss": 0.0}

    result = asian_handicap(_degenerate_matrix(1, 1), line=-0.5)  # margin 0 - 0.5 = -0.5 < 0
    assert result == {"win": 0.0, "push": 0.0, "loss": 1.0}


def test_asian_handicap_quarter_line_blends_two_neighbours():
    # margin = 1 (home wins by 1). Quarter line -0.75 blends -0.5 (win) and -1.0 (push).
    result = asian_handicap(_degenerate_matrix(2, 1), line=-0.75)
    assert result["win"] == pytest.approx(0.5)
    assert result["push"] == pytest.approx(0.5)
    assert result["loss"] == pytest.approx(0.0)


def test_asian_handicap_probabilities_sum_to_one():
    matrix = _degenerate_matrix(2, 1) * 0.6 + _degenerate_matrix(1, 1) * 0.4
    for line in (-1.5, -1.0, -0.75, -0.5, 0.0, 0.25):
        result = asian_handicap(matrix, line)
        assert sum(result.values()) == pytest.approx(1.0)