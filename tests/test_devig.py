import pytest

from betting_model.devig import devig


def test_devig_no_margin_two_way():
    # fair (no-vig) 50/50 odds are 2.0/2.0 -- de-vig should be a no-op
    result = devig(2.0, 2.0)
    assert result == pytest.approx((0.5, 0.5))


def test_devig_two_way_with_margin():
    # implied: 1/1.9 + 1/1.9 = 1.0526 -> ~5.26% margin, split evenly
    result = devig(1.9, 1.9)
    assert result == pytest.approx((0.5, 0.5))


def test_devig_three_way_sums_to_one():
    result = devig(2.5, 3.4, 3.0)
    assert sum(result) == pytest.approx(1.0)
    # order is preserved and the shortest price keeps the highest probability
    assert result[0] > result[2] > result[1]


def test_devig_rejects_invalid_odds():
    with pytest.raises(ValueError):
        devig(1.0, 2.0)

    with pytest.raises(ValueError):
        devig(0.0, 2.0)