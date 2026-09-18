import pytest

from betting_model.calibration import brier_score, log_loss, reliability_table


def test_brier_score_perfect_predictions():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_score_worst_case():
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_brier_score_uninformative_coin_flip():
    assert brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.25)


def test_log_loss_near_perfect_predictions_near_zero():
    assert log_loss([0.999999, 0.000001], [1, 0]) < 1e-4


def test_log_loss_penalizes_confident_wrong_prediction_more_than_unsure():
    confident_wrong = log_loss([0.99], [0])
    unsure = log_loss([0.5], [0])
    assert confident_wrong > unsure


def test_reliability_table_bucket_matches_actual_rate():
    predicted = [0.55] * 10
    actual = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 60% actual in the 0.5-0.6 bin
    table = reliability_table(predicted, actual, n_bins=10)

    nonzero = table[table["n"] > 0]
    assert len(nonzero) == 1
    assert nonzero.iloc[0]["n"] == 10
    assert nonzero.iloc[0]["actual_rate"] == pytest.approx(0.6)
    assert nonzero.iloc[0]["mean_predicted"] == pytest.approx(0.55)