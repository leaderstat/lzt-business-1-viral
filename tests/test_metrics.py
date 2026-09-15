"""Metrics, cross-checked against scikit-learn's published reference values."""

from __future__ import annotations

import pytest

from smartgate import metrics as M

Y_TRUE = [0, 0, 1, 1]
Y_SCORE = [0.1, 0.4, 0.35, 0.8]


def test_roc_auc_matches_reference_value():
    # scikit-learn documents 0.75 for this exact example.
    assert M.roc_auc_score(Y_TRUE, Y_SCORE) == pytest.approx(0.75)


def test_average_precision_matches_reference_value():
    assert M.average_precision_score(Y_TRUE, Y_SCORE) == pytest.approx(0.8333333, rel=1e-5)


def test_ties_give_chance_level_auc():
    assert M.roc_auc_score([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


def test_precision_recall_f1_and_fpr():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    assert M.precision_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert M.recall_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert M.f1_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert M.false_positive_rate(y_true, y_pred) == pytest.approx(1 / 3)


def test_precision_and_recall_at_k():
    y_true = [1, 0, 1, 0, 1]
    y_score = [0.9, 0.8, 0.7, 0.2, 0.1]
    assert M.precision_at_k(y_true, y_score, 2) == pytest.approx(0.5)
    assert M.precision_at_k(y_true, y_score, 3) == pytest.approx(2 / 3)
    assert M.recall_at_k(y_true, y_score, 3) == pytest.approx(2 / 3)


def test_k_larger_than_population_is_clipped():
    assert M.precision_at_k([1, 0], [0.9, 0.1], 10) == pytest.approx(0.5)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        M.precision_at_k([1, 0], [0.9, 0.1], 0)


def test_lead_time_sign_convention():
    assert M.lead_time(10, 20) == 10.0  # detected 10 days before it became obvious
    assert M.lead_time(25, 20) == -5.0  # late
    assert M.lead_time(None, 20) is None  # missed
    assert M.lead_time(10, None) is None  # never went viral


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        M.confusion([1, 0], [1])


def test_evaluate_aggregates_everything():
    rep = M.evaluate([1, 0, 1, 0], [1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], [5.0, None, None, None], k=2)
    assert (rep.tp, rep.fp, rep.tn, rep.fn) == (1, 1, 1, 1)
    assert rep.precision == pytest.approx(0.5)
    assert rep.mean_lead_time == pytest.approx(5.0)
    assert rep.n_detected_positives == 1
    assert set(rep.to_dict()) >= {"roc_auc", "pr_auc", "precision_at_k", "recall_at_k"}


@pytest.mark.parametrize("fn", [M.roc_auc_score, M.average_precision_score])
def test_single_class_returns_nan(fn):
    assert fn([0, 0, 0], [0.1, 0.2, 0.3]) != fn([0, 0, 0], [0.1, 0.2, 0.3])  # NaN != NaN
