"""Detector behaviour, including the regression test for the look-ahead bug."""

from __future__ import annotations

import random

import pytest

from smartgate.config import DetectorConfig
from smartgate.detectors import CUSUMDetector, EWMADetector, build_detector


def flat(n=60, level=50.0, seed=1):
    rng = random.Random(seed)
    return [level + rng.gauss(0, 2) for _ in range(n)]


def with_step(n=60, level=50.0, at=30, jump=40.0, seed=1):
    series = flat(n, level, seed)
    return [v + (jump if i >= at else 0.0) for i, v in enumerate(series)]


@pytest.mark.parametrize("cls", [EWMADetector, CUSUMDetector])
def test_no_alarm_on_stationary_series(cls):
    assert cls().run(flat()).fired is False


@pytest.mark.parametrize("cls", [EWMADetector, CUSUMDetector])
def test_alarm_after_step_change(cls):
    detection = cls().run(with_step(at=30))
    assert detection.fired
    assert detection.index >= 30, "must not fire before the change point"
    assert detection.index <= 36, "must react within a week of the change point"


@pytest.mark.parametrize("cls", [EWMADetector, CUSUMDetector])
def test_trace_is_causal(cls):
    """A value of the statistic must not depend on the future of the series."""
    series = with_step(at=30)
    prefix_trace = cls().score_series(series[:35])
    full_trace = cls().score_series(series)
    assert prefix_trace == pytest.approx(full_trace[:35])


def test_decision_horizon_limits_the_score():
    """Regression: without a horizon the score was the max over the *whole* series.

    That is a look-ahead the production system does not have and it inflated PR-AUC.
    """
    series = with_step(at=30, jump=40.0)
    detector = CUSUMDetector()
    unbounded = detector.run(series)
    bounded = detector.run(series, decision_horizon=3)
    assert bounded.index == unbounded.index
    assert bounded.score < unbounded.score


def test_constant_series_does_not_divide_by_zero():
    detection = EWMADetector().run([10.0] * 40)
    assert detection.fired is False
    assert all(v == pytest.approx(0.0) for v in detection.trace)


def test_warmup_never_fires():
    cfg = DetectorConfig(warmup=10)
    series = [1.0] * 10 + [1000.0] * 30
    detection = build_detector("cusum", cfg).run(series)
    assert detection.index is not None and detection.index >= 10


def test_unknown_detector():
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector("nope")
