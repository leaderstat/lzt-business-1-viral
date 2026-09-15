"""Causality, leakage and reproducibility tests for the real-world corpus.

These are the tests the sprint brief asks for by name. They are deliberately written
against the *protocol* (which days may influence which field) rather than against the
numbers, so they keep working when the corpus is rebuilt on new dates.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from smartgate import realworld
from smartgate.dataset import save_jsonl
from smartgate.realworld import (
    BASELINE_DAYS,
    DEV_PERIOD,
    TEST_PERIOD,
    Period,
    build_period_dataset,
    entity_key,
    label_for,
)


class StubClient:
    """Deterministic stand-in for the pageviews API, with a request log.

    The log is the point: several tests below assert on *which days were requested*,
    which is how leakage is caught before it reaches a metric.
    """

    def __init__(self, series_for=None, universe=None):
        self.series_for = series_for or (lambda project, article: None)
        self.universe = universe or ["Alpha", "Beta", "Gamma", "Delta"]
        self.top_calls: list[tuple[str, date]] = []
        self.series_calls: list[tuple[str, str, date, date]] = []

    def top_articles(self, project, day, limit=1000):
        self.top_calls.append((project, day))
        return [f"pad{i}" for i in range(200)] + list(self.universe)

    def daily_series(self, project, article, start, end):
        self.series_calls.append((project, article, start, end))
        days = (end - start).days + 1
        custom = self.series_for(project, article)
        return list(custom)[:days] if custom else [100.0] * days

    def stats(self):
        return {"http_requests": len(self.series_calls)}


SHORT = Period("unit", date(2024, 1, 1), selection_days=2, length_days=20, label_days=10)


def _build(series_for=None, period=SHORT, **kwargs):
    client = StubClient(series_for=series_for)
    samples, meta = build_period_dataset(
        client, period, projects=("en.wikipedia",), max_candidates=10, **kwargs
    )
    return client, samples, meta


# --------------------------------------------------------------- temporal causality
def test_windows_are_contiguous_and_never_overlap():
    for period in (DEV_PERIOD, TEST_PERIOD, SHORT):
        assert period.selection_end < period.observation_start
        assert period.observation_end < period.label_start
        assert period.observation_start == period.selection_end + timedelta(days=1)
        assert period.label_start == period.observation_end + timedelta(days=1)
        span = (period.observation_end - period.observation_start).days + 1
        assert span == period.length_days


def test_test_period_is_strictly_later_than_development_period():
    """A random split would let the model learn from its own future."""
    assert DEV_PERIOD.label_end < TEST_PERIOD.selection_start


def test_entity_selection_only_reads_the_selection_window():
    client, _, _ = _build()
    requested = {day for _, day in client.top_calls}
    assert requested
    assert all(SHORT.selection_start <= day <= SHORT.selection_end for day in requested)


def test_stored_series_stops_at_the_observation_end():
    """The label window is fetched (labels need it) but must never reach the features."""
    _, samples, _ = _build()
    assert samples
    for sample in samples:
        assert len(sample.series) == SHORT.length_days
        assert sample.meta["observation_end"] == SHORT.observation_end.isoformat()


# ------------------------------------------------------------------ baseline leakage
def test_context_is_computed_from_the_warmup_window_only():
    """The gate reads ``context`` before any alarm, so it may only summarise warmup days.

    The probe puts a unique, huge value after the warmup window: if any statistic over the
    post-warmup part of the series leaked into the context string, its magnitude would
    show up there.
    """
    spike = 999_999.0
    series = [100.0] * BASELINE_DAYS + [spike] * (SHORT.length_days - BASELINE_DAYS)
    series += [spike] * SHORT.label_days
    _, samples, _ = _build(series_for=lambda p, a: series)
    assert samples
    for sample in samples:
        assert "999999" not in sample.context.replace(",", "")
        assert "median 100" in sample.context


def test_label_uses_only_the_label_window_and_the_warmup_baseline():
    observation = [100.0] * 20
    quiet = [100.0] * 10
    surge = [400.0] * 10
    assert label_for(observation, quiet)[0] == 0
    assert label_for(observation, surge)[0] == 1


def test_a_spike_that_decays_is_labelled_negative():
    """The hard negative from Sprint 01, now defined on real calendar time."""
    observation = [100.0] * 14 + [900.0] * 6  # loud at the end of the observation window
    label_window = [110.0] * 10  # ... and gone by the label window
    label, evidence = label_for(observation, label_window)
    assert label == 0
    assert evidence["baseline"] == 100.0


def test_sustained_growth_needs_both_level_and_persistence():
    observation = [100.0] * 20
    # Mean clears 3x only because of two enormous days; persistence rejects it.
    bursty = [2000.0, 2000.0] + [100.0] * 8
    assert sum(bursty) / len(bursty) >= 3 * 100.0
    assert label_for(observation, bursty)[0] == 0


def test_label_evidence_lets_a_reviewer_recompute_the_label():
    _, samples, _ = _build(
        series_for=lambda p, a: [100.0] * SHORT.length_days + [500.0] * SHORT.label_days
    )
    for sample in samples:
        ev = sample.meta["label_evidence"]
        assert ev["label_window_ratio"] == pytest.approx(5.0)
        assert sample.label == 1


# ------------------------------------------------------------------- eligibility
def test_low_traffic_entities_are_dropped_using_observation_data_only():
    _, samples, meta = _build(series_for=lambda p, a: [1.0] * 30)
    assert samples == []
    assert meta.dropped["low_baseline"] == meta.n_candidates
    assert meta.eligibility["computed_from"] == "observation window only"


def test_partial_coverage_is_dropped_rather_than_padded():
    _, samples, meta = _build(series_for=lambda p, a: [100.0] * 5)
    assert samples == []
    assert meta.dropped["short_series"] == meta.n_candidates


# --------------------------------------------------------------- reproducibility
def test_same_seed_produces_byte_identical_corpora(tmp_path):
    _, first, _ = _build(seed=123)
    _, second, _ = _build(seed=123)
    a = save_jsonl(first, tmp_path / "a.jsonl").read_bytes()
    b = save_jsonl(second, tmp_path / "b.jsonl").read_bytes()
    assert a == b


def test_a_different_seed_selects_a_different_candidate_set():
    """Sampling must actually depend on the seed, otherwise ``seed`` in the metadata lies."""

    def picked(seed: int) -> list[str]:
        client = StubClient(universe=[f"Topic_{i:03d}" for i in range(50)])
        build_period_dataset(
            client, SHORT, projects=("en.wikipedia",), max_candidates=10, seed=seed
        )
        return [article for _, article, _, _ in client.series_calls]

    assert picked(1) != picked(2)
    assert picked(1) == picked(1)


def test_metadata_records_everything_needed_to_rebuild(tmp_path):
    _, _, meta = _build(seed=99)
    payload = json.loads(realworld.save_meta(meta, tmp_path / "m.json").read_text())
    for key in ("period", "projects", "rank_band", "seed", "label_rule", "eligibility"):
        assert key in payload
    assert payload["seed"] == 99
    assert payload["label_rule"]["viral_multiplier"] == realworld.VIRAL_MULTIPLIER


def test_excluded_entities_never_enter_a_later_period():
    """Entity overlap across splits leaks an entity's behaviour into its own test score."""
    client = StubClient()
    first, _ = build_period_dataset(client, SHORT, projects=("en.wikipedia",), max_candidates=10)
    assert first
    excluded = [entity_key(s) for s in first]
    second_client = StubClient()
    second, _ = build_period_dataset(
        second_client, SHORT, projects=("en.wikipedia",), max_candidates=10, exclude=excluded
    )
    assert not (set(entity_key(s) for s in second) & set(excluded))
