"""The experiment driver decides what gets compared, so its rules are tested too.

``experiments/sprint02.py`` is a script, not a package module; it is loaded here the same
way a reviewer would run it, so the test covers the file that actually produced the
artifacts rather than a copy of its logic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from smartgate.dataset import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("sprint02")


def test_the_paired_subsample_is_the_same_for_every_model(driver):
    """0.6B and 14B must be scored on identical topics or the comparison means nothing."""
    samples = generate_dataset(50, seed=3)
    first, meta = driver._paired_subsample(samples, 12, seed=1)
    second, _ = driver._paired_subsample(samples, 12, seed=1)
    assert [s.topic_id for s in first] == [s.topic_id for s in second]
    assert meta["size"] == 12 and meta["of"] == 50
    assert meta["topic_ids"] == [s.topic_id for s in first]


def test_the_subsample_records_its_own_positive_count(driver):
    samples = generate_dataset(50, seed=3)
    subset, meta = driver._paired_subsample(samples, 20, seed=5)
    assert meta["n_positive"] == sum(s.label for s in subset)


def test_no_subsample_means_the_whole_period(driver):
    samples = generate_dataset(10, seed=3)
    subset, meta = driver._paired_subsample(samples, 0, seed=1)
    assert meta is None and len(subset) == 10
    subset, meta = driver._paired_subsample(samples, 99, seed=1)
    assert meta is None and len(subset) == 10


def test_the_test_stage_refuses_to_run_without_a_frozen_config(driver, tmp_path, monkeypatch):
    """Test-period numbers may only be produced with parameters frozen on dev."""
    monkeypatch.setattr(driver, "EVAL", tmp_path)
    with pytest.raises(SystemExit):
        driver._load_frozen()


def test_the_detector_grid_never_tunes_the_warmup(driver):
    """Warmup defines the baseline the labels are computed against; tuning it moves the target."""
    warmups = {params.warmup for _, params in driver.DETECTOR_GRID}
    assert len(warmups) == 1


def test_the_precision_floor_is_a_constant_not_a_result(driver):
    assert driver.PRECISION_FLOOR == 0.50


def test_benchmark_workload_is_taken_in_dataset_order(tmp_path):
    """First N alarms, not the N fastest or the N most convincing ones."""
    bench = _load("benchmark_models")
    from smartgate.dataset import save_jsonl

    path = save_jsonl(generate_dataset(40, length=60, seed=8), tmp_path / "d.jsonl")
    frozen = {"detector": "cusum", "detector_params": {}, "decision_horizon": 7}
    work = bench._workload(str(path), 5, frozen)
    assert len(work) == 5
    ids = [s.topic_id for s, _, _ in work]
    assert ids == sorted(ids)
