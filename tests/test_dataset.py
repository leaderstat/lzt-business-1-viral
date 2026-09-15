from __future__ import annotations

import pytest

from smartgate.dataset import (
    NEGATIVE_KINDS,
    POSITIVE_KINDS,
    generate_dataset,
    load_jsonl,
    save_jsonl,
)


def test_generation_is_deterministic_for_a_seed():
    a = generate_dataset(30, seed=123)
    b = generate_dataset(30, seed=123)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    assert [s.to_dict() for s in a] != [s.to_dict() for s in generate_dataset(30, seed=124)]


def test_label_balance_and_kinds():
    samples = generate_dataset(100, positive_ratio=0.3, seed=1)
    assert sum(s.label for s in samples) == 30
    for s in samples:
        assert s.kind in (POSITIVE_KINDS if s.label else NEGATIVE_KINDS)
        assert (s.change_index is not None) == bool(s.label)


def test_series_are_non_negative_and_right_length():
    for s in generate_dataset(20, length=45, seed=2):
        assert len(s.series) == 45
        assert min(s.series) >= 0.0


def test_hard_negative_class_is_present():
    kinds = {s.kind for s in generate_dataset(200, seed=3) if s.label == 0}
    assert "growth_then_decay" in kinds, "the corpus must contain the hard negative"


def test_viral_index_is_after_the_change_point():
    for s in generate_dataset(100, seed=4):
        if s.label and s.viral_index is not None:
            assert s.viral_index >= s.change_index


def test_context_is_present_for_every_sample():
    assert all(s.context for s in generate_dataset(20, seed=5))


def test_roundtrip_jsonl(tmp_path):
    samples = generate_dataset(10, seed=6)
    path = save_jsonl(samples, tmp_path / "d.jsonl")
    assert [s.to_dict() for s in load_jsonl(path)] == [s.to_dict() for s in samples]


@pytest.mark.parametrize(
    "kwargs", [{"positive_ratio": 0.0}, {"positive_ratio": 1.0}, {"length": 10}]
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_dataset(10, **kwargs)
