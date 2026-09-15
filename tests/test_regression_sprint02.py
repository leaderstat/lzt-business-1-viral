"""Sprint-02 regression suite.

PHASE 10 of the brief names six properties that must keep holding. Each has a section
here. They are regressions, not features: every one of them is cheap to break silently
with an innocent-looking refactor, and none of them shows up as a crash — they show up as
a metric that is too good.
"""

from __future__ import annotations

import json

import pytest

from conftest import chat_response
from smartgate.config import SMOKE_MODEL, TARGET_MODEL, DetectorConfig, OllamaConfig, PipelineConfig
from smartgate.dataset import generate_dataset, load_jsonl, save_jsonl
from smartgate.detectors import build_detector
from smartgate.experiments import (
    _ARM_FEATURES,
    ABLATION_ARMS,
    HeuristicGate,
    choose_operating_point,
    gpu_report,
    precision_at_k_curve,
    run_arm,
)
from smartgate.llm_gate import RESPONSE_SCHEMA, GateFeatures, LLMGate, build_prompt
from smartgate.pipeline import run_pipeline

DETECTORS = ("threshold", "ewma", "cusum")


@pytest.fixture(scope="module")
def corpus():
    return generate_dataset(40, length=60, seed=4242)


# --------------------------------------------------------------- temporal causality
@pytest.mark.parametrize("name", DETECTORS)
def test_alarm_day_does_not_depend_on_data_that_arrives_later(name, corpus):
    """Truncating the series after the alarm must not move the alarm.

    Sprint 01 shipped a bug of exactly this shape (the warmup width depended on the total
    length of the series), and it was invisible in every metric except a causality check.
    """
    detector = build_detector(name, DetectorConfig())
    for sample in corpus:
        full = detector.run(sample.series, decision_horizon=7)
        if not full.fired:
            continue
        cut = full.index + 7 + 1
        truncated = detector.run(sample.series[:cut], decision_horizon=7)
        assert truncated.index == full.index
        assert truncated.score == pytest.approx(full.score)


@pytest.mark.parametrize("arm", [name for name, _ in ABLATION_ARMS])
def test_no_ablation_arm_can_see_past_the_decision_point(arm, corpus, monkeypatch):
    """Whatever the gate is given, it is given a prefix of the series, never the tail."""
    seen: list[int] = []
    original = LLMGate.judge
    heuristic_original = HeuristicGate.judge

    def spy(self, sample, alarm_index, alarm_score):
        seen.append(len(sample.series))
        assert alarm_index is None or len(sample.series) <= alarm_index + 7 + 1
        return heuristic_original(self, sample, alarm_index, alarm_score)

    monkeypatch.setattr(LLMGate, "judge", spy)
    monkeypatch.setattr(HeuristicGate, "judge", spy)
    config = PipelineConfig(detector="cusum", decision_horizon=7, top_k=10)
    run_arm(arm, corpus, config, cache_dir=None)
    assert original is not LLMGate.judge  # the spy really was installed
    if arm != "stats_only":
        assert seen


# ------------------------------------------------------------------ decision horizon
def test_a_longer_horizon_never_lowers_the_ranking_score(corpus):
    """The score is a running maximum over a widening window, so it must be monotone.

    If this ever fails, the horizon has stopped being "how long we are allowed to wait"
    and has become a free hyper-parameter to tune a metric with.
    """
    detector = build_detector("cusum", DetectorConfig())
    for sample in corpus:
        scores = [detector.run(sample.series, decision_horizon=h).score for h in (3, 5, 7, 10, 14)]
        assert scores == sorted(scores)


def test_horizon_widens_the_evidence_given_to_the_gate(fake_ollama):
    """The horizon must actually reach the gate, not just the detector's own score."""
    lengths: dict[int, int] = {}
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": true, "confidence": 0.9, "reason": "x"}')
    )
    samples = generate_dataset(6, length=60, seed=11)
    for horizon in (3, 14):
        gate = LLMGate(client=fake_ollama.client(), allow_fallback=False)
        config = PipelineConfig(
            detector="cusum",
            decision_horizon=horizon,
            ollama=OllamaConfig(host=fake_ollama.host),
        )
        run_pipeline(samples, config, gate=gate)
        prompts = [
            r["payload"]["messages"][-1]["content"]
            for r in fake_ollama.requests
            if r["path"].endswith("/chat")
        ]
        lengths[horizon] = len(prompts[-1])
        fake_ollama.requests.clear()
    assert lengths[14] > lengths[3]


# ------------------------------------------------------------------ baseline leakage
def test_baseline_is_estimated_on_the_warmup_window_only():
    """A baseline that sees the surge normalises the surge away — or inflates it."""
    quiet = [100.0] * 14
    detector = build_detector("cusum", DetectorConfig(warmup=14))
    calm = detector._baseline(quiet + [100.0] * 46)
    surging = detector._baseline(quiet + [10_000.0] * 46)
    assert calm == surging


def test_gate_prompt_contains_no_value_from_after_the_decision_point():
    """The prompt is the last place leakage can hide: it is a string, not a typed object."""
    samples = generate_dataset(20, length=60, seed=5)
    detector = build_detector("cusum", DetectorConfig())
    checked = 0
    for sample in samples:
        detection = detector.run(sample.series, 7)
        if not detection.fired:
            continue
        cut = min(detection.index + 8, len(sample.series))
        future = sample.series[cut:]
        from dataclasses import replace

        prompt = build_prompt(replace(sample, series=sample.series[:cut]), detection.index, 1.0)
        line = next(ln for ln in prompt.splitlines() if ln.startswith("Daily mentions"))
        shown = [v.strip() for v in line.split(": ", 1)[1].split(",")]
        assert len(shown) == cut
        # a distinctive future value must not appear in the visible series
        assert all(f"{v:g}" not in shown for v in future[:5])
        checked += 1
    assert checked


# ----------------------------------------------------------------- structured output
def test_gate_asks_the_server_for_a_json_schema(fake_ollama):
    """Ollama structured output is what makes JSON validity a runtime guarantee."""
    fake_ollama.set("chat", chat_response('{"is_emerging": true, "confidence": 1, "reason": "ok"}'))
    gate = LLMGate(client=fake_ollama.client(), allow_fallback=False)
    gate.judge(generate_dataset(1, length=40, seed=1)[0], 20, 1.5)
    payload = fake_ollama.requests[-1]["payload"]
    assert payload["format"] == RESPONSE_SCHEMA
    assert set(RESPONSE_SCHEMA["required"]) == {"is_emerging", "confidence", "reason"}


def test_invalid_json_and_transport_failures_are_counted_separately(fake_ollama):
    """PHASE 9 reports them as different numbers because they have different owners."""
    fake_ollama.set("chat", chat_response("not json at all"))
    gate = LLMGate(client=fake_ollama.client())
    sample = generate_dataset(1, length=40, seed=2)[0]
    gate.judge(sample, 20, 1.5)
    stats = gate.stats()
    assert stats["invalid_json"] == 1 and stats["transport_errors"] == 0
    assert stats["json_validity"] == 0.0 and stats["failure_rate"] == 0.0


# --------------------------------------------------------------- Qwen model selection
def test_model_ids_are_the_documented_ollama_tags():
    assert SMOKE_MODEL == "qwen3:0.6b"
    assert TARGET_MODEL == "qwen3:14b"


def test_the_configured_model_is_the_one_that_gets_called(fake_ollama):
    """A comparison is worthless if the request silently carries a different tag."""
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": false, "confidence": 0.2, "reason": ""}')
    )
    client = fake_ollama.client()
    object.__setattr__(client.config, "model", TARGET_MODEL)
    gate = LLMGate(client=client, allow_fallback=False)
    gate.judge(generate_dataset(1, length=40, seed=3)[0], 20, 1.2)
    assert fake_ollama.requests[-1]["payload"]["model"] == TARGET_MODEL
    assert gate.stats()["model"] == TARGET_MODEL


def test_every_ablation_arm_uses_the_shared_prompt_builder():
    """One template for all arms: a per-arm prompt would make the ablation meaningless."""
    sample = generate_dataset(1, length=40, seed=6)[0]
    prompts = {
        arm: build_prompt(sample, 20, 1.3, features) for arm, features in _ARM_FEATURES.items()
    }
    assert prompts["llm_context_only"].count("Daily mentions") == 0
    assert prompts["llm_series_only"].count("Context:") == 0
    assert prompts["llm_full"].count("Context:") == 1
    # disabled evidence is omitted, not replaced by a placeholder that leaks its absence
    assert "hidden" not in " ".join(prompts.values()).lower()


def test_verdict_cache_is_keyed_by_model_and_prompt(tmp_path, fake_ollama):
    """Two models must never share a cached verdict."""
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": true, "confidence": 0.7, "reason": "y"}')
    )
    sample = generate_dataset(1, length=40, seed=7)[0]
    gate = LLMGate(client=fake_ollama.client(), cache_dir=tmp_path, allow_fallback=False)
    first = gate._cache_path(build_prompt(sample, 20, 1.0, GateFeatures()))
    object.__setattr__(gate.client.config, "model", TARGET_MODEL)
    second = gate._cache_path(build_prompt(sample, 20, 1.0, GateFeatures()))
    assert first != second

    gate.judge(sample, 20, 1.0)
    assert gate.cache_hits == 0
    gate.judge(sample, 20, 1.0)
    assert gate.cache_hits == 1 and gate.llm_count == 2
    assert len(fake_ollama.requests) == 1


# ------------------------------------------------------------- dataset reproducibility
def test_synthetic_corpus_is_byte_identical_for_a_seed(tmp_path):
    a = save_jsonl(generate_dataset(30, seed=99), tmp_path / "a.jsonl").read_bytes()
    b = save_jsonl(generate_dataset(30, seed=99), tmp_path / "b.jsonl").read_bytes()
    assert a == b


def test_saved_corpus_round_trips_without_losing_metadata(tmp_path):
    original = generate_dataset(10, seed=17)
    path = save_jsonl(original, tmp_path / "c.jsonl")
    reloaded = load_jsonl(path)
    assert [s.to_dict() for s in reloaded] == [s.to_dict() for s in original]


def test_sprint01_numbers_are_still_reproducible_from_the_committed_corpus():
    """Sprint 01's headline corpus must keep producing Sprint 01's detector numbers."""
    samples = load_jsonl("artifacts/dataset_eval60.jsonl")
    assert len(samples) == 60 and sum(s.label for s in samples) == 18
    report = run_pipeline(
        samples,
        PipelineConfig(detector="cusum", use_llm_gate=False, top_k=10, decision_horizon=7),
    ).detector_only_report
    assert report.recall == pytest.approx(1.0)
    assert report.precision == pytest.approx(0.409, abs=0.001)
    assert report.pr_auc == pytest.approx(0.435, abs=0.001)
    assert report.precision_at_k == pytest.approx(0.200, abs=0.001)


# ------------------------------------------------------------------ reporting honesty
def test_gpu_report_never_invents_a_gpu():
    report = gpu_report()
    if not report["available"]:
        assert report["note"] == "GPU benchmark unavailable in this environment"
    else:  # pragma: no cover - no GPU in CI
        assert report["gpu"] and report["vram"]


def test_operating_point_prefers_the_largest_k_above_the_precision_floor():
    curve = {
        "5": {"precision_at_k": 0.8, "recall_at_k": 0.2},
        "10": {"precision_at_k": 0.6, "recall_at_k": 0.3},
        "20": {"precision_at_k": 0.3, "recall_at_k": 0.4},
    }
    assert choose_operating_point(curve, 0.5)["k"] == 10
    assert choose_operating_point(curve, 0.95)["k"] == 5  # nothing clears the floor


def test_precision_at_k_curve_covers_more_than_one_k(corpus):
    result = run_pipeline(corpus, PipelineConfig(use_llm_gate=False, detector="cusum"))
    curve = precision_at_k_curve(result)
    assert len(curve) > 1 and "10" in curve
    assert json.dumps(curve)  # artifact-serialisable
