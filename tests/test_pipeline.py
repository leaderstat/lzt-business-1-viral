"""End-to-end pipeline behaviour, plus the live-Ollama integration check."""

from __future__ import annotations

import json
import os

import pytest

from conftest import chat_response
from smartgate.cli import main
from smartgate.config import OllamaConfig, PipelineConfig
from smartgate.dataset import generate_dataset
from smartgate.llm_gate import LLMGate
from smartgate.ollama_client import OllamaClient
from smartgate.pipeline import run_pipeline, save_run, sweep_detectors

SAMPLES = generate_dataset(40, seed=11)


def test_pipeline_without_gate_is_dependency_free():
    result = run_pipeline(SAMPLES, PipelineConfig(detector="cusum", use_llm_gate=False))
    assert len(result.results) == len(SAMPLES)
    assert result.report.to_dict() == result.detector_only_report.to_dict()
    assert 0.0 <= result.report.precision <= 1.0
    assert all(r.gate_source == "none" for r in result.results)


def test_gate_rejection_flips_the_prediction_and_lowers_the_rank(fake_ollama):
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": false, "confidence": 0.9, "reason": "r"}')
    )
    gate = LLMGate(client=fake_ollama.client())
    config = PipelineConfig(detector="cusum", use_llm_gate=True)
    result = run_pipeline(SAMPLES[:5], config, gate=gate)
    reviewed = [r for r in result.results if r.detector_fired]
    assert reviewed, "the detector must fire at least once on this slice"
    for r in reviewed:
        assert r.final_prediction == 0
        assert r.final_score < r.detector_score
        assert r.lead_time is None


def test_gate_only_sees_evidence_up_to_the_decision_point(fake_ollama):
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": true, "confidence": 0.5, "reason": "r"}')
    )
    horizon = 5
    config = PipelineConfig(detector="cusum", use_llm_gate=True, decision_horizon=horizon)
    run_pipeline(SAMPLES[:5], config, gate=LLMGate(client=fake_ollama.client()))
    for request in fake_ollama.requests:
        prompt = request["payload"]["messages"][-1]["content"]
        alarm_day = int(prompt.split("Statistical alarm at day: ")[1].split("\n")[0])
        series = prompt.split("Daily mentions (day 0 first): ")[1].split("\n")[0].split(", ")
        assert len(series) <= alarm_day + horizon + 1, "the gate must not see the future"


def test_lead_time_is_positive_for_early_detections():
    result = run_pipeline(SAMPLES, PipelineConfig(detector="ewma", use_llm_gate=False))
    leads = [r.lead_time for r in result.results if r.label == 1 and r.lead_time is not None]
    assert leads, "at least some positives must be detected before they go viral"
    assert sum(leads) / len(leads) > 0


def test_sweep_compares_all_detectors():
    reports = sweep_detectors(SAMPLES)
    assert set(reports) == {"threshold", "ewma", "cusum"}
    assert all(0.0 <= r.recall <= 1.0 for r in reports.values())


def test_run_artifact_is_json_serialisable(tmp_path):
    result = run_pipeline(SAMPLES, PipelineConfig(use_llm_gate=False))
    path = save_run(result, tmp_path / "run.json")
    payload = json.loads(path.read_text())
    assert payload["config"]["n_samples"] == len(SAMPLES)
    assert "detector_only" in payload["metrics"]
    assert len(payload["results"]) == len(SAMPLES)


def test_cli_dataset_and_sweep(tmp_path, capsys):
    ds = tmp_path / "d.jsonl"
    assert main(["dataset", "--n", "20", "--out", str(ds)]) == 0
    assert ds.exists()
    assert main(["sweep", "--dataset", str(ds), "--out", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "cusum" in out and "PR-AUC" in out


def test_cli_run_without_gate(tmp_path, capsys):
    assert main(["run", "--n", "20", "--no-gate", "--out", str(tmp_path / "r.json")]) == 0
    assert "detector_plus_gate" in capsys.readouterr().out


def test_cli_doctor_reports_unavailable_server(capsys):
    assert main(["doctor", "--host", "http://127.0.0.1:1"]) == 1
    assert "UNAVAILABLE" in capsys.readouterr().out


# --------------------------------------------------------------------- integration
requires_ollama = pytest.mark.skipif(
    not OllamaClient().is_available(),
    reason="no Ollama server on OLLAMA_HOST (start it with `ollama serve`)",
)


@pytest.mark.integration
@requires_ollama
def test_live_qwen3_returns_a_parseable_verdict():
    model = os.environ.get("SMARTGATE_MODEL", "qwen3:0.6b")
    client = OllamaClient(OllamaConfig(model=model, num_predict=96, timeout=300))
    if not client.has_model(model):
        pytest.skip(f"model {model} not pulled")
    gate = LLMGate(client=client, allow_fallback=False)
    obvious_spike = generate_dataset(200, seed=11)
    sample = next(s for s in obvious_spike if s.kind == "one_off_spike")
    verdict = gate.judge(sample, 30, 2.0)
    assert verdict.source == "llm"
    assert isinstance(verdict.is_emerging, bool)
    assert 0.0 <= verdict.confidence <= 1.0
    assert gate.stats()["llm_calls"] == 1
