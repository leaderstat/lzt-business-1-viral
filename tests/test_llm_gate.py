"""The semantic gate: prompt content, tolerant parsing and the fallback contract."""

from __future__ import annotations

import pytest

from conftest import chat_response
from smartgate.dataset import generate_dataset
from smartgate.llm_gate import GateFeatures, LLMGate, build_prompt, heuristic_verdict, parse_verdict
from smartgate.ollama_client import OllamaError

SAMPLE = generate_dataset(10, seed=99)[0]


def test_prompt_contains_the_evidence_the_model_needs():
    prompt = build_prompt(SAMPLE, 30, 2.5)
    assert SAMPLE.topic in prompt
    assert SAMPLE.context in prompt
    assert "Statistical alarm at day: 30" in prompt
    assert "is_emerging" in prompt


@pytest.mark.parametrize(
    "text",
    [
        '{"is_emerging": true, "confidence": 0.8, "reason": "spreading"}',
        '```json\n{"is_emerging": true, "confidence": 0.8, "reason": "spreading"}\n```',
        'Sure!\n{"is_emerging": true, "confidence": 0.8, "reason": "spreading"}\nHope that helps',
        '<think>hmm</think>{"is_emerging": true, "confidence": 0.8, "reason": "spreading"}',
    ],
)
def test_parse_verdict_is_tolerant(text):
    assert parse_verdict(text) == {"is_emerging": True, "confidence": 0.8, "reason": "spreading"}


def test_confidence_is_clamped_and_coerced():
    assert parse_verdict('{"is_emerging": false, "confidence": 7}')["confidence"] == 1.0
    assert parse_verdict('{"is_emerging": false, "confidence": -1}')["confidence"] == 0.0
    assert parse_verdict('{"is_emerging": false, "confidence": "high"}')["confidence"] == 0.5


def test_full_prompt_separates_semantic_and_numeric_evidence():
    prompt = build_prompt(SAMPLE, 20, 1.3, GateFeatures())
    assert "SEMANTIC EVIDENCE (independent of the time series)" in prompt
    assert "NUMERIC EVIDENCE (pageview time series)" in prompt
    assert prompt.index("SEMANTIC EVIDENCE") < prompt.index("NUMERIC EVIDENCE")
    assert "Use both evidence channels" in prompt


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '{"foo": 1}'])
def test_parse_verdict_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_gate_uses_the_llm_when_the_server_answers(fake_ollama):
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": true, "confidence": 0.9, "reason": "r"}')
    )
    gate = LLMGate(client=fake_ollama.client())
    verdict = gate.judge(SAMPLE, 30, 2.0)
    assert verdict.is_emerging is True
    assert verdict.source == "llm"
    assert gate.stats()["llm_calls"] == 1
    assert gate.stats()["fallback_calls"] == 0


def test_gate_falls_back_when_the_server_is_down(fake_ollama):
    fake_ollama.set("chat", 500)
    gate = LLMGate(client=fake_ollama.client())
    verdict = gate.judge(SAMPLE, 30, 2.0)
    assert verdict.source == "heuristic"
    assert "fallback" in verdict.reason
    assert gate.stats()["fallback_calls"] == 1


def test_gate_falls_back_on_unparseable_output(fake_ollama):
    fake_ollama.set("chat", chat_response("I cannot answer that"))
    assert LLMGate(client=fake_ollama.client()).judge(SAMPLE, 30, 2.0).source == "heuristic"


def test_strict_mode_propagates_the_error(fake_ollama):
    fake_ollama.set("chat", 500)
    gate = LLMGate(client=fake_ollama.client(), allow_fallback=False)
    with pytest.raises(OllamaError):
        gate.judge(SAMPLE, 30, 2.0)


def test_heuristic_separates_sustained_growth_from_a_spike():
    series = [10.0] * 30 + [60.0] * 30
    spike = [10.0] * 30 + [60.0] + [10.0] * 29
    fake = type(SAMPLE)(**{**SAMPLE.to_dict(), "series": series})
    fake_spike = type(SAMPLE)(**{**SAMPLE.to_dict(), "series": spike})
    assert heuristic_verdict(fake, 30).is_emerging is True
    assert heuristic_verdict(fake_spike, 30).is_emerging is False


def test_heuristic_without_post_alarm_evidence():
    assert heuristic_verdict(SAMPLE, None).is_emerging is False
