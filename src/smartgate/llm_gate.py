"""Semantic gate: Qwen3 (served by Ollama) re-ranks statistical alarms.

Architecture decision (see Report.md, DECISION-04): the LLM never scans the firehose.
The statistical stage fires a *candidate*, and only candidates reach the model. This
keeps token cost proportional to the alarm rate instead of the topic count, and it
means the LLM can only ever improve precision — recall is owned by the cheap stage.

The model is asked for strict JSON and is called through ``/api/chat`` with Ollama's
structured-output ``format`` schema (https://docs.ollama.com/api/introduction).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .config import OllamaConfig
from .dataset import Sample
from .ollama_client import ChatResult, OllamaClient, OllamaError

SYSTEM_PROMPT = (
    "You are a trend analyst. You receive a topic, a short context and a daily mention "
    "series that already triggered a statistical alarm. Decide whether this is a genuine "
    "EMERGING TREND (sustained, spreading growth) or a FALSE ALARM (one-off spike, "
    "seasonal wave, or plain noise). Answer with JSON only."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_emerging": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["is_emerging", "confidence", "reason"],
}


@dataclass
class GateVerdict:
    is_emerging: bool
    confidence: float
    reason: str
    source: str  # "llm" | "heuristic"
    latency_s: float = 0.0
    tokens_per_second: float = 0.0


def build_prompt(sample: Sample, alarm_index: int | None, alarm_score: float) -> str:
    series = ", ".join(f"{v:g}" for v in sample.series)
    return (
        f"Topic: {sample.topic}\n"
        f"Context: {sample.context}\n"
        f"Statistical alarm at day: {alarm_index}\n"
        f"Alarm strength (1.0 = control limit): {alarm_score:.2f}\n"
        f"Daily mentions (day 0 first): {series}\n\n"
        'Reply with JSON: {"is_emerging": bool, "confidence": 0..1, "reason": "<=200 chars"}'
    )


def parse_verdict(text: str) -> dict:
    """Tolerant JSON extraction.

    Qwen3 in non-thinking mode is well behaved, but a gate that crashes on one stray
    token is useless in a pipeline, so we also accept JSON embedded in prose.
    """
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in model answer: {text[:200]!r}") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict) or "is_emerging" not in payload:
        raise ValueError(f"unexpected verdict payload: {payload!r}")
    raw_conf = payload.get("confidence", 0.5)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "is_emerging": bool(payload["is_emerging"]),
        "confidence": min(1.0, max(0.0, confidence)),
        "reason": str(payload.get("reason", ""))[:300],
    }


def heuristic_verdict(sample: Sample, alarm_index: int | None) -> GateVerdict:
    """Deterministic offline fallback used when no Ollama server is reachable.

    It encodes the same rule of thumb we ask the model for — growth must *persist* —
    so the pipeline stays runnable (and testable) on a machine without a GPU.
    """
    series = sample.series
    if alarm_index is None or alarm_index >= len(series) - 1:
        return GateVerdict(False, 0.5, "no post-alarm evidence", "heuristic")
    head = series[: max(1, alarm_index)]
    tail = series[alarm_index:]
    baseline = sum(head) / len(head)
    sustained = sum(1 for v in tail if v > baseline * 1.5) / len(tail)
    is_emerging = sustained >= 0.5
    return GateVerdict(
        is_emerging=is_emerging,
        confidence=min(1.0, max(0.0, sustained)),
        reason=f"{sustained:.0%} of post-alarm days stay above 1.5x baseline",
        source="heuristic",
    )


class LLMGate:
    """Qwen3-backed gate with an explicit, logged fallback path."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        config: OllamaConfig | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.client = client or OllamaClient(config)
        self.allow_fallback = allow_fallback
        self.fallback_count = 0
        self.llm_count = 0
        self.latencies: list[float] = []
        self.token_rates: list[float] = []

    @property
    def model(self) -> str:
        return self.client.config.model

    def judge(self, sample: Sample, alarm_index: int | None, alarm_score: float) -> GateVerdict:
        prompt = build_prompt(sample, alarm_index, alarm_score)
        try:
            result: ChatResult = self.client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                fmt=RESPONSE_SCHEMA,
            )
            parsed = parse_verdict(result.content)
        except (OllamaError, ValueError) as exc:
            if not self.allow_fallback:
                raise
            self.fallback_count += 1
            verdict = heuristic_verdict(sample, alarm_index)
            verdict.reason = f"{verdict.reason} [fallback: {type(exc).__name__}]"
            return verdict
        self.llm_count += 1
        self.latencies.append(result.latency_s)
        if result.tokens_per_second:
            self.token_rates.append(result.tokens_per_second)
        return GateVerdict(
            is_emerging=parsed["is_emerging"],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            source="llm",
            latency_s=result.latency_s,
            tokens_per_second=result.tokens_per_second,
        )

    def stats(self) -> dict:
        def _avg(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return {
            "model": self.model,
            "llm_calls": self.llm_count,
            "fallback_calls": self.fallback_count,
            "mean_latency_s": round(_avg(self.latencies), 3),
            "mean_tokens_per_second": round(_avg(self.token_rates), 2),
        }
