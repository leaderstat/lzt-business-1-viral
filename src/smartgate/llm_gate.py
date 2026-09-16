"""Semantic gate: Qwen3 (served by Ollama) re-ranks statistical alarms.

Architecture decision (see Report.md, DECISION-04): the LLM never scans the firehose.
The statistical stage fires a *candidate*, and only candidates reach the model. This
keeps token cost proportional to the alarm rate instead of the topic count, and it
means the LLM can only ever improve precision — recall is owned by the cheap stage.

The model is asked for strict JSON and is called through ``/api/chat`` with Ollama's
structured-output ``format`` schema (https://docs.ollama.com/api/introduction).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import OllamaConfig
from .dataset import Sample
from .ollama_client import ChatResult, OllamaClient, OllamaError

SYSTEM_PROMPT = (
    "You are a trend analyst. You receive independent semantic evidence and a pageview "
    "series that already triggered a statistical alarm. Decide whether this is a genuine "
    "EMERGING TREND (sustained, spreading growth) or a FALSE ALARM (one-off spike, "
    "seasonal wave, or plain noise). Treat the article description as background, not as "
    "proof of growth. Answer with JSON only."
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


@dataclass(frozen=True)
class GateFeatures:
    """Which evidence the gate is allowed to see.

    Sprint 02 ablation (brief PHASE: feature ablation): the gate's win in Sprint 01 was
    reported as a single number, so it was impossible to tell whether Qwen was reading the
    numeric series, the semantic context, or just the alarm strength. Each field here can
    be switched off independently, and the same prompt builder serves every variant — a
    per-variant prompt would make the comparison meaningless.
    """

    use_series: bool = True
    use_context: bool = True
    use_alarm: bool = True

    @property
    def name(self) -> str:
        on = [n for n, v in (("series", self.use_series), ("context", self.use_context),
                             ("alarm", self.use_alarm)) if v]
        return "+".join(on) if on else "none"

    def to_dict(self) -> dict:
        return {
            "use_series": self.use_series,
            "use_context": self.use_context,
            "use_alarm": self.use_alarm,
        }


@dataclass
class GateVerdict:
    is_emerging: bool
    confidence: float
    reason: str
    source: str  # "llm" | "heuristic"
    latency_s: float = 0.0
    tokens_per_second: float = 0.0


def build_prompt(
    sample: Sample,
    alarm_index: int | None,
    alarm_score: float,
    features: GateFeatures | None = None,
) -> str:
    """One prompt template for every variant; disabled features drop their line entirely.

    Disabled evidence is *omitted* rather than blanked out: a line reading
    ``Context: (hidden)`` is itself information, and it changes the task the model sees.
    """
    features = features or GateFeatures()
    lines = [f"Topic: {sample.topic}"]
    if features.use_context and sample.context:
        lines.append("SEMANTIC EVIDENCE (independent of the time series)")
        lines.append(f"Context: {sample.context}")
    if features.use_alarm or features.use_series:
        lines.append("NUMERIC EVIDENCE (pageview time series)")
    if features.use_alarm:
        lines.append(f"Statistical alarm at day: {alarm_index}")
        lines.append(f"Alarm strength (1.0 = control limit): {alarm_score:.2f}")
    if features.use_series:
        series = ", ".join(f"{v:g}" for v in sample.series)
        lines.append(f"Daily mentions (day 0 first): {series}")
    lines.append("")
    if features.use_context and (features.use_alarm or features.use_series):
        lines.append(
            "Use both evidence channels: semantics explain what the topic is; only the "
            "numeric evidence establishes whether attention is sustained and emerging."
        )
    lines.append(
        'Reply with JSON: {"is_emerging": bool, "confidence": 0..1, "reason": "<=200 chars"}'
    )
    return "\n".join(lines)


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


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


class LLMGate:
    """Qwen3-backed gate with an explicit, logged fallback path."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        config: OllamaConfig | None = None,
        allow_fallback: bool = True,
        features: GateFeatures | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.client = client or OllamaClient(config)
        self.allow_fallback = allow_fallback
        self.features = features or GateFeatures()
        # Verdicts are deterministic (temperature 0, fixed seed), so caching them by the
        # exact request is not an approximation — it is memoisation. It is what makes a
        # five-variant ablation on a CPU-only box affordable (backlog S2-14).
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fallback_count = 0
        self.llm_count = 0
        self.cache_hits = 0
        self.transport_errors = 0
        self.invalid_json = 0
        self.latencies: list[float] = []
        self.token_rates: list[float] = []
        self.timings: list[dict] = []

    @property
    def model(self) -> str:
        return self.client.config.model

    def _cache_path(self, prompt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {
                "model": self.client.config.model,
                "think": self.client.config.think,
                "options": self.client.config.options(),
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def judge(self, sample: Sample, alarm_index: int | None, alarm_score: float) -> GateVerdict:
        prompt = build_prompt(sample, alarm_index, alarm_score, self.features)
        cache = self._cache_path(prompt)
        if cache is not None and cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.cache_hits += 1
            self.llm_count += 1
            self.latencies.append(payload.get("latency_s", 0.0))
            if payload.get("tokens_per_second"):
                self.token_rates.append(payload["tokens_per_second"])
            return GateVerdict(
                is_emerging=payload["is_emerging"],
                confidence=payload["confidence"],
                reason=payload["reason"],
                source="llm",
                latency_s=payload.get("latency_s", 0.0),
                tokens_per_second=payload.get("tokens_per_second", 0.0),
            )
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
            # Split the two failure modes: PHASE 9 asks for "failure rate" and
            # "JSON validity" separately, and they have different owners — one is the
            # runtime, the other is the model.
            if isinstance(exc, OllamaError):
                self.transport_errors += 1
            else:
                self.invalid_json += 1
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
        self.timings.append(result.timings_s)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    {
                        **parsed,
                        "latency_s": result.latency_s,
                        "tokens_per_second": result.tokens_per_second,
                        "eval_count": result.eval_count,
                    }
                ),
                encoding="utf-8",
            )
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

        attempts = self.llm_count + self.fallback_count
        return {
            "model": self.model,
            "features": self.features.to_dict(),
            "llm_calls": self.llm_count,
            "cache_hits": self.cache_hits,
            "fallback_calls": self.fallback_count,
            "transport_errors": self.transport_errors,
            "invalid_json": self.invalid_json,
            "failure_rate": round(self.transport_errors / attempts, 4) if attempts else 0.0,
            "json_validity": round(1 - self.invalid_json / attempts, 4) if attempts else 1.0,
            "mean_latency_s": round(_avg(self.latencies), 3),
            "p95_latency_s": round(_percentile(self.latencies, 95), 3),
            "mean_tokens_per_second": round(_avg(self.token_rates), 2),
        }
