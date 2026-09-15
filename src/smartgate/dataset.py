"""Reproducible synthetic corpus of topic-popularity signals.

Sprint 01 has no labelled production data yet, and RULE 4 says we must not touch the
weights before a baseline dataset + evaluation exist. So we build the dataset first:
a seeded generator that produces time series with *known* change points, plus a short
textual context per topic so the LLM gate has something semantic to reason about.

Every series is a daily mention count for one topic over ``length`` days.

Positive class ("emerging trend"): a change point at ``change_index`` after which the
level grows, eventually crossing the "obviously viral" line at ``viral_index``.
Negative class: stationary noise, a one-off spike, or a slow seasonal wave — the three
families that generate most false positives in production.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

VIRAL_MULTIPLIER = 3.0  # a topic is "obviously viral" at 3x its own baseline level

# ``growth_then_decay`` is the hard negative: for the first days it is statistically
# indistinguishable from a real trend (that is the point), it only reveals itself later.
# It is the class the semantic gate has to earn its token budget on.
NEGATIVE_KINDS = ("stationary", "one_off_spike", "seasonal_wave", "growth_then_decay")
POSITIVE_KINDS = ("linear_growth", "exponential_growth", "step_shift")

_TOPIC_WORDS = (
    "ai agents", "quantum ads", "retro sneakers", "cold brew matcha", "vertical farming",
    "ambient computing", "local llm", "solid state battery", "micro drama", "sleep tech",
    "silent walking", "edge inference", "rewilding", "modular housing", "creatine gummies",
    "voice cloning", "desk treadmill", "carbon concrete", "slow travel", "fermented soda",
)


@dataclass
class Sample:
    """One labelled topic."""

    topic_id: str
    topic: str
    kind: str
    label: int  # 1 = emerging trend, 0 = noise
    series: list[float]
    change_index: int | None
    viral_index: int | None
    context: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Sample":
        return cls(**payload)


def _viral_index(series: Sequence[float], baseline_level: float) -> int | None:
    """First index where the signal is ``VIRAL_MULTIPLIER`` x baseline and stays there."""
    limit = baseline_level * VIRAL_MULTIPLIER
    for i in range(len(series) - 2):
        if series[i] >= limit and series[i + 1] >= limit and series[i + 2] >= limit:
            return i
    return None


def _noise(rng: random.Random, level: float) -> float:
    return rng.gauss(0.0, max(1.0, math.sqrt(level)))


def generate_sample(rng: random.Random, index: int, length: int, positive: bool) -> Sample:
    base = rng.uniform(20.0, 120.0)
    topic = f"{rng.choice(_TOPIC_WORDS)} #{index}"
    kind = rng.choice(POSITIVE_KINDS if positive else NEGATIVE_KINDS)
    series = [max(0.0, base + _noise(rng, base)) for _ in range(length)]
    change_index: int | None = None

    if positive:
        change_index = rng.randrange(length // 3, int(length * 0.6))
        growth = rng.uniform(0.06, 0.18)
        step = rng.uniform(1.6, 3.2)
        slope = base * rng.uniform(0.05, 0.15)
        for i in range(change_index, length):
            t = i - change_index
            if kind == "linear_growth":
                series[i] += slope * t
            elif kind == "exponential_growth":
                series[i] += base * (math.exp(growth * t) - 1)
            else:  # step_shift
                series[i] += base * (step - 1)
    elif kind == "one_off_spike":
        spike_at = rng.randrange(length // 3, length - 3)
        series[spike_at] += base * rng.uniform(2.5, 5.0)  # loud but not sustained
    elif kind == "growth_then_decay":
        start = rng.randrange(length // 4, length // 2)
        peak = start + rng.randrange(5, 12)
        amp = base * rng.uniform(1.5, 3.0)
        for i in range(start, length):
            if i <= peak:
                series[i] += amp * (i - start) / max(1, peak - start)
            else:
                series[i] += amp * math.exp(-(i - peak) / rng.uniform(2.5, 5.0))
    elif kind == "seasonal_wave":
        amp, period = base * rng.uniform(0.15, 0.35), rng.uniform(6.0, 12.0)
        series = [v + amp * math.sin(2 * math.pi * i / period) for i, v in enumerate(series)]

    series = [round(max(0.0, v), 3) for v in series]
    viral = _viral_index(series, base) if positive else None
    context = _context_for(topic, kind, positive)
    return Sample(
        topic_id=f"t{index:04d}",
        topic=topic,
        kind=kind,
        label=1 if positive else 0,
        series=series,
        change_index=change_index,
        viral_index=viral,
        context=context,
        meta={"baseline_level": round(base, 3)},
    )


def _context_for(topic: str, kind: str, positive: bool) -> str:
    """Short human-readable context; the only input the LLM gate gets besides the series."""
    if positive:
        return (
            f"Topic '{topic}': mentions come from a growing number of distinct communities, "
            "new independent accounts keep joining, and no single account exceeds 5% of volume."
        )
    if kind == "growth_then_decay":
        return (
            f"Topic '{topic}': the rise is driven by one paid campaign inside a single "
            "community; 80% of mentions come from 5 accounts, no organic spread so far."
        )
    if kind == "one_off_spike":
        return (
            f"Topic '{topic}': a single large post drove almost all mentions in one day, "
            "activity returned to the usual level right after."
        )
    if kind == "seasonal_wave":
        return (
            f"Topic '{topic}': mentions oscillate with a weekly rhythm that repeats every year, "
            "no new communities involved."
        )
    return f"Topic '{topic}': steady background chatter from the same small set of accounts."


def generate_dataset(
    n_samples: int = 200, length: int = 90, positive_ratio: float = 0.3, seed: int = 20240501
) -> list[Sample]:
    """Deterministic for a given ``seed`` — regenerate any reported number exactly."""
    if not 0.0 < positive_ratio < 1.0:
        raise ValueError("positive_ratio must be in (0, 1)")
    if length < 30:
        raise ValueError("length must be >= 30 to leave room for warmup + change point")
    rng = random.Random(seed)
    n_pos = round(n_samples * positive_ratio)
    flags = [True] * n_pos + [False] * (n_samples - n_pos)
    rng.shuffle(flags)
    return [generate_sample(rng, i, length, positive) for i, positive in enumerate(flags)]


def save_jsonl(samples: Sequence[Sample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[Sample]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Sample.from_dict(json.loads(line)) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[Sample]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Sample.from_dict(json.loads(line))
