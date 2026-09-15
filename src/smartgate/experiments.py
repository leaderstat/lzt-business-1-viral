"""Sprint-02 experiment harness: ablations, operating point, bootstrap, benchmarks.

Everything in here answers one of the questions the sprint brief asks by name:

* *feature ablation* — which piece of evidence does the semantic gate actually use?
* *operating point* — which ``K`` should the product ship? ("do not assume K=10")
* *0.6B vs 14B* — same corpus, same prompt, same decoding, only the weights change.
* *performance* — latency / tokens-per-second / memory / failure rate / JSON validity.

Two rules are enforced structurally rather than by discipline:

1. Every variant is scored on the *same* samples with the *same* prompt builder. A
   variant-specific prompt would turn an ablation into two unrelated experiments.
2. Anything tuned is tuned on the development period and then frozen; the test period is
   scored once, with the frozen values passed in.
"""

from __future__ import annotations

import os
import platform
import random
import resource
import shutil
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .dataset import Sample
from .llm_gate import GateFeatures, LLMGate, heuristic_verdict
from .metrics import precision_at_k, recall_at_k
from .pipeline import RunResult, run_pipeline

DEFAULT_CACHE = Path("artifacts/cache/verdicts")


# --------------------------------------------------------------------------- gate stubs
class HeuristicGate(LLMGate):
    """The deterministic fallback promoted to a first-class ablation arm.

    It is the honest control for "does the win come from the language model, or would any
    post-alarm persistence check do the same job?" Without this arm a reported gate win is
    uninterpretable.
    """

    def __init__(self) -> None:
        self.features = GateFeatures(use_series=True, use_context=False, use_alarm=True)
        self.fallback_count = 0
        self.llm_count = 0
        self.cache_hits = 0
        self.transport_errors = 0
        self.invalid_json = 0
        self.latencies = []
        self.token_rates = []
        self._model = "heuristic"

    @property
    def model(self) -> str:
        return self._model

    def judge(self, sample: Sample, alarm_index, alarm_score):  # noqa: ANN001 - base sig
        self.llm_count += 1
        return heuristic_verdict(sample, alarm_index)


def make_gate(
    config: PipelineConfig,
    features: GateFeatures,
    cache_dir: Path | None = DEFAULT_CACHE,
    strict: bool = True,
) -> LLMGate:
    """Gate wired to the run's model and a per-model verdict cache.

    ``strict=True`` by default on purpose: in an *experiment* a silent fallback to the
    heuristic would contaminate the arm being measured with the arm it is compared to.
    """
    sub = None if cache_dir is None else Path(cache_dir) / config.ollama.model.replace(":", "-")
    return LLMGate(
        config=config.ollama,
        allow_fallback=not strict,
        features=features,
        cache_dir=sub,
    )


# --------------------------------------------------------------------------- ablation
# Frozen ablation grid. ``stats_only`` and ``heuristic`` are the two controls; the three
# LLM arms isolate the two evidence channels the gate is given.
ABLATION_ARMS: tuple[tuple[str, str], ...] = (
    ("stats_only", "detector alone, no gate"),
    ("heuristic", "detector + deterministic persistence check (no LLM)"),
    ("llm_series_only", "gate sees the numeric series and the alarm, no semantic context"),
    ("llm_context_only", "gate sees the topic and its semantic context, no numbers"),
    ("llm_full", "gate sees everything (Sprint 01 configuration)"),
)

_ARM_FEATURES = {
    "llm_series_only": GateFeatures(use_series=True, use_context=False, use_alarm=True),
    "llm_context_only": GateFeatures(use_series=False, use_context=True, use_alarm=True),
    "llm_full": GateFeatures(use_series=True, use_context=True, use_alarm=True),
}


def run_arm(
    arm: str,
    samples: Sequence[Sample],
    config: PipelineConfig,
    cache_dir: Path | None = DEFAULT_CACHE,
) -> RunResult:
    if arm == "stats_only":
        return run_pipeline(samples, replace(config, use_llm_gate=False))
    if arm == "heuristic":
        return run_pipeline(samples, replace(config, use_llm_gate=True), gate=HeuristicGate())
    gate = make_gate(config, _ARM_FEATURES[arm], cache_dir)
    return run_pipeline(samples, replace(config, use_llm_gate=True), gate=gate)


def gate_ablation(
    samples: Sequence[Sample],
    config: PipelineConfig,
    arms: Sequence[str] = tuple(name for name, _ in ABLATION_ARMS),
    cache_dir: Path | None = DEFAULT_CACHE,
    bootstrap: int = 1000,
    seed: int = 7,
) -> dict:
    """Run every arm on one corpus and report metrics with bootstrap intervals."""
    out: dict = {"arms": {}, "descriptions": dict(ABLATION_ARMS), "n_samples": len(samples)}
    for arm in arms:
        result = run_arm(arm, samples, config, cache_dir)
        report = (
            result.detector_only_report if arm == "stats_only" else result.report
        ).to_dict()
        out["arms"][arm] = {
            "metrics": report,
            "gate_stats": result.gate_stats,
            "ci": bootstrap_intervals(result, k=config.top_k, n=bootstrap, seed=seed),
            "precision_at_k_curve": precision_at_k_curve(result),
        }
    return out


# --------------------------------------------------------------------------- statistics
def _arrays(
    result: RunResult, stats_only: bool = False
) -> tuple[list[int], list[int], list[float]]:
    y_true = [r.label for r in result.results]
    if stats_only:
        return y_true, [int(r.detector_fired) for r in result.results], [
            r.detector_score for r in result.results
        ]
    return (
        y_true,
        [r.final_prediction for r in result.results],
        [r.final_score for r in result.results],
    )


def bootstrap_intervals(result: RunResult, k: int = 10, n: int = 1000, seed: int = 7) -> dict:
    """Percentile bootstrap over topics.

    Sprint 01 reported point estimates on 60 topics, where two examples move a metric by
    several points (Report.md §7.6). Resampling topics with replacement gives the width of
    that uncertainty, which is what decides whether "0.72 vs 0.65" is a finding or noise.
    """
    y_true, y_pred, y_score = _arrays(result)
    rng = random.Random(seed)
    size = len(y_true)
    if size == 0 or n <= 0:
        return {}
    draws: dict[str, list[float]] = {"precision": [], "recall": [], "precision_at_k": []}
    for _ in range(n):
        idx = [rng.randrange(size) for _ in range(size)]
        t = [y_true[i] for i in idx]
        p = [y_pred[i] for i in idx]
        s = [y_score[i] for i in idx]
        tp = sum(1 for a, b in zip(t, p) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(t, p) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(t, p) if a == 1 and b == 0)
        draws["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        draws["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
        draws["precision_at_k"].append(precision_at_k(t, s, k))
    return {
        name: {
            "lo": round(_percentile(vals, 2.5), 4),
            "hi": round(_percentile(vals, 97.5), 4),
            "median": round(_percentile(vals, 50), 4),
        }
        for name, vals in draws.items()
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def precision_at_k_curve(result: RunResult, ks: Sequence[int] = ()) -> dict:
    """Precision@K / Recall@K for a range of K — the operating-point evidence.

    The brief is explicit that K=10 must not be assumed. The product question is "how many
    topics can one analyst review per day", and the answer is read off this curve, not
    copied from Sprint 01.
    """
    y_true, _, y_score = _arrays(result)
    ks = ks or tuple(k for k in (5, 10, 15, 20, 25, 30, 40, 50) if k <= len(y_true))
    return {
        str(k): {
            "precision_at_k": round(precision_at_k(y_true, y_score, k), 4),
            "recall_at_k": round(recall_at_k(y_true, y_score, k), 4),
        }
        for k in ks
    }


def choose_operating_point(curve: dict, min_precision: float, max_k: int | None = None) -> dict:
    """Largest K whose Precision@K still clears ``min_precision``.

    Rationale: below the precision floor the analyst stops trusting the queue, so extra
    recall bought there is worth nothing; above it, more K is strictly more trends found.
    The floor is a *product* constant, fixed on the development period, never re-tuned
    after seeing the test period.
    """
    eligible = [
        (int(k), v)
        for k, v in curve.items()
        if v["precision_at_k"] >= min_precision and (max_k is None or int(k) <= max_k)
    ]
    if not eligible:
        best_k, best = max(curve.items(), key=lambda kv: kv[1]["precision_at_k"])
        return {"k": int(best_k), "reason": "no K clears the precision floor", **best}
    k, values = max(eligible, key=lambda kv: kv[0])
    return {"k": k, "reason": f"largest K with Precision@K >= {min_precision}", **values}


# --------------------------------------------------------------------------- environment
def gpu_report() -> dict:
    """Honest hardware report.

    The brief forbids simulating a GPU benchmark. If ``nvidia-smi`` is absent we say so in
    the artifact itself, so no downstream reader can mistake a CPU number for a GPU one.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {
            "available": False,
            "note": "GPU benchmark unavailable in this environment",
        }
    try:  # pragma: no cover - no GPU in CI
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "note": f"nvidia-smi present but failed: {exc}"}
    name, memory, driver = (part.strip() for part in out.splitlines()[0].split(","))
    return {"available": True, "gpu": name, "vram": memory, "driver": driver}


def _total_memory_gb() -> float | None:
    try:
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2
        )
    except (ValueError, OSError):  # pragma: no cover - non-POSIX
        return None


def host_report() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "total_memory_gb": _total_memory_gb(),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "gpu": gpu_report(),
    }


def latency_summary(latencies: Sequence[float]) -> dict:
    if not latencies:
        return {}
    return {
        "n": len(latencies),
        "mean_s": round(statistics.fmean(latencies), 3),
        "median_s": round(statistics.median(latencies), 3),
        "p95_s": round(_percentile(latencies, 95), 3),
        "max_s": round(max(latencies), 3),
    }
