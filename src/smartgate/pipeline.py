"""Sprint-01 pipeline orchestration.

    dataset -> statistical detector -> (optional) Qwen3 gate -> metrics -> artifacts

Everything is deterministic given ``seed`` and the frozen inference mode, so any number
in Report.md can be regenerated with a single command.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import __version__
from .config import DetectorConfig, OllamaConfig, PipelineConfig
from .dataset import Sample, generate_dataset
from .detectors import build_detector
from .llm_gate import LLMGate
from .metrics import ClassificationReport, evaluate, lead_time


@dataclass
class TopicResult:
    topic_id: str
    topic: str
    kind: str
    label: int
    detector_fired: bool
    detector_index: int | None
    detector_score: float
    gate_passed: bool | None
    gate_confidence: float | None
    gate_reason: str
    gate_source: str
    final_prediction: int
    final_score: float
    viral_index: int | None
    lead_time: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    config: dict
    report: ClassificationReport
    detector_only_report: ClassificationReport
    results: list[TopicResult]
    gate_stats: dict
    runtime_s: float
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "smartgate_version": __version__,
            "config": self.config,
            "runtime_s": round(self.runtime_s, 3),
            "environment": self.environment,
            "gate_stats": self.gate_stats,
            "metrics": {
                "detector_plus_gate": self.report.to_dict(),
                "detector_only": self.detector_only_report.to_dict(),
            },
            "results": [r.to_dict() for r in self.results],
        }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_pipeline(
    samples: Sequence[Sample] | None = None,
    config: PipelineConfig | None = None,
    gate: LLMGate | None = None,
    n_samples: int = 200,
    seed: int = 20240501,
) -> RunResult:
    config = config or PipelineConfig()
    samples = list(samples) if samples is not None else generate_dataset(n_samples, seed=seed)
    detector = build_detector(config.detector, config.detectors)
    if config.use_llm_gate and gate is None:
        gate = LLMGate(config=config.ollama)

    started = time.time()
    results: list[TopicResult] = []
    for sample in samples:
        detection = detector.run(sample.series, config.decision_horizon)
        gate_passed: bool | None = None
        confidence: float | None = None
        reason, source = "", "none"

        if detection.fired and config.use_llm_gate and gate is not None:
            # The gate must not see the future either: it only gets the evidence that
            # exists at the decision point.
            visible = sample.series[
                : min(detection.index + config.decision_horizon + 1, len(sample.series))
            ]
            verdict = gate.judge(replace(sample, series=visible), detection.index, detection.score)
            gate_passed = verdict.is_emerging
            confidence = verdict.confidence
            reason, source = verdict.reason, verdict.source

        final = int(detection.fired and (gate_passed is not False))
        # Ranking score: statistical strength, modulated by the gate's confidence so that
        # a confidently-rejected alarm sinks below an unreviewed one.
        score = detection.score
        if gate_passed is True:
            score *= 1.0 + confidence
        elif gate_passed is False:
            score *= max(0.05, 1.0 - (confidence or 0.0))
        lt = lead_time(detection.index if final else None, sample.viral_index)
        results.append(
            TopicResult(
                topic_id=sample.topic_id,
                topic=sample.topic,
                kind=sample.kind,
                label=sample.label,
                detector_fired=detection.fired,
                detector_index=detection.index,
                detector_score=round(detection.score, 4),
                gate_passed=gate_passed,
                gate_confidence=confidence,
                gate_reason=reason,
                gate_source=source,
                final_prediction=final,
                final_score=round(score, 4),
                viral_index=sample.viral_index,
                lead_time=lt,
            )
        )

    y_true = [r.label for r in results]
    report = evaluate(
        y_true,
        [r.final_prediction for r in results],
        [r.final_score for r in results],
        [r.lead_time for r in results],
        k=config.top_k,
    )
    detector_only = evaluate(
        y_true,
        [int(r.detector_fired) for r in results],
        [r.detector_score for r in results],
        [lead_time(r.detector_index if r.detector_fired else None, r.viral_index) for r in results],
        k=config.top_k,
    )
    cfg_dump = {
        "detector": config.detector,
        "use_llm_gate": config.use_llm_gate,
        "top_k": config.top_k,
        "decision_horizon": config.decision_horizon,
        "seed": seed,
        "n_samples": len(samples),
        "detector_params": asdict(config.detectors),
        "ollama": config.ollama.to_dict() if config.use_llm_gate else None,
    }
    return RunResult(
        config=cfg_dump,
        report=report,
        detector_only_report=detector_only,
        results=results,
        gate_stats=gate.stats() if gate is not None else {},
        runtime_s=time.time() - started,
        environment=_environment(),
    )


def save_run(result: RunResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def sweep_detectors(
    samples: Sequence[Sample],
    detector_names: Sequence[str] = ("threshold", "ewma", "cusum"),
    detector_config: DetectorConfig | None = None,
    top_k: int = 10,
    decision_horizon: int = 7,
) -> dict[str, ClassificationReport]:
    """Statistics-only comparison — no LLM, so it runs in milliseconds in CI."""
    out: dict[str, ClassificationReport] = {}
    for name in detector_names:
        cfg = PipelineConfig(
            detector=name,
            use_llm_gate=False,
            top_k=top_k,
            decision_horizon=decision_horizon,
            detectors=detector_config or DetectorConfig(),
            ollama=OllamaConfig(),
        )
        out[name] = run_pipeline(samples, cfg).detector_only_report
    return out
