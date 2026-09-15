"""Sprint-02 end-to-end experiment driver.

Run order is the whole point of this file, so it is enforced in code:

    development period  ->  choose  ->  FREEZE  ->  test period  ->  report

Nothing that is measured on the test period is allowed to change a parameter. The frozen
choices are written to ``evaluation/frozen_config.json`` *before* the test period is
scored, so the artifact itself is the evidence that the order was respected.

Usage::

    PYTHONPATH=src python3 experiments/sprint02.py --stage dev
    PYTHONPATH=src python3 experiments/sprint02.py --stage test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartgate.config import (  # noqa: E402
    SMOKE_MODEL,
    TARGET_MODEL,
    DetectorConfig,
    OllamaConfig,
    PipelineConfig,
)
from smartgate.dataset import load_jsonl  # noqa: E402
from smartgate.experiments import (  # noqa: E402
    choose_operating_point,
    gate_ablation,
    host_report,
    precision_at_k_curve,
    run_arm,
)
from smartgate.ollama_client import OllamaClient  # noqa: E402
from smartgate.pipeline import run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
ARTIFACTS = ROOT / "artifacts"

# ---------------------------------------------------------------------- product constant
# How pure the queue must be before an analyst keeps using it. This is a product decision,
# taken once, before any Sprint-02 metric existed; it is *not* re-tuned per period.
PRECISION_FLOOR = 0.50

# Development-period search space (backlog S2-05: hyper-parameters by experiment, not by
# taste). ``warmup`` is deliberately absent: it defines the baseline the *labels* are
# computed against, so moving it would move the target instead of the model.
DETECTOR_GRID = [
    ("cusum", DetectorConfig(cusum_k=k, cusum_h=h))
    for k in (0.25, 0.5, 1.0)
    for h in (3.0, 5.0, 8.0)
] + [
    ("ewma", DetectorConfig(ewma_alpha=a, ewma_k=k))
    for a in (0.2, 0.3, 0.5)
    for k in (2.0, 3.0, 4.0)
] + [("threshold", DetectorConfig())]
HORIZON_GRID = (3, 5, 7, 10, 14)


def _write(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def tune_detector(samples) -> dict:
    """Grid search on the development period, selected by PR-AUC (ranking quality).

    PR-AUC and not F1: the gate consumes a *ranked* candidate list, so how well the
    statistic orders topics matters more than where its own threshold happens to sit.
    """
    rows = []
    for name, detector_cfg in DETECTOR_GRID:
        for horizon in HORIZON_GRID:
            cfg = PipelineConfig(
                detector=name,
                use_llm_gate=False,
                top_k=10,
                decision_horizon=horizon,
                detectors=detector_cfg,
            )
            report = run_pipeline(samples, cfg).detector_only_report
            rows.append(
                {
                    "detector": name,
                    "params": asdict(detector_cfg),
                    "decision_horizon": horizon,
                    "metrics": report.to_dict(),
                }
            )
    best = max(rows, key=lambda r: (r["metrics"]["pr_auc"], -r["decision_horizon"]))
    return {"grid": rows, "best": best}


def stage_dev(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.dev_dataset)
    print(f"development corpus: {len(samples)} topics, {sum(s.label for s in samples)} positive")

    tuning = tune_detector(samples)
    best = tuning["best"]
    print(
        f"best detector: {best['detector']} {best['params']} H={best['decision_horizon']} "
        f"PR-AUC={best['metrics']['pr_auc']:.3f}"
    )

    config = PipelineConfig(
        detector=best["detector"],
        top_k=10,
        decision_horizon=best["decision_horizon"],
        detectors=DetectorConfig(**best["params"]),
        ollama=OllamaConfig(model=args.model),
    )

    ablation = gate_ablation(samples, config, bootstrap=args.bootstrap)
    ablation["period"] = "dev"
    ablation["dataset_path"] = args.dev_dataset
    ablation["detector_tuning"] = tuning
    ablation["environment"] = host_report()
    _write(ablation, EVAL / "gate_ablation.json")

    # Arm selection, decided on dev and then frozen. Two filters, in this order:
    #   1. only LLM arms are candidates — the point of the sprint is the semantic gate;
    #   2. an arm that lets *nothing* through is not a gate, it is an off switch. Such an
    #      arm still scores a respectable PR-AUC, because ranking then falls back to the
    #      detector statistic alone, so selecting on PR-AUC without this filter would
    #      freeze a configuration that never fires. Recall > 0 is the minimal sanity bar.
    llm_arms = {a: d for a, d in ablation["arms"].items() if a.startswith("llm_")}
    alive = {a: d for a, d in llm_arms.items() if d["metrics"]["recall"] > 0}
    scored = alive or llm_arms or ablation["arms"]
    if not alive:
        print("WARNING: every LLM arm rejected all alarms on the development period")
    best_arm = max(scored.items(), key=lambda kv: kv[1]["metrics"]["pr_auc"])[0]
    operating = choose_operating_point(
        ablation["arms"][best_arm]["precision_at_k_curve"], PRECISION_FLOOR
    )
    print(f"best gate arm: {best_arm}; operating point K={operating['k']} ({operating['reason']})")

    frozen = {
        "frozen_on": "development period only",
        "precision_floor": PRECISION_FLOOR,
        "detector": best["detector"],
        "detector_params": best["params"],
        "decision_horizon": best["decision_horizon"],
        "gate_arm": best_arm,
        "arm_selection": {
            "rule": "highest PR-AUC among LLM arms that pass at least one alarm",
            "candidates": {a: d["metrics"]["pr_auc"] for a, d in scored.items()},
            "degenerate_arms": [a for a in llm_arms if a not in alive],
        },
        "top_k": operating["k"],
        "operating_point": operating,
        "gate_model_used_for_selection": args.model,
        "dev_dataset": args.dev_dataset,
    }
    _write(frozen, EVAL / "frozen_config.json")
    return 0


def _load_frozen() -> dict:
    path = EVAL / "frozen_config.json"
    if not path.exists():
        raise SystemExit("run `--stage dev` first: the test period may only use frozen values")
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline_config(frozen: dict, model: str) -> PipelineConfig:
    return PipelineConfig(
        detector=frozen["detector"],
        top_k=frozen["top_k"],
        decision_horizon=frozen["decision_horizon"],
        detectors=DetectorConfig(**frozen["detector_params"]),
        ollama=OllamaConfig(model=model),
    )


def _model_status(model: str) -> dict | None:
    """``None`` when the model can be measured; otherwise the reason it cannot.

    The brief forbids substituting one model's numbers for another's, so an unavailable
    model produces an explicit unavailability record and never a borrowed metric.
    """
    client = OllamaClient(OllamaConfig(model=model))
    if not client.is_available():
        return {"status": "unavailable", "reason": "no Ollama server reachable"}
    if not client.has_model(model):
        return {"status": "unavailable", "reason": f"not pulled; run `ollama pull {model}`"}
    return None


def _paired_subsample(samples, size: int, seed: int):
    """A seeded, model-independent subset of the test period.

    Qwen3-14B answers at roughly one verdict per several minutes on a CPU-only host, so
    scoring it on every alarm is not affordable inside this sprint. The mitigation is a
    *paired* comparison: the subset is drawn once, before any model runs, and every model
    is then scored on exactly the same topics. That keeps the 0.6B-vs-14B contrast honest
    (same topics, same labels) at the cost of a wider confidence interval, and the full
    test period is still reported for the cheap model. The subset is never chosen with any
    model's output in view.
    """
    if size <= 0 or size >= len(samples):
        return list(samples), None
    index = sorted(random.Random(seed).sample(range(len(samples)), size))
    subset = [samples[i] for i in index]
    return subset, {
        "size": size,
        "seed": seed,
        "of": len(samples),
        "n_positive": sum(s.label for s in subset),
        "topic_ids": [s.topic_id for s in subset],
        "reason": "Qwen3-14B throughput on this host; drawn before any model was run",
    }


def stage_test(args: argparse.Namespace) -> int:
    frozen = _load_frozen()
    samples = load_jsonl(args.test_dataset)
    print(f"test corpus: {len(samples)} topics, {sum(s.label for s in samples)} positive")
    shown = ("detector", "detector_params", "decision_horizon", "gate_arm", "top_k")
    print("frozen config: " + json.dumps({k: frozen[k] for k in shown}))

    comparison: dict = {
        "period": "test",
        "dataset_path": args.test_dataset,
        "frozen_config": frozen,
        "arm": frozen["gate_arm"],
        "environment": host_report(),
        "models": {},
    }

    baseline_cfg = _pipeline_config(frozen, args.models[0])
    stats_only = run_arm("stats_only", samples, baseline_cfg)
    comparison["stats_only"] = {
        "metrics": stats_only.detector_only_report.to_dict(),
        "precision_at_k_curve": precision_at_k_curve(stats_only),
    }

    subset, subsample_meta = _paired_subsample(samples, args.subsample, args.subsample_seed)
    comparison["paired_subsample"] = subsample_meta
    heavy = set(args.subsample_models)
    # The frozen arm is the headline configuration. Extra arms may be scored *in addition*
    # — the 0.6B-vs-14B question is about semantic reasoning, and the frozen arm turned out
    # to be the numbers-only one, which would have answered a different question. Extra
    # arms are reported separately and never replace the frozen one.
    arms = [frozen["gate_arm"]] + [a for a in args.extra_arms if a != frozen["gate_arm"]]
    comparison["arms_scored"] = arms
    runs: list[tuple[str, str, str, list]] = []
    for arm in arms:
        for model in args.models:
            suffix = "" if arm == frozen["gate_arm"] else f"|{arm}"
            if subsample_meta is None:
                runs.append((f"{model}{suffix}", model, arm, samples))
                continue
            # cheap models are scored twice: once on the full period (the headline number)
            # and once on the subset (the only fair basis for comparing against 14B)
            if model not in heavy:
                runs.append((f"{model}{suffix}", model, arm, samples))
            runs.append((f"{model}{suffix}@subsample", model, arm, subset))

    for key, model, arm, scored_samples in runs:
        unavailable = _model_status(model)
        if unavailable:
            comparison["models"][key] = unavailable
            print(f"{key}: {unavailable['reason']}")
            continue
        config = _pipeline_config(frozen, model)
        print(f"scoring {key} ({arm}, {len(scored_samples)} topics) ...")
        result = run_arm(arm, scored_samples, config)
        details = OllamaClient(config.ollama).show(model).get("details", {})
        comparison["models"][key] = {
            "status": "measured",
            "model": model,
            "arm": arm,
            "n_scored": len(scored_samples),
            "scope": "paired_subsample" if key.endswith("@subsample") else "full_test_period",
            "model_details": details,
            "inference_mode": config.ollama.to_dict(),
            "metrics": result.report.to_dict(),
            "precision_at_k_curve": precision_at_k_curve(result),
            "gate_stats": result.gate_stats,
            "runtime_s": round(result.runtime_s, 2),
            "verdicts": [
                {
                    "topic_id": r.topic_id,
                    "topic": r.topic,
                    "source": r.kind,
                    "label": r.label,
                    "gate_passed": r.gate_passed,
                    "gate_confidence": r.gate_confidence,
                    "gate_reason": r.gate_reason,
                    "detector_index": r.detector_index,
                    "lead_time": r.lead_time,
                }
                for r in result.results
                if r.detector_fired
            ],
        }
        m = result.report.to_dict()
        print(
            f"{key}: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"P@{m['k']}={m['precision_at_k']:.3f} PR-AUC={m['pr_auc']:.3f}"
        )
    _write(comparison, EVAL / "qwen_model_comparison.json")

    final = {
        "sprint": "SPRINT-02",
        "frozen_config": frozen,
        "test_period": {
            "dataset_path": args.test_dataset,
            "n_samples": len(samples),
            "n_positive": sum(s.label for s in samples),
        },
        "paired_subsample": comparison.get("paired_subsample"),
        "stats_only": comparison["stats_only"]["metrics"],
        "models": {
            key: (
                {
                    "model": data["model"],
                    "arm": data["arm"],
                    "scope": data["scope"],
                    "n_scored": data["n_scored"],
                    "metrics": data["metrics"],
                    "gate_stats": data["gate_stats"],
                    "precision_at_k_curve": data["precision_at_k_curve"],
                }
                if data.get("status") == "measured"
                else data
            )
            for key, data in comparison["models"].items()
        },
        "environment": host_report(),
    }
    _write(final, EVAL / "final_metrics.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "test"], required=True)
    parser.add_argument("--dev-dataset", default=str(ARTIFACTS / "real_world_dataset.dev.jsonl"))
    parser.add_argument("--test-dataset", default=str(ARTIFACTS / "real_world_dataset.test.jsonl"))
    parser.add_argument("--model", default=SMOKE_MODEL, help="gate model used for dev selection")
    parser.add_argument("--models", nargs="+", default=[SMOKE_MODEL, TARGET_MODEL])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="score expensive models on a seeded subset of the test period (0 = full period)",
    )
    parser.add_argument("--subsample-seed", type=int, default=20250215)
    parser.add_argument("--subsample-models", nargs="*", default=[TARGET_MODEL])
    parser.add_argument(
        "--extra-arms",
        nargs="*",
        default=[],
        help="additional gate arms to score alongside the frozen one (reported separately)",
    )
    args = parser.parse_args(argv)
    return stage_dev(args) if args.stage == "dev" else stage_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
