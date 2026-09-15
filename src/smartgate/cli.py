"""Command line entry point: ``python -m smartgate <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import SMOKE_MODEL, TARGET_MODEL, DetectorConfig, OllamaConfig, PipelineConfig
from .dataset import generate_dataset, load_jsonl, save_jsonl
from .experiments import (
    ABLATION_ARMS,
    gate_ablation,
    host_report,
    make_gate,
    run_arm,
)
from .llm_gate import GateFeatures
from .llm_gate import LLMGate
from .ollama_client import OllamaClient, OllamaError
from .pipeline import run_pipeline, save_run, sweep_detectors
from .realworld import PERIODS, build_period_dataset, entity_key, save_meta
from .wikipedia import PageviewsClient


def _detector_config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        ewma_alpha=args.ewma_alpha,
        ewma_k=args.ewma_k,
        cusum_k=args.cusum_k,
        cusum_h=args.cusum_h,
        warmup=args.warmup,
    )


def _ollama_config(args: argparse.Namespace) -> OllamaConfig:
    kwargs = {}
    if getattr(args, "model", None):
        kwargs["model"] = args.model
    if getattr(args, "host", None):
        kwargs["host"] = args.host
    return OllamaConfig(**kwargs)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Verify the runtime layer before anything else runs (Day-1 engineer checklist)."""
    client = OllamaClient(_ollama_config(args))
    print(f"endpoint: {client.config.api_base}")
    try:
        print(f"server version: {client.version()}")
    except OllamaError as exc:
        print(f"UNAVAILABLE: {exc}")
        print("hint: install from https://docs.ollama.com/linux and run `ollama serve`")
        return 1
    models = client.list_models()
    print(f"models: {', '.join(models) or '(none)'}")
    wanted = client.config.model
    if not client.has_model(wanted):
        print(f"MISSING MODEL: {wanted} — run `ollama pull {wanted}`")
        return 2
    info = client.show(wanted)
    details = info.get("details", {})
    print(f"model: {wanted}")
    print(f"  family: {details.get('family')}  params: {details.get('parameter_size')}")
    print(f"  quantization: {details.get('quantization_level')}")
    print(f"  format: {details.get('format')}")
    print(f"  frozen inference mode: think={client.config.think} options={client.config.options()}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    samples = generate_dataset(
        n_samples=args.n, length=args.length, positive_ratio=args.positive_ratio, seed=args.seed
    )
    path = save_jsonl(samples, args.out)
    positives = sum(s.label for s in samples)
    print(f"wrote {len(samples)} samples ({positives} positive) to {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.dataset) if args.dataset else generate_dataset(args.n, seed=args.seed)
    config = PipelineConfig(
        detector=args.detector,
        use_llm_gate=not args.no_gate,
        top_k=args.top_k,
        decision_horizon=args.decision_horizon,
        ollama=_ollama_config(args),
        detectors=_detector_config(args),
    )
    gate = None
    if config.use_llm_gate:
        gate = LLMGate(config=config.ollama, allow_fallback=not args.strict_llm)
    result = run_pipeline(samples, config, gate=gate, seed=args.seed, dataset_path=args.dataset)
    if args.out:
        print(f"run saved to {save_run(result, args.out)}")
    combined = result.report.to_dict()
    baseline = result.detector_only_report.to_dict()
    print(json.dumps({"detector_only": baseline, "detector_plus_gate": combined}, indent=2))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.dataset) if args.dataset else generate_dataset(args.n, seed=args.seed)
    reports = sweep_detectors(
        samples,
        detector_config=_detector_config(args),
        top_k=args.top_k,
        decision_horizon=args.decision_horizon,
    )
    table = {name: rep.to_dict() for name, rep in reports.items()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"sweep saved to {args.out}")
    print(f"{'detector':<12}{'P':>8}{'R':>8}{'F1':>8}{'FPR':>8}{'PR-AUC':>9}{'lead':>8}")
    for name, rep in table.items():
        print(
            f"{name:<12}{rep['precision']:>8.3f}{rep['recall']:>8.3f}{rep['f1']:>8.3f}"
            f"{rep['false_positive_rate']:>8.3f}{rep['pr_auc']:>9.3f}{rep['mean_lead_time']:>8.2f}"
        )
    return 0



def cmd_realworld(args: argparse.Namespace) -> int:
    """Build the causally split real-world corpus from Wikimedia pageviews (S2-01)."""
    client = PageviewsClient()
    out_dir = Path(args.out_dir)
    seen: list[tuple[str, str]] = []
    summary = {}
    # Periods are built in calendar order so the later one can exclude entities already
    # used by the earlier one: an entity in both splits would leak its behaviour across
    # the train/test boundary.
    for name in args.periods:
        period = PERIODS[name]
        samples, meta = build_period_dataset(
            client,
            period,
            max_candidates=args.max_candidates,
            seed=args.seed,
            exclude=seen,
        )
        seen.extend(entity_key(s) for s in samples)
        path = save_jsonl(samples, out_dir / f"real_world_dataset.{name}.jsonl")
        save_meta(meta, out_dir / f"real_world_dataset.{name}.meta.json")
        summary[name] = {
            "path": str(path),
            "n": len(samples),
            "positive": sum(s.label for s in samples),
            "dropped": meta.dropped,
        }
        print(
            f"{name}: {len(samples)} samples, {sum(s.label for s in samples)} positive "
            f"-> {path}"
        )
    print(json.dumps({"api": client.stats(), "periods": summary}, indent=2))
    return 0



def _write_json(payload: dict, path: str | None, label: str) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{label} saved to {path}")


def cmd_ablation(args: argparse.Namespace) -> int:
    """Feature ablation of the semantic gate on one corpus."""
    samples = load_jsonl(args.dataset)
    config = PipelineConfig(
        detector=args.detector,
        top_k=args.top_k,
        decision_horizon=args.decision_horizon,
        ollama=_ollama_config(args),
        detectors=_detector_config(args),
    )
    payload = gate_ablation(samples, config, arms=args.arms, bootstrap=args.bootstrap)
    payload["dataset_path"] = args.dataset
    payload["config"] = {
        "detector": args.detector,
        "decision_horizon": args.decision_horizon,
        "top_k": args.top_k,
        "model": config.ollama.model,
    }
    payload["environment"] = host_report()
    _write_json(payload, args.out, "ablation")
    print(f"{'arm':<20}{'P':>8}{'R':>8}{'F1':>8}{'PR-AUC':>9}{'P@K':>8}{'lead':>8}")
    for arm, data in payload["arms"].items():
        m = data["metrics"]
        print(
            f"{arm:<20}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
            f"{m['pr_auc']:>9.3f}{m['precision_at_k']:>8.3f}{m['mean_lead_time']:>8.2f}"
        )
    return 0


def cmd_compare_models(args: argparse.Namespace) -> int:
    """Same corpus, same prompt, same decoding — only the weights differ (PHASE 8)."""
    samples = load_jsonl(args.dataset)
    payload: dict = {"dataset_path": args.dataset, "models": {}, "environment": host_report()}
    for model in args.models:
        config = PipelineConfig(
            detector=args.detector,
            top_k=args.top_k,
            decision_horizon=args.decision_horizon,
            ollama=OllamaConfig(model=model),
            detectors=_detector_config(args),
        )
        client = OllamaClient(config.ollama)
        if not client.is_available():
            payload["models"][model] = {"status": "unavailable", "reason": "no Ollama server"}
            print(f"{model}: SKIPPED (no Ollama server)")
            continue
        if not client.has_model(model):
            payload["models"][model] = {
                "status": "unavailable",
                "reason": f"model not pulled; run `ollama pull {model}`",
            }
            print(f"{model}: SKIPPED (not pulled)")
            continue
        gate = make_gate(config, GateFeatures(), strict=args.strict_llm)
        result = run_pipeline(samples, config, gate=gate, dataset_path=args.dataset)
        payload["models"][model] = {
            "status": "measured",
            "details": client.show(model).get("details", {}),
            "metrics": result.report.to_dict(),
            "gate_stats": result.gate_stats,
            "runtime_s": round(result.runtime_s, 2),
            "verdicts": [
                {
                    "topic_id": r.topic_id,
                    "label": r.label,
                    "gate_passed": r.gate_passed,
                    "gate_confidence": r.gate_confidence,
                    "gate_reason": r.gate_reason,
                }
                for r in result.results
                if r.detector_fired
            ],
        }
        m = result.report.to_dict()
        print(
            f"{model}: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"P@{m['k']}={m['precision_at_k']:.3f} lead={m['mean_lead_time']:.2f}"
        )
    _write_json(payload, args.out, "model comparison")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartgate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=20240501)
        p.add_argument("--top-k", type=int, default=10)
        p.add_argument("--ewma-alpha", type=float, default=0.3)
        p.add_argument("--ewma-k", type=float, default=3.0)
        p.add_argument("--cusum-k", type=float, default=0.5)
        p.add_argument("--cusum-h", type=float, default=5.0)
        p.add_argument("--warmup", type=int, default=14)
        p.add_argument("--decision-horizon", type=int, default=7)

    doctor = sub.add_parser("doctor", help="check the Ollama runtime + model")
    doctor.add_argument("--model")
    doctor.add_argument("--host")
    doctor.set_defaults(func=cmd_doctor)

    ds = sub.add_parser("dataset", help="generate the labelled benchmark corpus")
    ds.add_argument("--n", type=int, default=200)
    ds.add_argument("--length", type=int, default=90)
    ds.add_argument("--positive-ratio", type=float, default=0.3)
    ds.add_argument("--seed", type=int, default=20240501)
    ds.add_argument("--out", default="artifacts/dataset.jsonl")
    ds.set_defaults(func=cmd_dataset)

    run = sub.add_parser("run", help="run detector (+ Qwen3 gate) and evaluate")
    run.add_argument("--dataset")
    run.add_argument("--n", type=int, default=200)
    run.add_argument("--detector", default="ewma", choices=["ewma", "cusum", "threshold"])
    run.add_argument("--no-gate", action="store_true", help="statistics only")
    run.add_argument("--strict-llm", action="store_true", help="fail instead of falling back")
    run.add_argument("--model")
    run.add_argument("--host")
    run.add_argument("--out")
    add_common(run)
    run.set_defaults(func=cmd_run)

    rw = sub.add_parser("realworld", help="build the real-world corpus (Wikimedia pageviews)")
    rw.add_argument("--periods", nargs="+", default=["dev", "test"], choices=sorted(PERIODS))
    rw.add_argument("--max-candidates", type=int, default=400)
    rw.add_argument("--seed", type=int, default=20250201)
    rw.add_argument("--out-dir", default="artifacts")
    rw.set_defaults(func=cmd_realworld)

    abl = sub.add_parser("ablation", help="feature ablation of the semantic gate")
    abl.add_argument("--dataset", required=True)
    abl.add_argument("--detector", default="cusum", choices=["ewma", "cusum", "threshold"])
    abl.add_argument("--arms", nargs="+", default=[name for name, _ in ABLATION_ARMS])
    abl.add_argument("--bootstrap", type=int, default=1000)
    abl.add_argument("--model")
    abl.add_argument("--host")
    abl.add_argument("--out")
    add_common(abl)
    abl.set_defaults(func=cmd_ablation)

    cmp_ = sub.add_parser("compare-models", help="Qwen3 0.6B vs 14B on one corpus")
    cmp_.add_argument("--dataset", required=True)
    cmp_.add_argument("--models", nargs="+", default=[SMOKE_MODEL, TARGET_MODEL])
    cmp_.add_argument("--detector", default="cusum", choices=["ewma", "cusum", "threshold"])
    cmp_.add_argument("--strict-llm", action="store_true")
    cmp_.add_argument("--out")
    add_common(cmp_)
    cmp_.set_defaults(func=cmd_compare_models)

    sweep = sub.add_parser("sweep", help="compare statistical detectors (no LLM)")
    sweep.add_argument("--dataset")
    sweep.add_argument("--n", type=int, default=200)
    sweep.add_argument("--out")
    add_common(sweep)
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
