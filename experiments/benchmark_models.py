"""PHASE 9 — performance benchmark of the gate models.

Measures, per model, on identical prompts: latency (mean/median/p95/max), generation
throughput in tokens/sec, process memory, failure rate and JSON validity. Hardware is
reported as it is: if there is no GPU on the host, the artifact says so literally instead
of quoting a number that was never measured on one.

Caching is disabled here on purpose — a cached verdict would report the latency of a
filesystem read.

Usage::

    PYTHONPATH=src python3 experiments/benchmark_models.py --n 12 --out evaluation/performance.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartgate.config import (  # noqa: E402
    SMOKE_MODEL,
    TARGET_MODEL,
    DetectorConfig,
    OllamaConfig,
)
from smartgate.dataset import load_jsonl  # noqa: E402
from smartgate.detectors import build_detector  # noqa: E402
from smartgate.experiments import host_report, latency_summary  # noqa: E402
from smartgate.llm_gate import LLMGate, build_prompt  # noqa: E402
from smartgate.ollama_client import OllamaClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _workload(dataset: str, n: int, frozen: dict) -> list[tuple]:
    """The first ``n`` alarms of the test period, in dataset order (no cherry-picking)."""
    samples = load_jsonl(dataset)
    detector = build_detector(frozen["detector"], DetectorConfig(**frozen["detector_params"]))
    work = []
    for sample in samples:
        detection = detector.run(sample.series, frozen["decision_horizon"])
        if detection.fired:
            work.append((sample, detection.index, detection.score))
        if len(work) >= n:
            break
    return work


def benchmark(model: str, work: list[tuple], num_predict: int | None = None) -> dict:
    config = OllamaConfig(model=model)
    if num_predict is not None:
        object.__setattr__(config, "num_predict", num_predict)
    client = OllamaClient(config)
    if not client.is_available():
        return {"status": "unavailable", "reason": "no Ollama server reachable"}
    if not client.has_model(model):
        return {"status": "unavailable", "reason": f"not pulled; run `ollama pull {model}`"}

    gate = LLMGate(client=client, allow_fallback=False, cache_dir=None)
    prompt_chars, started = [], time.time()
    for sample, index, score in work:
        prompt_chars.append(len(build_prompt(sample, index, score, gate.features)))
        try:
            gate.judge(sample, index, score)
        except Exception as exc:  # noqa: BLE001 - the failure rate is the measurement
            print(f"  {model}: {type(exc).__name__}: {exc}")
    wall = time.time() - started
    stats = gate.stats()
    return {
        "status": "measured",
        "model": model,
        "n_prompts": len(work),
        "inference_mode": config.to_dict(),
        "model_details": client.show(model).get("details", {}),
        "latency": latency_summary(gate.latencies),
        "server_timings_mean_s": {
            key: round(sum(t[key] for t in gate.timings) / len(gate.timings), 3)
            for key in ("load_duration", "prompt_eval_duration", "eval_duration",
                        "total_duration")
        }
        if gate.timings
        else {},
        "tokens_per_second": {
            "mean": stats["mean_tokens_per_second"],
            "samples": len(gate.token_rates),
        },
        "prompt_size_chars": {
            "mean": round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else 0,
            "max": max(prompt_chars) if prompt_chars else 0,
        },
        "failure_rate": stats["failure_rate"],
        "json_validity": stats["json_validity"],
        "transport_errors": stats["transport_errors"],
        # A retried call still returns a verdict, so it never shows up as a failure — but
        # its latency is the sum of the timed-out attempt and the successful one. Reported
        # explicitly so nobody reads such a number as single-call latency.
        "silent_retries": client.retry_count,
        "socket_timeout_s": config.timeout,
        "invalid_json": stats["invalid_json"],
        "wall_clock_s": round(wall, 2),
        "prompts_per_minute": round(60 * len(work) / wall, 2) if wall else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "artifacts/real_world_dataset.test.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "evaluation/frozen_config.json"))
    parser.add_argument("--models", nargs="+", default=[SMOKE_MODEL, TARGET_MODEL])
    parser.add_argument("--n", type=int, default=12, help="prompts per model")
    parser.add_argument("--out", default=str(ROOT / "evaluation/performance.json"))
    args = parser.parse_args(argv)

    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    work = _workload(args.dataset, args.n, frozen)
    print(f"workload: {len(work)} alarms from {Path(args.dataset).name}")

    payload = {
        "phase": "PHASE 9 - performance",
        "environment": host_report(),
        "workload": {
            "dataset": args.dataset,
            "n_prompts": len(work),
            "selection": "first N alarms in dataset order",
            "cache": "disabled (a cached verdict would measure the filesystem)",
        },
        "models": {},
    }
    for model in args.models:
        print(f"benchmarking {model} ...")
        result = benchmark(model, work)
        payload["models"][model] = result
        if result["status"] == "measured":
            print(
                f"  latency mean {result['latency']['mean_s']}s "
                f"p95 {result['latency']['p95_s']}s, "
                f"{result['tokens_per_second']['mean']} tok/s, "
                f"JSON validity {result['json_validity']}"
            )
        else:
            print(f"  {result['reason']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
