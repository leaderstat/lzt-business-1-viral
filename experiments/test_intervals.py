"""Bootstrap intervals for the frozen test-period configuration.

The dev-period ablation already reports intervals per arm; the headline test number was
missing them, and a point estimate on 23 positives invites over-reading. Every verdict is
cached (decoding is deterministic), so this re-scores from disk instead of calling Ollama.

    PYTHONPATH=src python3 experiments/test_intervals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartgate.config import DetectorConfig, OllamaConfig, PipelineConfig  # noqa: E402
from smartgate.dataset import load_jsonl  # noqa: E402
from smartgate.experiments import bootstrap_intervals, run_arm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"


def main() -> int:
    final = json.loads((EVAL / "final_metrics.json").read_text(encoding="utf-8"))
    frozen = final["frozen_config"]
    samples = load_jsonl(str(ROOT / "artifacts" / "real_world_dataset.test.jsonl"))
    config = PipelineConfig(
        detector=frozen["detector"],
        detectors=DetectorConfig(**frozen["detector_params"]),
        decision_horizon=frozen["decision_horizon"],
        top_k=frozen["top_k"],
        use_llm_gate=True,
        ollama=OllamaConfig(model=frozen["gate_model_used_for_selection"]),
    )
    out = {}
    for arm in ("stats_only", frozen["gate_arm"]):
        result = run_arm(arm, samples, config)
        out[arm] = bootstrap_intervals(result, k=frozen["top_k"])
        print(arm, json.dumps(out[arm]))
    final["bootstrap_ci_test"] = {
        "n_resamples": 1000,
        "unit": "topic",
        "seed": 7,
        "arms": out,
    }
    (EVAL / "final_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("updated evaluation/final_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
