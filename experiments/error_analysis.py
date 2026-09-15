"""Sprint-02 error analysis: what the false positives and false negatives actually are.

Sections 12 and 13 of the required report ask for *categories*, not counts. Counts come
from the metrics artifact; this script goes back to the stored per-topic verdicts and the
corpus itself, so every claim in the report is traceable to a topic id.

    PYTHONPATH=src python3 experiments/error_analysis.py --key qwen3:0.6b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartgate.dataset import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _shape(sample) -> dict:
    """Coarse description of the observation window, used to bucket the errors."""
    series = list(sample.series)
    warm = series[:14] or series
    base = statistics.median(warm) or 1.0
    tail = series[-7:]
    peak = max(series)
    return {
        "baseline": round(base, 1),
        "peak_ratio": round(peak / base, 2),
        "tail_ratio": round(statistics.fmean(tail) / base, 2),
        "peak_day": series.index(peak),
        "n_days_above_2x": sum(1 for v in series if v > 2 * base),
    }


def bucket(shape: dict, label: int) -> str:
    """One label per error, chosen by the observation-window shape only."""
    if shape["n_days_above_2x"] <= 1:
        return "one-day spike" if label == 0 else "quiet run-up (nothing visible yet)"
    if shape["tail_ratio"] < 1.5 <= shape["peak_ratio"]:
        return "spike that already decayed inside the window"
    if shape["peak_ratio"] < 2.0:
        return "low-amplitude wobble around the baseline"
    return "sustained elevation"


def analyse(comparison: dict, key: str, samples) -> dict:
    entry = comparison["models"][key]
    if entry["scope"] == "paired_subsample":
        # otherwise "positives the detector never flagged" would be counted against the
        # whole test period while the model only ever saw 40 topics
        scored = set(comparison["paired_subsample"]["topic_ids"])
        samples = [s for s in samples if s.topic_id in scored]
    by_id = {s.topic_id: s for s in samples}
    fp, fn = [], []
    for v in entry["verdicts"]:
        sample = by_id.get(v["topic_id"])
        if sample is None:
            continue
        shape = _shape(sample)
        row = {**{k: v[k] for k in ("topic_id", "topic", "label", "gate_passed",
                                    "gate_reason", "lead_time")}, **shape,
               "bucket": bucket(shape, v["label"])}
        if v["gate_passed"] and v["label"] == 0:
            fp.append(row)
        elif not v["gate_passed"] and v["label"] == 1:
            fn.append(row)
    # positives the detector never even flagged: a different failure, owned by the cheap
    # stage, and the report must not blame them on the gate
    flagged = {v["topic_id"] for v in entry["verdicts"]}
    missed_by_detector = [
        {"topic_id": s.topic_id, "topic": s.topic, **_shape(s)}
        for s in samples
        if s.label == 1 and s.topic_id not in flagged
    ]

    def _counts(rows):
        out: dict[str, int] = {}
        for r in rows:
            out[r["bucket"]] = out.get(r["bucket"], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    return {
        "key": key,
        "arm": entry["arm"],
        "scope": entry["scope"],
        "n_scored": entry["n_scored"],
        "false_positives": {"n": len(fp), "by_bucket": _counts(fp), "examples": fp[:10]},
        "false_negatives": {"n": len(fn), "by_bucket": _counts(fn), "examples": fn[:10]},
        "positives_never_flagged_by_detector": {
            "n": len(missed_by_detector),
            "examples": missed_by_detector[:10],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", default=str(ROOT / "evaluation" / "qwen_model_comparison.json"))
    p.add_argument("--dataset", default=str(ROOT / "artifacts" / "real_world_dataset.test.jsonl"))
    p.add_argument("--keys", nargs="*", default=None)
    p.add_argument("--out", default=str(ROOT / "evaluation" / "error_analysis.json"))
    args = p.parse_args()

    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    samples = load_jsonl(args.dataset)
    keys = args.keys or [
        k for k, v in comparison["models"].items() if v.get("status") == "measured"
    ]
    payload = {"dataset": args.dataset, "analyses": [analyse(comparison, k, samples) for k in keys]}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for a in payload["analyses"]:
        print(f"{a['key']} ({a['scope']}, n={a['n_scored']}): "
              f"FP={a['false_positives']['n']} {a['false_positives']['by_bucket']} | "
              f"FN={a['false_negatives']['n']} {a['false_negatives']['by_bucket']} | "
              f"detector-missed positives={a['positives_never_flagged_by_detector']['n']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
