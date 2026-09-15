"""How much precision do we buy by waiting longer after the alarm?

This is the core trade-off of the product: every extra day of confirmation costs
lead time. The experiment sweeps the decision horizon H and records PR-AUC.

    python experiments/horizon_sensitivity.py --out artifacts/horizon_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartgate.dataset import generate_dataset, load_jsonl
from smartgate.pipeline import sweep_detectors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20240501)
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 7, 10, 14, 21, 30])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="artifacts/horizon_sensitivity.json")
    args = ap.parse_args()

    samples = load_jsonl(args.dataset) if args.dataset else generate_dataset(args.n, seed=args.seed)
    table = {}
    print(f"{'H':>4}{'threshold':>12}{'ewma':>10}{'cusum':>10}   (PR-AUC)")
    for h in args.horizons:
        reports = sweep_detectors(samples, top_k=args.top_k, decision_horizon=h)
        table[h] = {name: rep.to_dict() for name, rep in reports.items()}
        row = "".join(f"{reports[n].pr_auc:>10.3f}  " for n in ("threshold", "ewma", "cusum"))
        print(f"{h:>4}{row}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
