"""Diagnose the positive-class rate of candidate label rules on the DEV period only.

Why this exists: the first frozen rule ("the whole 30-day label window averages >= 3x the
pre-alarm baseline AND 60% of its days sit above 2x") produced 1 positive out of 335 dev
samples. A 0.3% prevalence is not a hard problem, it is an unmeasurable one: no ranking
metric has usable resolution, and the bootstrap over topics would be degenerate.

This script is a *dataset design* diagnostic, run before any detector or gate has been
scored on this corpus, and it reads the DEV period only. It reports prevalence, never a
model metric, so it cannot be used to tune anything towards a nicer-looking result.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartgate import realworld as rw  # noqa: E402
from smartgate.wikipedia import PageviewsClient  # noqa: E402


def peak_window_mean(series, width):
    if len(series) < width:
        return 0.0
    return max(sum(series[i : i + width]) / width for i in range(len(series) - width + 1))


def peak_window_median(series, width):
    import statistics as st
    if len(series) < width:
        return 0.0
    return max(st.median(series[i : i + width]) for i in range(len(series) - width + 1))


def longest_run(series, level):
    best = run = 0
    for v in series:
        run = run + 1 if v >= level else 0
        best = max(best, run)
    return best


RULES = {
    "frozen_v1 (mean>=3x & 60% days>=2x)": lambda o, lab, b: (
        sum(lab) / len(lab) >= 3 * b and sum(1 for v in lab if v >= 2 * b) / len(lab) >= 0.6
    ),
    "peak7 >= 3x": lambda o, lab, b: peak_window_mean(lab, 7) >= 3 * b,
    "peak7 >= 2x": lambda o, lab, b: peak_window_mean(lab, 7) >= 2 * b,
    "peak7 >= 1.5x": lambda o, lab, b: peak_window_mean(lab, 7) >= 1.5 * b,
    "median7 >= 3x": lambda o, lab, b: peak_window_median(lab, 7) >= 3 * b,
    "median7 >= 2.5x": lambda o, lab, b: peak_window_median(lab, 7) >= 2.5 * b,
    "median7 >= 2x": lambda o, lab, b: peak_window_median(lab, 7) >= 2 * b,
    "peak7>=3x AND median7>=2x": lambda o, lab, b: (
        peak_window_mean(lab, 7) >= 3 * b and peak_window_median(lab, 7) >= 2 * b
    ),
    "run7 >= 2x": lambda o, lab, b: longest_run(lab, 2 * b) >= 7,
    "run7 >= 1.5x": lambda o, lab, b: longest_run(lab, 1.5 * b) >= 7,
    "label mean >= 1.5x": lambda o, lab, b: sum(lab) / len(lab) >= 1.5 * b,
}


def main() -> int:
    client = PageviewsClient()
    period = rw.DEV_PERIOD
    universe = rw.build_universe(client, period)
    import random

    candidates = sorted(random.Random(20250201).sample(universe, min(400, len(universe))))
    total = period.length_days + period.label_days
    counts = dict.fromkeys(RULES, 0)
    ratios = []
    n = 0
    for project, article in candidates:
        full = client.daily_series(project, article, period.observation_start, period.label_end)
        if len(full) != total:
            continue
        obs, lab = full[: period.length_days], full[period.length_days :]
        base = rw._baseline(obs)
        if base < rw.MIN_BASELINE_VIEWS:
            continue
        n += 1
        ratios.append(peak_window_mean(lab, 7) / base)
        for name, rule in RULES.items():
            counts[name] += int(rule(obs, lab, base))
    print(f"eligible samples: {n}")
    print(f"peak7/baseline: median {statistics.median(ratios):.2f} "
          f"p90 {sorted(ratios)[int(0.9 * len(ratios))]:.2f} max {max(ratios):.2f}")
    for name, c in counts.items():
        print(f"{c:4d}  {c / n:6.2%}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
