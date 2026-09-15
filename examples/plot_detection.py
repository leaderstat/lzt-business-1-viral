"""Render a detection example as a standalone SVG (no plotting dependencies).

    python examples/plot_detection.py --out docs/screenshots

Produces one chart per class showing the series, the change point, the day the
detector fired and the day the topic became obviously viral — i.e. the lead time
we are selling, drawn to scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smartgate.dataset import generate_dataset
from smartgate.detectors import build_detector

W, H, PAD = 720, 260, 44
INK, GRID, LINE = "#1f2933", "#e4e7eb", "#2563eb"
ALARM, CHANGE, VIRAL = "#d97706", "#7c3aed", "#dc2626"


def _svg(sample, detection, horizon: int) -> str:
    series = sample.series
    top = max(series) or 1.0
    sx = lambda i: PAD + i * (W - 2 * PAD) / max(1, len(series) - 1)  # noqa: E731
    sy = lambda v: H - PAD - v * (H - 2 * PAD) / top  # noqa: E731
    points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" stroke="{GRID}"/>',
        f'<line x1="{PAD}" y1="{PAD - 10}" x2="{PAD}" y2="{H - PAD}" stroke="{GRID}"/>',
        f'<polyline fill="none" stroke="{LINE}" stroke-width="2" points="{points}"/>',
    ]

    def marker(idx, color, label, dy):
        if idx is None or idx >= len(series):
            return
        x = sx(idx)
        parts.append(
            f'<line x1="{x:.1f}" y1="{PAD - 10}" x2="{x:.1f}" y2="{H - PAD}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{min(x + 4, W - PAD - 120):.1f}" y="{PAD + dy}" font-size="11" '
            f'fill="{color}">{label} (d{idx})</text>'
        )

    marker(sample.change_index, CHANGE, "change point", 2)
    marker(detection.index, ALARM, "alarm", 16)
    marker(sample.viral_index, VIRAL, "obviously viral", 30)
    lead = (
        f"lead time = {sample.viral_index - detection.index} d"
        if sample.viral_index is not None and detection.index is not None
        else "no lead time (not a trend)"
    )
    label = "EMERGING TREND" if sample.label else "NOISE"
    parts.append(
        f'<text x="{PAD}" y="20" font-size="13" font-weight="600" fill="{INK}">'
        f"{sample.topic} — {label} / {sample.kind}</text>"
    )
    parts.append(
        f'<text x="{PAD}" y="{H - 14}" font-size="11" fill="{INK}">'
        f"decision horizon H={horizon} d · {lead} · daily mentions</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--detector", default="cusum")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20240501)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = build_detector(args.detector)
    samples = generate_dataset(200, seed=args.seed)
    wanted = ("exponential_growth", "growth_then_decay", "one_off_spike")
    for kind in wanted:
        sample = next(
            s for s in samples if s.kind == kind and detector.run(s.series, args.horizon).fired
        )
        detection = detector.run(sample.series, args.horizon)
        path = out / f"{kind}.svg"
        path.write_text(_svg(sample, detection, args.horizon), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
