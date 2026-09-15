"""Cheap statistical change-point baselines.

References read for this module (RULE 1 — official documentation first):

* NIST/SEMATECH e-Handbook, 6.3.2.3 CUSUM control charts
  https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
* NIST/SEMATECH e-Handbook, 6.3.2.4 EWMA control charts
  https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm

Both are implemented online (one pass, O(1) memory per series) so they can run on a
firehose of topics; ``ruptures`` style offline search is intentionally left for Sprint 02
because it needs the whole series up-front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import DetectorConfig


@dataclass
class Detection:
    """Result of running a detector over one series."""

    fired: bool
    index: int | None  # first index where the alarm fired
    score: float  # max normalised statistic, used as a ranking / ROC score
    trace: list[float] = field(default_factory=list)


class BaseDetector:
    name = "base"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def score_series(self, series: Sequence[float]) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def threshold(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def run(self, series: Sequence[float], decision_horizon: int | None = None) -> Detection:
        """Run online and score the alarm at the *decision point*.

        ``decision_horizon`` is the number of extra steps we are allowed to wait after the
        first alarm before committing to a verdict. Without it the ranking score would be
        the maximum over the whole series, i.e. a look-ahead the production system does not
        have — that inflates PR-AUC and hides the whole problem (see Report.md, DECISION-03).
        """
        trace = self.score_series(series)
        thr = self.threshold()
        index = next((i for i, v in enumerate(trace) if v >= thr), None)
        if index is None:
            score = max(trace) if trace else 0.0
        elif decision_horizon is None:
            score = max(trace)
        else:
            decision_at = min(index + decision_horizon, len(trace) - 1)
            score = max(trace[: decision_at + 1])
        return Detection(fired=index is not None, index=index, score=score, trace=trace)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float], mean: float | None = None) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values) if mean is None else mean
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


class EWMADetector(BaseDetector):
    """Exponentially weighted moving average control chart.

    The control limit is ``k * sigma * sqrt(alpha / (2 - alpha) * (1 - (1-alpha)^{2i}))``
    exactly as in the NIST handbook; the reported statistic is the normalised deviation
    ``(z_i - mu0) / limit_i`` so that ``>= 1`` means "out of control".
    """

    name = "ewma"

    def threshold(self) -> float:
        return 1.0

    def score_series(self, series: Sequence[float]) -> list[float]:
        cfg = self.config
        warmup = min(cfg.warmup, max(2, len(series) // 3))
        baseline = list(series[:warmup])
        mu0 = _mean(baseline)
        sigma = max(_std(baseline, mu0), cfg.min_sigma)
        alpha = cfg.ewma_alpha
        z = mu0
        out: list[float] = []
        for i, x in enumerate(series):
            z = alpha * x + (1 - alpha) * z
            if i < warmup:
                out.append(0.0)
                continue
            spread = math.sqrt((alpha / (2 - alpha)) * (1 - (1 - alpha) ** (2 * (i + 1))))
            limit = cfg.ewma_k * sigma * spread
            limit = max(limit, cfg.min_sigma)
            out.append((z - mu0) / limit)  # one-sided: we only care about growth
        return out


class CUSUMDetector(BaseDetector):
    """Tabular (one-sided, upper) CUSUM on standardised observations.

    ``S_i = max(0, S_{i-1} + (x_i - mu0)/sigma - k)``; alarm when ``S_i >= h``.
    Reported statistic is ``S_i / h`` so ``>= 1`` means "out of control".
    """

    name = "cusum"

    def threshold(self) -> float:
        return 1.0

    def score_series(self, series: Sequence[float]) -> list[float]:
        cfg = self.config
        warmup = min(cfg.warmup, max(2, len(series) // 3))
        baseline = list(series[:warmup])
        mu0 = _mean(baseline)
        sigma = max(_std(baseline, mu0), cfg.min_sigma)
        s = 0.0
        out: list[float] = []
        for i, x in enumerate(series):
            s = max(0.0, s + (x - mu0) / sigma - cfg.cusum_k)
            out.append(0.0 if i < warmup else s / max(cfg.cusum_h, cfg.min_sigma))
        return out


class ThresholdDetector(BaseDetector):
    """Naive "it is already big" baseline — the thing a product manager would ship first.

    Deliberately kept: it is the reference point that shows whether CUSUM/EWMA buy us
    any *lead time* at all.
    """

    name = "threshold"

    def threshold(self) -> float:
        return 1.0

    def score_series(self, series: Sequence[float]) -> list[float]:
        cfg = self.config
        warmup = min(cfg.warmup, max(2, len(series) // 3))
        baseline = list(series[:warmup])
        mu0 = _mean(baseline)
        sigma = max(_std(baseline, mu0), cfg.min_sigma)
        limit = mu0 + cfg.ewma_k * sigma
        out: list[float] = []
        for i, x in enumerate(series):
            out.append(0.0 if i < warmup else x / max(limit, cfg.min_sigma))
        return out


DETECTORS = {d.name: d for d in (EWMADetector, CUSUMDetector, ThresholdDetector)}


def build_detector(name: str, config: DetectorConfig | None = None) -> BaseDetector:
    try:
        cls = DETECTORS[name]
    except KeyError:
        raise ValueError(f"unknown detector {name!r}, available: {sorted(DETECTORS)}") from None
    return cls(config)
