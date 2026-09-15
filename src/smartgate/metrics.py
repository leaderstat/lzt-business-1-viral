"""Evaluation metrics.

Standard part mirrors scikit-learn's definitions
(https://scikit-learn.org/stable/modules/model_evaluation.html) but is implemented in
pure Python so the evaluation harness has zero runtime dependencies and identical
behaviour in CI. ``tests/test_metrics.py`` cross-checks the values against
scikit-learn when it happens to be installed.

Project-specific part (required by the sprint brief): Precision@K, Recall@K,
Lead Time and False Positive Rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass
class ClassificationReport:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    roc_auc: float
    pr_auc: float
    precision_at_k: float
    recall_at_k: float
    k: int
    mean_lead_time: float
    median_lead_time: float
    n_detected_positives: int

    def to_dict(self) -> dict:
        return asdict(self)


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, fp, tn, fn


def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp, fp, _, _ = confusion(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp, _, _, fn = confusion(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    p, r = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0


def false_positive_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    _, fp, tn, _ = confusion(y_true, y_pred)
    return fp / (fp + tn) if fp + tn else 0.0


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Rank-based AUC (Mann-Whitney U) with proper handling of tied scores."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(zip(y_score, y_true))
    i = 0
    rank_of: list[float] = [0.0] * len(order)
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank for the tie group
        for idx in range(i, j + 1):
            rank_of[idx] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, t) in zip(rank_of, order) if t == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def precision_recall_curve(
    y_true: Sequence[int], y_score: Sequence[float]
) -> tuple[list[float], list[float]]:
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    n_pos = sum(y_true)
    precisions, recalls = [], []
    tp = fp = 0
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        # only emit a point at the end of a tie group
        if rank < len(order) and y_score[order[rank]] == y_score[idx]:
            continue
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos if n_pos else 0.0)
    return precisions, recalls


def average_precision_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Step-wise average precision, the same estimator scikit-learn uses for PR-AUC."""
    if not any(y_true):
        return float("nan")
    precisions, recalls = precision_recall_curve(y_true, y_score)
    ap, prev_recall = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap


def precision_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
    """Share of true emerging trends inside the K highest-ranked alerts.

    This is the metric an analyst actually feels: they can only review K topics a day.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(y_score))
    if k == 0:
        return 0.0
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)[:k]
    return sum(y_true[i] for i in order) / k


def recall_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    n_pos = sum(y_true)
    if not n_pos:
        return float("nan")
    k = min(k, len(y_score))
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)[:k]
    return sum(y_true[i] for i in order) / n_pos


def lead_time(detected_index: int | None, viral_index: int | None) -> float | None:
    """Steps of early warning: how long before the topic became obviously viral we fired.

    Positive = early warning (the value we sell), negative = we were late,
    ``None`` = not applicable (missed detection or the topic never went viral).
    """
    if detected_index is None or viral_index is None:
        return None
    return float(viral_index - detected_index)


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    lead_times: Sequence[float | None] = (),
    k: int = 10,
) -> ClassificationReport:
    tp, fp, tn, fn = confusion(y_true, y_pred)
    lts = [v for v in lead_times if v is not None]
    return ClassificationReport(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        false_positive_rate=false_positive_rate(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        precision_at_k=precision_at_k(y_true, y_score, k),
        recall_at_k=recall_at_k(y_true, y_score, k),
        k=k,
        mean_lead_time=(sum(lts) / len(lts)) if lts else float("nan"),
        median_lead_time=_median(lts),
        n_detected_positives=len(lts),
    )
