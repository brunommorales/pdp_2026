from __future__ import annotations

import math

import torch


SPECIFICITY_TARGET_SENSITIVITY = 0.95


def _flatten_binary_inputs(y_true: torch.Tensor, y_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if y_true.numel() != y_prob.numel():
        raise ValueError("y_true and y_prob must have the same length")

    device = y_prob.device
    y_true = y_true.reshape(-1).to(device=device, dtype=torch.float32)
    y_prob = y_prob.reshape(-1).to(device=device, dtype=torch.float32)
    finite_mask = torch.isfinite(y_true) & torch.isfinite(y_prob)
    if not bool(finite_mask.all().item()):
        y_true = y_true[finite_mask]
        y_prob = y_prob[finite_mask]
    if y_prob.numel() == 0:
        return y_true, y_prob
    return y_true, y_prob.clamp(1e-7, 1.0 - 1e-7)


def _compute_auc_rank_sum(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true, y_score = _flatten_binary_inputs(y_true, y_score)
    if y_true.numel() == 0 or y_score.numel() == 0:
        return float("nan")
    positives = y_true > 0.5
    negatives = ~positives

    n_pos = int(positives.sum().item())
    n_neg = int(negatives.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(y_score, stable=True)
    sorted_scores = y_score[order]

    new_group = torch.ones_like(sorted_scores, dtype=torch.bool)
    if sorted_scores.numel() > 1:
        new_group[1:] = sorted_scores[1:] != sorted_scores[:-1]

    group_starts = torch.nonzero(new_group, as_tuple=False).flatten()
    group_ends = torch.cat(
        [group_starts[1:], torch.tensor([sorted_scores.numel()], device=sorted_scores.device, dtype=group_starts.dtype)]
    )
    counts = group_ends - group_starts
    avg_ranks = 0.5 * ((group_starts + 1).to(torch.float64) + group_ends.to(torch.float64))
    group_ids = torch.repeat_interleave(torch.arange(counts.numel(), device=sorted_scores.device), counts)
    sorted_ranks = avg_ranks[group_ids]

    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks

    auc = (ranks[positives].sum() - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc.item())


def compute_auc(y_true: torch.Tensor, y_prob: torch.Tensor) -> float:
    return _compute_auc_rank_sum(y_true, y_prob)


def compute_sens_spec(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    y_true, y_prob = _flatten_binary_inputs(y_true, y_prob)
    if y_true.numel() == 0 or y_prob.numel() == 0:
        return float("nan"), float("nan")
    y_true = y_true > 0.5

    y_pred = y_prob >= threshold

    tp = float(torch.sum(y_true & y_pred).item())
    fn = float(torch.sum(y_true & (~y_pred)).item())
    tn = float(torch.sum((~y_true) & (~y_pred)).item())
    fp = float(torch.sum((~y_true) & y_pred).item())

    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)

    return float(sens), float(spec)


def _compute_confusion_counts(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    y_true, y_prob = _flatten_binary_inputs(y_true, y_prob)
    if y_true.numel() == 0 or y_prob.numel() == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    y_true = y_true > 0.5
    y_pred = y_prob >= threshold

    tp = float(torch.sum(y_true & y_pred).item())
    fn = float(torch.sum(y_true & (~y_pred)).item())
    tn = float(torch.sum((~y_true) & (~y_pred)).item())
    fp = float(torch.sum((~y_true) & y_pred).item())
    return tp, fn, tn, fp


def compute_specificity_at_sensitivity(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    target_sensitivity: float = SPECIFICITY_TARGET_SENSITIVITY,
) -> float:
    y_true, y_prob = _flatten_binary_inputs(y_true, y_prob)
    if y_true.numel() == 0 or y_prob.numel() == 0:
        return float("nan")

    positives = y_true > 0.5
    n_pos = int(positives.sum().item())
    n_neg = int((~positives).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    target = float(target_sensitivity)
    if target <= 0.0:
        return 1.0
    if target > 1.0:
        return float("nan")

    order = torch.argsort(y_prob, descending=True, stable=True)
    sorted_scores = y_prob[order]
    sorted_true = positives[order]

    group_end = torch.ones(sorted_scores.numel(), device=sorted_scores.device, dtype=torch.bool)
    if sorted_scores.numel() > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_idxs = torch.nonzero(group_end, as_tuple=False).flatten()

    tp = torch.cumsum(sorted_true.to(torch.float64), dim=0)[threshold_idxs]
    fp = torch.cumsum((~sorted_true).to(torch.float64), dim=0)[threshold_idxs]
    tpr = torch.cat(
        [
            torch.zeros(1, device=y_prob.device, dtype=torch.float64),
            tp / float(n_pos),
        ]
    )
    fpr = torch.cat(
        [
            torch.zeros(1, device=y_prob.device, dtype=torch.float64),
            fp / float(n_neg),
        ]
    )

    target_tensor = torch.tensor(target, device=y_prob.device, dtype=torch.float64)
    target_indices = torch.nonzero(tpr >= target_tensor, as_tuple=False).flatten()
    if target_indices.numel() == 0:
        return 0.0

    idx = int(target_indices[0].item())
    if idx == 0:
        fpr_at_target = fpr[0]
    else:
        tpr_lo = tpr[idx - 1]
        tpr_hi = tpr[idx]
        fpr_lo = fpr[idx - 1]
        fpr_hi = fpr[idx]
        if float(torch.abs(tpr_hi - tpr_lo).item()) <= 1e-12:
            fpr_at_target = torch.minimum(fpr_lo, fpr_hi)
        else:
            fraction = (target_tensor - tpr_lo) / (tpr_hi - tpr_lo)
            fpr_at_target = fpr_lo + fraction * (fpr_hi - fpr_lo)

    specificity = 1.0 - float(fpr_at_target.item())
    return float(max(0.0, min(1.0, specificity)))


def compute_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5) -> dict:
    auc = compute_auc(y_true, y_prob)
    tp, fn, tn, fp = _compute_confusion_counts(y_true, y_prob, threshold)
    if any(math.isnan(value) for value in (tp, fn, tn, fp)):
        nan = float("nan")
        return {
            "auc": auc,
            "precision": nan,
            "sensitivity": nan,
            "recall": nan,
            "specificity": nan,
            "specificity_at_sens95": nan,
            "f1": nan,
            "threshold": threshold,
        }
    precision = tp / (tp + fp + 1e-7)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    spec95 = compute_specificity_at_sensitivity(y_true, y_prob)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    return {
        "auc": auc,
        "precision": float(precision),
        "sensitivity": float(sens),
        "recall": float(sens),
        "specificity": float(spec),
        "specificity_at_sens95": float(spec95),
        "f1": float(f1),
        "threshold": threshold,
    }


def find_optimal_threshold(y_true: torch.Tensor, y_prob: torch.Tensor, metric: str = "f1") -> tuple[float, float]:
    y_true, y_prob = _flatten_binary_inputs(y_true, y_prob)
    if y_true.numel() == 0 or y_prob.numel() == 0:
        nan = float("nan")
        return nan, nan
    y_true = y_true > 0.5

    thresholds = torch.arange(0.1, 0.95, 0.01, device=y_prob.device, dtype=y_prob.dtype)
    y_pred = y_prob.unsqueeze(0) >= thresholds.unsqueeze(1)
    y_true = y_true.unsqueeze(0)

    tp = (y_true & y_pred).sum(dim=1).to(torch.float32)
    fn = (y_true & (~y_pred)).sum(dim=1).to(torch.float32)
    tn = ((~y_true) & (~y_pred)).sum(dim=1).to(torch.float32)
    fp = ((~y_true) & y_pred).sum(dim=1).to(torch.float32)

    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)

    if metric == "f1":
        precision = tp / (tp + fp + 1e-7)
        recall = sens
        values = 2 * (precision * recall) / (precision + recall + 1e-7)
    elif metric == "gmean":
        values = torch.sqrt(sens * spec)
    elif metric == "youdenj":
        values = sens + spec - 1
    else:
        raise ValueError(f"Unknown metric: {metric}")

    best_idx = int(torch.argmax(values).item())
    return float(thresholds[best_idx].item()), float(values[best_idx].item())
