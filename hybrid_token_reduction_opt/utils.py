import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_shared.training_utils_opt import count_parameters
from hybrid_simple.utils import get_backbone_feature_dim, modify_backbone_for_feature_extraction


def apply_mixup_cutmix(xb, yb, mixup_alpha: float, cutmix_alpha: float):
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return xb, yb

    device = xb.device
    use_mixup = mixup_alpha > 0.0 and cutmix_alpha > 0.0 and (torch.rand((), device=device) < 0.5).item()
    if mixup_alpha > 0.0 and cutmix_alpha <= 0.0:
        use_mixup = True
    if cutmix_alpha > 0.0 and mixup_alpha <= 0.0:
        use_mixup = False

    perm = torch.randperm(xb.size(0), device=xb.device)
    x2, y2 = xb[perm], yb[perm]

    if use_mixup:
        alpha = torch.tensor(mixup_alpha, device=device, dtype=torch.float32)
        lam = torch.distributions.Beta(alpha, alpha).sample()
        xb_mixed = lam * xb + (1.0 - lam) * x2
        yb_mixed = lam * yb + (1.0 - lam) * y2
        return xb_mixed, yb_mixed

    alpha = torch.tensor(cutmix_alpha, device=device, dtype=torch.float32)
    lam = torch.distributions.Beta(alpha, alpha).sample()
    _, _, H, W = xb.shape

    cut_ratio = torch.sqrt(torch.clamp(1.0 - lam, min=0.0)).item()
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = int(torch.randint(0, W, (), device=device).item())
    cy = int(torch.randint(0, H, (), device=device).item())

    x1 = max(cx - cut_w // 2, 0)
    x2b = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2b = min(cy + cut_h // 2, H)

    xb_mixed = xb.clone()
    xb_mixed[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]

    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / float(W * H))
    yb_mixed = lam_adj * yb + (1.0 - lam_adj) * y2

    return xb_mixed, yb_mixed


def compute_token_reduction_stats(keep_ratio: float, typical_n: int = 64) -> dict:
    k = max(1, int(typical_n * keep_ratio))

    original_complexity = typical_n ** 2
    reduced_complexity = k ** 2
    complexity_reduction_factor = original_complexity / reduced_complexity

    original_memory = typical_n ** 2
    reduced_memory = k ** 2
    memory_reduction_percent = (1 - reduced_memory / original_memory) * 100

    speedup_heuristic = (complexity_reduction_factor - 1) * 0.7 + 1

    return {
        "keep_ratio": keep_ratio,
        "num_input_tokens": typical_n,
        "num_output_tokens": k,
        "compression_factor": typical_n / k,
        "original_transformer_complexity": original_complexity,
        "reduced_transformer_complexity": reduced_complexity,
        "complexity_reduction_factor": complexity_reduction_factor,
        "memory_savings_percent": memory_reduction_percent,
        "speedup_heuristic": speedup_heuristic,
    }


def model_summary_with_token_analysis(
    model: nn.Module,
    model_name: str = "Hybrid Token Reduction",
    keep_ratio: float = 0.5,
) -> str:
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    token_stats = compute_token_reduction_stats(keep_ratio)

    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  MODEL SUMMARY: {model_name:<40}║
╚════════════════════════════════════════════════════════════════╝

Parâmetros:
  Treináveis:         {trainable:>15,}
  Totais:             {total:>15,}
  Taxa treino:        {trainable/total*100:>14.1f}%

Token Reduction Analysis:
  Keep Ratio:         {token_stats['keep_ratio']:>14.1%}
  Compression Factor: {token_stats['compression_factor']:>14.1f}x
  Tokens: {token_stats['num_input_tokens']} → {token_stats['num_output_tokens']}

Redução analítica da atenção (limite teórico, NÃO medido):
  Original:           O(n²) = O({token_stats['original_transformer_complexity']:>12,})
  Reduced:            O(k²) = O({token_stats['reduced_transformer_complexity']:>12,})
  Reduction Factor:   {token_stats['complexity_reduction_factor']:>14.1f}x
  Memory Savings:     {token_stats['memory_savings_percent']:>14.1f}%
  Speedup teórico:    {token_stats['speedup_heuristic']:>14.1f}x
  (o ganho REAL está em train_throughput_img_s no CSV de métricas)
"""
    return summary


def analyze_token_selection(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: str = "cuda",
) -> dict:
    model.eval()
    with torch.no_grad():
        input_batch = input_batch.to(device)

        _, analysis = model.forward_with_token_analysis(input_batch)

        scores = analysis["token_scores"]
        kept_ratio = analysis["kept_ratio"]
        compression = 1 / kept_ratio

        scores_np = scores.cpu().numpy()

        return {
            "kept_ratio": kept_ratio,
            "compression_factor": compression,
            "scores_mean": scores_np.mean(),
            "scores_std": scores_np.std(),
            "scores_min": scores_np.min(),
            "scores_max": scores_np.max(),
            "num_input_tokens": analysis["num_input_tokens"],
            "num_selected_tokens": analysis["num_selected_tokens"],
            "raw_scores": scores,
            "selected_indices": analysis["selected_indices"],
        }


def format_token_analysis_report(analysis: dict, model_name: str = "Token Reduction") -> str:
    report = f"""
╔════════════════════════════════════════════════════════════════╗
║  TOKEN SELECTION ANALYSIS: {model_name:<25}║
╚════════════════════════════════════════════════════════════════╝

Compression:
  Kept Ratio:         {analysis['kept_ratio']:.1%}
  Compression Factor: {analysis['compression_factor']:.1f}x
  Tokens: {analysis['num_input_tokens']} → {analysis['num_selected_tokens']}

Token Importance Scores:
  Mean:               {analysis['scores_mean']:.4f}
  Std Dev:            {analysis['scores_std']:.4f}
  Min:                {analysis['scores_min']:.4f}
  Max:                {analysis['scores_max']:.4f}
  Range:              {analysis['scores_max'] - analysis['scores_min']:.4f}
"""
    return report
