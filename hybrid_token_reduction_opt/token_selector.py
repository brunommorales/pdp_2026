import torch
import torch.nn as nn
from typing import Tuple, Optional


class TokenSelector(nn.Module):

    def __init__(
        self,
        token_dim: int,
        keep_ratio: float = 0.5,
        keep_k: Optional[int] = None,
        include_cls_token: bool = False,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.keep_ratio = keep_ratio
        self.keep_k = keep_k
        self.include_cls_token = include_cls_token

        self.score_layer = nn.Linear(token_dim, 1)

    def forward(self, tokens: torch.Tensor, score_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape

        if score_override is not None:
            scores = score_override
        else:
            scores = self.score_layer(tokens).squeeze(-1)

        cls_score = None
        if self.include_cls_token and N > 1:
            cls_score = scores[:, 0:1]
            scores_for_topk = scores[:, 1:]
            tokens_for_topk = tokens[:, 1:, :]
            cls_token = tokens[:, 0:1, :]
            n_without_cls = N - 1
        else:
            scores_for_topk = scores
            tokens_for_topk = tokens
            cls_token = None
            n_without_cls = N

        if self.keep_k is not None:
            k = min(self.keep_k, n_without_cls)
        else:
            k = max(1, int(n_without_cls * self.keep_ratio))

        _, top_indices = torch.topk(scores_for_topk, k=k, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)

        selected_tokens = torch.gather(
            tokens_for_topk,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, D)
        )

        if cls_token is not None:
            selected_tokens = torch.cat([cls_token, selected_tokens], dim=1)
            selected_indices_with_cls = torch.full((B, 1), 0, dtype=top_indices.dtype, device=top_indices.device)
            selected_indices = torch.cat([selected_indices_with_cls, top_indices + 1], dim=1)
        else:
            selected_indices = top_indices

        return selected_tokens, selected_indices, scores

    def compute_keep_ratio(self, tokens: torch.Tensor) -> float:
        N = tokens.size(1)

        if self.include_cls_token and N > 1:
            n_without_cls = N - 1
        else:
            n_without_cls = N

        if self.keep_k is not None:
            k = min(self.keep_k, n_without_cls)
        else:
            k = max(1, int(n_without_cls * self.keep_ratio))

        if self.include_cls_token:
            return (k + 1) / N
        else:
            return k / N


class DynamicTokenSelector(nn.Module):

    def __init__(
        self,
        token_dim: int,
        min_kept_ratio: float = 0.1,
        max_kept_ratio: float = 1.0,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.min_kept_ratio = min_kept_ratio
        self.max_kept_ratio = max_kept_ratio

        self.ratio_predictor = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.score_layer = nn.Linear(token_dim, 1)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape

        predicted_ratio_logits = self.ratio_predictor(tokens.mean(dim=1))
        predicted_ratio = (
            self.min_kept_ratio
            + (self.max_kept_ratio - self.min_kept_ratio) * predicted_ratio_logits
        )

        k_per_sample = torch.clamp(
            (predicted_ratio * N).long().squeeze(-1),
            min=1,
            max=N - 1,
        )

        scores = self.score_layer(tokens).squeeze(-1)

        k_actual = k_per_sample.min().item()

        top_scores, top_indices = torch.topk(scores, k=k_actual, dim=1)

        selected_tokens = torch.gather(
            tokens,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, D),
        )

        return selected_tokens, top_indices, predicted_ratio


def visualize_token_importance(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        scores_np = scores.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(scores_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Token Importance Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Token Importance Scores')
        axes[0].grid(True, alpha=0.3)

        avg_scores = scores_np.mean(axis=0)
        axes[1].plot(avg_scores, marker='o', linestyle='-', linewidth=2)
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Average Importance Score')
        axes[1].set_title('Token Importance by Position')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Token importance visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Matplotlib não disponível para visualização")
