"""Focal Loss used to train MTLLFM under class imbalance."""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal loss for (imbalanced) classification.

    Args:
        alpha: per-class weights, e.g. [0.45, 0.55] (higher for the minority class).
        gamma: focusing parameter (higher => focus more on hard examples).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=(0.45, 0.55), gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(inputs, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        ce_loss = -torch.log(pt + 1e-8)
        focal_factor = (1 - pt) ** self.gamma
        alpha_factor = self.alpha.to(targets.device)[targets]
        loss = alpha_factor * focal_factor * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
