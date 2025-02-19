# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional.regression import spearman_corrcoef


def masked_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked MSE loss between input and target."""
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_cross_entropy_loss(
    logits: torch.Tensor,  # (B, N, #bins) expression decoder output - Raw
    targets: torch.Tensor,  # (B, N) bin indices
    mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """Compute cross-entropy loss between input and output on non-masked
    locations."""
    # Ensure mask is boolean
    mask = mask.bool()

    # Flatten everything
    B, N, num_bins = logits.shape
    logits_flatten = logits.reshape(B * N, num_bins)
    targets_flatten = targets.reshape(B * N).long()
    mask_flatten = mask.reshape(B * N)

    # Cross-entropy across non-masked positions
    # PyTorch's F.cross_entropy expects shape [N, C] for logits, and [N] for targets.
    # target range should be from 0 to C-1.
    # bin indices range is: [1, ...,  num_bins] plus one bin for masked values equal to -2
    # target Values range from 1 to num_bins, change back to 0-based
    loss = F.cross_entropy(
        logits_flatten[mask_flatten],
        targets_flatten[mask_flatten] - 1,
        reduction="sum",
    )
    return loss


def criterion_neg_log_bernoulli(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the negative log-likelihood of Bernoulli distribution."""
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.LongTensor,
) -> torch.Tensor:
    """Compute the masked relative error between input and target."""
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


class MaskedCEMetric(Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_ce",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mask",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:

        B, N, num_bins = preds.shape
        preds_flatten = preds.reshape(B * N, num_bins)
        target_flatten = target.reshape(B * N).long()
        mask_flatten = mask.reshape(B * N)

        self.sum_ce += torch.nn.functional.cross_entropy(
            preds_flatten[mask_flatten],
            target_flatten[mask_flatten]
            - 1,  # -1 is to shift bins from [1, ..., num_bins] to [0, ..., num_bins-1]
            reduction="sum",
        )
        self.sum_mask += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum_ce / self.sum_mask


class MaskedMseMetric(Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_mse",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mask",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        mask = mask.float()
        self.sum_mse += torch.nn.functional.mse_loss(
            preds * mask,
            target * mask,
            reduction="sum",
        )
        self.sum_mask += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum_mse / self.sum_mask


class MaskedSpearmanMetric(Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_spearman",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_examples",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds: {preds.shape} and target: {target.shape} must have the same shape",
            )
        for pred_i, target_i, mask_i in zip(preds, target, mask):
            non_mask_preds = pred_i[mask_i].to("cpu")
            non_mask_targets = target_i[mask_i].to("cpu")
            self.sum_spearman += spearman_corrcoef(non_mask_preds, non_mask_targets)
            self.num_examples += 1

    def compute(self) -> torch.Tensor:
        return self.sum_spearman / self.num_examples
