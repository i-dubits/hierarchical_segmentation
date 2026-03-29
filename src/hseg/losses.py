"""Losses for hierarchical semantic segmentation."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .hierarchy import aggregate_fine_probabilities
from .utils import parse_bool


def _consistency_loss(
    probs_target: torch.Tensor,
    probs_pred: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Compute consistency penalty between hierarchical probability tensors.

    Args:
        probs_target (torch.Tensor): Target probability tensor.
        probs_pred (torch.Tensor): Predicted probability tensor.
        loss_type (str): Consistency loss type (`mse` or `kl`).

    Returns:
        torch.Tensor: Scalar consistency loss.
    """
    if loss_type == "mse":
        return F.mse_loss(probs_pred, probs_target)
    if loss_type == "kl":
        eps = 1e-8
        kl = F.kl_div(
            torch.log(probs_pred.clamp_min(eps)),
            probs_target.clamp_min(eps),
            reduction="none",
        )
        return kl.mean()
    raise ValueError(f"Unknown consistency loss type: {loss_type}")


def _resolve_consistency_mode(consistency: str) -> tuple[str, bool]:
    """Resolve consistency loss type and whether it is bidirectional."""
    normalized = consistency.strip().lower()
    bidirectional_aliases = {
        "mse_bidirectional": "mse",
        "bidirectional_mse": "mse",
        "kl_bidirectional": "kl",
        "bidirectional_kl": "kl",
    }
    if normalized in {"mse", "kl"}:
        return normalized, False
    if normalized in bidirectional_aliases:
        return bidirectional_aliases[normalized], True
    raise ValueError(
        "Unknown consistency mode. Expected one of: "
        "'mse', 'kl', 'mse_bidirectional', 'kl_bidirectional'. "
        f"Got: {consistency!r}"
    )


def _soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float,
    eps: float,
    exclude_background: bool,
) -> torch.Tensor:
    """Compute soft Dice loss for multi-class segmentation.

    Dice is averaged only across classes present in the target mask for the
    current batch. Absent classes are excluded from the mean to avoid
    artificially optimistic Dice values from smoothing.

    Args:
        probs (torch.Tensor): Softmax probabilities with shape `(B, C, H, W)`.
        targets (torch.Tensor): Integer class targets with shape `(B, H, W)`.
        smooth (float): Additive smoothing term.
        eps (float): Numerical stability epsilon.
        exclude_background (bool): Whether to exclude class `0`.

    Returns:
        torch.Tensor: Scalar Dice loss value.
    """
    num_classes = int(probs.size(1))
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    probs_for_dice = probs
    one_hot_for_dice = one_hot
    if exclude_background and num_classes > 1:
        probs_for_dice = probs_for_dice[:, 1:, ...]
        one_hot_for_dice = one_hot_for_dice[:, 1:, ...]

    intersection = torch.sum(probs_for_dice * one_hot_for_dice, dim=(0, 2, 3))
    cardinality = torch.sum(probs_for_dice + one_hot_for_dice, dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (cardinality + smooth + eps)

    present_target = torch.sum(one_hot_for_dice, dim=(0, 2, 3)) > 0
    if not torch.any(present_target):
        # No foreground classes present in batch (with background excluded).
        return probs_for_dice.sum() * 0.0

    return 1.0 - dice[present_target].mean()


def _tversky_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float,
    eps: float,
    exclude_background: bool,
    focal_gamma: float,
) -> torch.Tensor:
    """Compute Tversky (optionally focal-Tversky) loss for multi-class logits."""
    num_classes = int(probs.size(1))
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    probs_for_loss = probs
    one_hot_for_loss = one_hot
    if exclude_background and num_classes > 1:
        probs_for_loss = probs_for_loss[:, 1:, ...]
        one_hot_for_loss = one_hot_for_loss[:, 1:, ...]

    true_positive = torch.sum(probs_for_loss * one_hot_for_loss, dim=(0, 2, 3))
    false_positive = torch.sum(probs_for_loss * (1.0 - one_hot_for_loss), dim=(0, 2, 3))
    false_negative = torch.sum((1.0 - probs_for_loss) * one_hot_for_loss, dim=(0, 2, 3))

    tversky = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth + eps
    )
    per_class_loss = 1.0 - tversky
    if focal_gamma != 1.0:
        per_class_loss = per_class_loss.clamp_min(0.0).pow(focal_gamma)

    present_target = torch.sum(one_hot_for_loss, dim=(0, 2, 3)) > 0
    if not torch.any(present_target):
        return probs_for_loss.sum() * 0.0
    return per_class_loss[present_target].mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute Lovasz gradient for sorted binary foreground labels."""
    gts = torch.sum(gt_sorted)
    intersection = gts - torch.cumsum(gt_sorted, dim=0)
    union = gts + torch.cumsum(1.0 - gt_sorted, dim=0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(
    probs_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    class_indices: list[int],
    present_only: bool,
) -> torch.Tensor:
    """Compute Lovasz-Softmax on flattened class probabilities."""
    if probs_flat.numel() == 0:
        return probs_flat.sum() * 0.0

    losses: list[torch.Tensor] = []
    for class_idx in class_indices:
        foreground = (labels_flat == class_idx).to(dtype=probs_flat.dtype)
        if present_only and torch.sum(foreground) <= 0:
            continue

        class_prob = probs_flat[:, class_idx]
        errors = torch.abs(foreground - class_prob)
        if errors.numel() == 0:
            continue
        errors_sorted, permutation = torch.sort(errors, descending=True)
        fg_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if not losses:
        return probs_flat.sum() * 0.0
    return torch.stack(losses).mean()


def _lovasz_softmax_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    per_image: bool,
    ignore_index: int | None,
    exclude_background: bool,
    present_only: bool,
) -> torch.Tensor:
    """Compute Lovasz-Softmax multi-class loss."""
    num_classes = int(probs.size(1))
    class_indices = list(range(num_classes))
    if exclude_background and num_classes > 1:
        class_indices = class_indices[1:]
    if not class_indices:
        return probs.sum() * 0.0

    def _flatten(prob_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs_flat = prob_tensor.permute(0, 2, 3, 1).reshape(-1, num_classes)
        labels_flat = target_tensor.reshape(-1)
        if ignore_index is None:
            return probs_flat, labels_flat
        valid = labels_flat != int(ignore_index)
        return probs_flat[valid], labels_flat[valid]

    if per_image:
        per_image_losses: list[torch.Tensor] = []
        for probs_item, targets_item in zip(probs, targets):
            probs_flat, labels_flat = _flatten(probs_item.unsqueeze(0), targets_item.unsqueeze(0))
            if labels_flat.numel() == 0:
                continue
            per_image_losses.append(
                _lovasz_softmax_flat(
                    probs_flat=probs_flat,
                    labels_flat=labels_flat,
                    class_indices=class_indices,
                    present_only=present_only,
                )
            )
        if not per_image_losses:
            return probs.sum() * 0.0
        return torch.stack(per_image_losses).mean()

    probs_flat, labels_flat = _flatten(probs, targets)
    if labels_flat.numel() == 0:
        return probs.sum() * 0.0
    return _lovasz_softmax_flat(
        probs_flat=probs_flat,
        labels_flat=labels_flat,
        class_indices=class_indices,
        present_only=present_only,
    )


def hierarchical_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float],
    consistency: str = "mse",
    class_weights_l2: torch.Tensor | None = None,
    dice_cfg: Mapping[str, Any] | None = None,
    lovasz_cfg: Mapping[str, Any] | None = None,
    tversky_cfg: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted hierarchical loss with optional Dice term on L2.

    Args:
        outputs (Mapping[str, torch.Tensor]): Model output logits per hierarchy level.
        targets (Mapping[str, torch.Tensor]): Ground-truth masks per hierarchy level.
        loss_weights (Mapping[str, float]): Scalar weights for CE and consistency parts.
        consistency (str): Consistency loss type (`mse` or `kl`).
        class_weights_l2 (torch.Tensor | None): Optional CE weights for fine classes.
        dice_cfg (Mapping[str, Any] | None): Optional Dice config for fine level.
        lovasz_cfg (Mapping[str, Any] | None): Optional Lovasz config for fine level.
        tversky_cfg (Mapping[str, Any] | None): Optional Tversky config for fine level.

    Returns:
        tuple[torch.Tensor, dict[str, float]]:
            Total scalar loss tensor and detached scalar loss components.
    """
    logits_l0 = outputs["logits_l0"]
    logits_l1 = outputs["logits_l1"]
    logits_l2 = outputs["logits_l2"]

    target_l0 = targets["mask_l0"]
    target_l1 = targets["mask_l1"]
    target_l2 = targets["mask_l2"]

    ce_l0 = F.cross_entropy(logits_l0, target_l0)
    ce_l1 = F.cross_entropy(logits_l1, target_l1)
    ce_l2 = F.cross_entropy(logits_l2, target_l2, weight=class_weights_l2)

    probs_l2 = torch.softmax(logits_l2, dim=1)
    probs_l0_from_l2, probs_l1_from_l2 = aggregate_fine_probabilities(probs_l2)
    probs_l0 = torch.softmax(logits_l0, dim=1)
    probs_l1 = torch.softmax(logits_l1, dim=1)

    consistency_type, bidirectional_consistency = _resolve_consistency_mode(consistency)
    consistency_forward_l0 = _consistency_loss(
        probs_l0_from_l2.detach(),
        probs_l0,
        consistency_type,
    )
    consistency_forward_l1 = _consistency_loss(
        probs_l1_from_l2.detach(),
        probs_l1,
        consistency_type,
    )
    consistency_forward = 0.5 * (consistency_forward_l0 + consistency_forward_l1)

    consistency_reverse = torch.tensor(0.0, dtype=ce_l2.dtype, device=ce_l2.device)
    if bidirectional_consistency:
        consistency_reverse_l0 = _consistency_loss(
            probs_l0.detach(),
            probs_l0_from_l2,
            consistency_type,
        )
        consistency_reverse_l1 = _consistency_loss(
            probs_l1.detach(),
            probs_l1_from_l2,
            consistency_type,
        )
        consistency_reverse = 0.5 * (consistency_reverse_l0 + consistency_reverse_l1)

    consistency_loss = consistency_forward
    if bidirectional_consistency:
        consistency_loss = 0.5 * (consistency_forward + consistency_reverse)

    w_l0 = float(loss_weights.get("l0", 0.25))
    w_l1 = float(loss_weights.get("l1", 0.5))
    w_l2 = float(loss_weights.get("l2", 1.0))
    w_cons = float(loss_weights.get("consistency", 0.2))

    dice_cfg_resolved = dict(dice_cfg or {})
    dice_enabled = parse_bool(dice_cfg_resolved.get("enabled", False), "training.dice.enabled")
    dice_weight = float(dice_cfg_resolved.get("weight", 0.0))
    dice_smooth = float(dice_cfg_resolved.get("smooth", 1.0))
    dice_eps = float(dice_cfg_resolved.get("eps", 1e-7))
    dice_exclude_background = parse_bool(
        dice_cfg_resolved.get("exclude_background", True),
        "training.dice.exclude_background",
    )

    lovasz_cfg_resolved = dict(lovasz_cfg or {})
    lovasz_enabled = parse_bool(lovasz_cfg_resolved.get("enabled", False), "training.lovasz.enabled")
    lovasz_weight = float(lovasz_cfg_resolved.get("weight", 0.0))
    lovasz_per_image = parse_bool(
        lovasz_cfg_resolved.get("per_image", True),
        "training.lovasz.per_image",
    )
    lovasz_exclude_background = parse_bool(
        lovasz_cfg_resolved.get("exclude_background", True),
        "training.lovasz.exclude_background",
    )
    lovasz_present_only = parse_bool(
        lovasz_cfg_resolved.get("present_only", True),
        "training.lovasz.present_only",
    )
    lovasz_ignore_index_raw = lovasz_cfg_resolved.get("ignore_index", None)
    lovasz_ignore_index: int | None
    if lovasz_ignore_index_raw is None:
        lovasz_ignore_index = None
    else:
        lovasz_ignore_index = int(lovasz_ignore_index_raw)

    tversky_cfg_resolved = dict(tversky_cfg or {})
    tversky_enabled = parse_bool(tversky_cfg_resolved.get("enabled", False), "training.tversky.enabled")
    tversky_weight = float(tversky_cfg_resolved.get("weight", 0.0))
    tversky_alpha = float(tversky_cfg_resolved.get("alpha", 0.5))
    tversky_beta = float(tversky_cfg_resolved.get("beta", 0.5))
    tversky_smooth = float(tversky_cfg_resolved.get("smooth", 1.0))
    tversky_eps = float(tversky_cfg_resolved.get("eps", 1e-7))
    tversky_focal_gamma = float(tversky_cfg_resolved.get("focal_gamma", 1.0))
    tversky_exclude_background = parse_bool(
        tversky_cfg_resolved.get("exclude_background", True),
        "training.tversky.exclude_background",
    )

    if dice_weight < 0.0:
        raise ValueError(f"dice weight must be >= 0, got {dice_weight}")
    if dice_smooth < 0.0:
        raise ValueError(f"dice smooth must be >= 0, got {dice_smooth}")
    if dice_eps <= 0.0:
        raise ValueError(f"dice eps must be > 0, got {dice_eps}")
    if lovasz_weight < 0.0:
        raise ValueError(f"lovasz weight must be >= 0, got {lovasz_weight}")
    if tversky_weight < 0.0:
        raise ValueError(f"tversky weight must be >= 0, got {tversky_weight}")
    if tversky_alpha < 0.0 or tversky_beta < 0.0:
        raise ValueError(
            f"tversky alpha and beta must be >= 0, got alpha={tversky_alpha}, beta={tversky_beta}"
        )
    if (tversky_alpha + tversky_beta) <= 0.0:
        raise ValueError("tversky alpha + beta must be > 0")
    if tversky_smooth < 0.0:
        raise ValueError(f"tversky smooth must be >= 0, got {tversky_smooth}")
    if tversky_eps <= 0.0:
        raise ValueError(f"tversky eps must be > 0, got {tversky_eps}")
    if tversky_focal_gamma <= 0.0:
        raise ValueError(f"tversky focal_gamma must be > 0, got {tversky_focal_gamma}")
    if dice_enabled and dice_weight > 0.0 and tversky_enabled and tversky_weight > 0.0:
        raise ValueError("Enable either Dice or Tversky for L2 auxiliary loss, not both at once.")

    dice_l2 = torch.tensor(0.0, dtype=ce_l2.dtype, device=ce_l2.device)
    if dice_enabled and dice_weight > 0.0:
        dice_l2 = _soft_dice_loss(
            probs=probs_l2,
            targets=target_l2,
            smooth=dice_smooth,
            eps=dice_eps,
            exclude_background=dice_exclude_background,
        )

    lovasz_l2 = torch.tensor(0.0, dtype=ce_l2.dtype, device=ce_l2.device)
    if lovasz_enabled and lovasz_weight > 0.0:
        lovasz_l2 = _lovasz_softmax_loss(
            probs=probs_l2,
            targets=target_l2,
            per_image=lovasz_per_image,
            ignore_index=lovasz_ignore_index,
            exclude_background=lovasz_exclude_background,
            present_only=lovasz_present_only,
        )

    tversky_l2 = torch.tensor(0.0, dtype=ce_l2.dtype, device=ce_l2.device)
    if tversky_enabled and tversky_weight > 0.0:
        tversky_l2 = _tversky_loss(
            probs=probs_l2,
            targets=target_l2,
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=tversky_smooth,
            eps=tversky_eps,
            exclude_background=tversky_exclude_background,
            focal_gamma=tversky_focal_gamma,
        )

    total = (
        w_l2 * ce_l2
        + w_l1 * ce_l1
        + w_l0 * ce_l0
        + w_cons * consistency_loss
        + dice_weight * dice_l2
        + lovasz_weight * lovasz_l2
        + tversky_weight * tversky_l2
    )

    components = {
        "loss": float(total.detach().item()),
        "ce_l0": float(ce_l0.detach().item()),
        "ce_l1": float(ce_l1.detach().item()),
        "ce_l2": float(ce_l2.detach().item()),
        "consistency": float(consistency_loss.detach().item()),
        "consistency_forward": float(consistency_forward.detach().item()),
        "consistency_reverse": float(consistency_reverse.detach().item()),
        "dice_l2": float(dice_l2.detach().item()),
        "dice_weighted_l2": float((dice_weight * dice_l2).detach().item()),
        "lovasz_l2": float(lovasz_l2.detach().item()),
        "lovasz_weighted_l2": float((lovasz_weight * lovasz_l2).detach().item()),
        "tversky_l2": float(tversky_l2.detach().item()),
        "tversky_weighted_l2": float((tversky_weight * tversky_l2).detach().item()),
    }
    return total, components
