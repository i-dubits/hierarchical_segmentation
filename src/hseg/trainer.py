"""Training and evaluation loops."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import hierarchical_loss
from .metrics import HierarchicalIoUMeter


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Move batch tensors to target device.

    Args:
        batch (dict[str, Any]): Input batch from dataloader.
        device (torch.device): Target device.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            Input image tensor and hierarchical target tensors.
    """
    image = batch["image"].to(device, non_blocking=True)
    targets = {
        "mask_l0": batch["mask_l0"].to(device, non_blocking=True),
        "mask_l1": batch["mask_l1"].to(device, non_blocking=True),
        "mask_l2": batch["mask_l2"].to(device, non_blocking=True),
    }
    return image, targets


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
    loss_weights: Mapping[str, float],
    consistency: str,
    class_weights_l2: torch.Tensor | None = None,
    dice_cfg: Mapping[str, Any] | None = None,
    lovasz_cfg: Mapping[str, Any] | None = None,
    tversky_cfg: Mapping[str, Any] | None = None,
    grad_accum_steps: int = 1,
    desc: str = "epoch",
) -> dict[str, Any]:
    """Run one training or evaluation epoch.

    Args:
        model (torch.nn.Module): Segmentation model.
        loader (DataLoader): Epoch dataloader.
        device (torch.device): Target compute device.
        optimizer (torch.optim.Optimizer | None): Optimizer for training mode.
        scaler (torch.cuda.amp.GradScaler | None): AMP gradient scaler.
        amp (bool): Whether to enable AMP.
        loss_weights (Mapping[str, float]): Hierarchical loss weights.
        consistency (str): Consistency loss type.
        class_weights_l2 (torch.Tensor | None): Optional CE class weights.
        dice_cfg (Mapping[str, Any] | None): Optional Dice config.
        lovasz_cfg (Mapping[str, Any] | None): Optional Lovasz config.
        tversky_cfg (Mapping[str, Any] | None): Optional Tversky config.
        grad_accum_steps (int): Number of gradient accumulation steps.
        desc (str): Progress bar description.

    Returns:
        dict[str, Any]: Aggregated loss and IoU metrics for the epoch.
    """
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    is_train = optimizer is not None
    model.train(is_train)

    meter = HierarchicalIoUMeter()
    running_loss = 0.0
    running_components: dict[str, float] = defaultdict(float)
    n_samples = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    total_steps = len(loader)
    progress = tqdm(loader, desc=desc, leave=False)
    for step_idx, batch in enumerate(progress, start=1):
        images, targets = _to_device(batch, device)
        batch_size = int(images.size(0))

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(images)
                loss, components = hierarchical_loss(
                    outputs=outputs,
                    targets=targets,
                    loss_weights=loss_weights,
                    consistency=consistency,
                    class_weights_l2=class_weights_l2,
                    dice_cfg=dice_cfg,
                    lovasz_cfg=lovasz_cfg,
                    tversky_cfg=tversky_cfg,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected during '{desc}' at step {step_idx}: {loss.detach().item()}"
                )

            if is_train:
                scaled_loss = loss / float(grad_accum_steps)
                should_step = (step_idx % grad_accum_steps == 0) or (step_idx == total_steps)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    scaled_loss.backward()
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        meter.update(outputs, targets)
        loss_value = float(loss.detach().item())
        running_loss += loss_value * batch_size
        for key, value in components.items():
            running_components[key] += float(value) * batch_size

        n_samples += batch_size
        progress.set_postfix(loss=f"{loss_value:.4f}")

    iou = meter.compute()

    metrics: dict[str, Any] = {
        "loss": running_loss / max(n_samples, 1),
        "miou_l0": iou.miou_l0,
        "miou_l1": iou.miou_l1,
        "miou_l2": iou.miou_l2,
        "iou_l0": iou.iou_l0,
        "iou_l1": iou.iou_l1,
        "iou_l2": iou.iou_l2,
    }
    for key, value in running_components.items():
        metrics[f"avg_{key}"] = value / max(n_samples, 1)

    return metrics
