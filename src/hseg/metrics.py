"""Metrics for hierarchical segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .hierarchy import FINE_CLASS_NAMES, LEVEL0_CLASS_NAMES, LEVEL1_CLASS_NAMES


@dataclass
class IoUResult:
    miou_l0: float
    miou_l1: float
    miou_l2: float
    iou_l0: dict[str, float]
    iou_l1: dict[str, float]
    iou_l2: dict[str, float]


def _fast_confusion(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]
    k = target * num_classes + pred
    bins = torch.bincount(k, minlength=num_classes * num_classes)
    return bins.view(num_classes, num_classes)


def _iou_from_confusion(conf: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    conf = conf.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return iou.cpu().numpy(), (denom > 0).cpu().numpy()


def _mean_iou_excluding_absent(iou_values: np.ndarray, present_mask: np.ndarray) -> float:
    valid_values = iou_values[present_mask]
    if valid_values.size == 0:
        return 0.0
    return float(np.mean(valid_values))


class HierarchicalIoUMeter:
    """Accumulates confusion matrices and computes hierarchical mIoU values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.conf_l0 = torch.zeros((2, 2), dtype=torch.int64)
        self.conf_l1 = torch.zeros((3, 3), dtype=torch.int64)
        self.conf_l2 = torch.zeros((7, 7), dtype=torch.int64)

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        pred_l0 = outputs["logits_l0"].argmax(dim=1).cpu()
        pred_l1 = outputs["logits_l1"].argmax(dim=1).cpu()
        pred_l2 = outputs["logits_l2"].argmax(dim=1).cpu()

        tgt_l0 = targets["mask_l0"].detach().cpu()
        tgt_l1 = targets["mask_l1"].detach().cpu()
        tgt_l2 = targets["mask_l2"].detach().cpu()

        self.conf_l0 += _fast_confusion(pred_l0, tgt_l0, 2)
        self.conf_l1 += _fast_confusion(pred_l1, tgt_l1, 3)
        self.conf_l2 += _fast_confusion(pred_l2, tgt_l2, 7)

    def compute(self) -> IoUResult:
        iou_l0_values, present_l0 = _iou_from_confusion(self.conf_l0)
        iou_l1_values, present_l1 = _iou_from_confusion(self.conf_l1)
        iou_l2_values, present_l2 = _iou_from_confusion(self.conf_l2)

        iou_l0 = {name: float(iou_l0_values[idx]) for idx, name in enumerate(LEVEL0_CLASS_NAMES)}
        iou_l1 = {name: float(iou_l1_values[idx]) for idx, name in enumerate(LEVEL1_CLASS_NAMES)}
        iou_l2 = {name: float(iou_l2_values[idx]) for idx, name in enumerate(FINE_CLASS_NAMES)}

        # Assignment requires background exclusion.
        miou_l0 = _mean_iou_excluding_absent(iou_l0_values[1:2], present_l0[1:2])
        miou_l1 = _mean_iou_excluding_absent(iou_l1_values[1:3], present_l1[1:3])
        miou_l2 = _mean_iou_excluding_absent(iou_l2_values[1:7], present_l2[1:7])

        return IoUResult(
            miou_l0=miou_l0,
            miou_l1=miou_l1,
            miou_l2=miou_l2,
            iou_l0=iou_l0,
            iou_l1=iou_l1,
            iou_l2=iou_l2,
        )
