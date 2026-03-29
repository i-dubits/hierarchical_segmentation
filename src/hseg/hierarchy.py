"""Hierarchy definitions and conversion helpers for Pascal-Part subset labels."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

FINE_CLASS_NAMES = [
    "background",
    "low_hand",
    "torso",
    "low_leg",
    "head",
    "up_leg",
    "up_hand",
]

LEVEL0_CLASS_NAMES = ["background", "body"]
LEVEL1_CLASS_NAMES = ["background", "upper_body", "lower_body"]

LEVEL2_COLOR_PALETTE = (
    (0, 0, 0),      # background
    (255, 120, 60),  # low_hand
    (255, 70, 70),   # torso
    (50, 135, 255),  # low_leg
    (255, 225, 50),  # head
    (70, 235, 120),  # up_leg
    (185, 80, 255),  # up_hand
)
LEVEL1_COLOR_PALETTE = (
    (0, 0, 0),      # background
    (255, 145, 70),  # upper_body
    (65, 155, 255),  # lower_body
)
LEVEL0_COLOR_PALETTE = (
    (0, 0, 0),        # background
    (255, 255, 255),  # body
)

BACKGROUND_ID = 0
UPPER_FINE_IDS = (1, 2, 4, 6)
LOWER_FINE_IDS = (3, 5)

NUM_FINE_CLASSES = len(FINE_CLASS_NAMES)
NUM_LEVEL0_CLASSES = len(LEVEL0_CLASS_NAMES)
NUM_LEVEL1_CLASSES = len(LEVEL1_CLASS_NAMES)


def validate_mask_ids(mask: np.ndarray | torch.Tensor) -> None:
    """Raises ValueError if mask contains class ids outside [0..6]."""
    if isinstance(mask, np.ndarray):
        if mask.size == 0:
            return
        min_id = int(mask.min())
        max_id = int(mask.max())
    else:
        if mask.numel() == 0:
            return
        min_id = int(mask.min().item())
        max_id = int(mask.max().item())
    if min_id < 0 or max_id >= NUM_FINE_CLASSES:
        raise ValueError(
            f"Mask contains class ids outside [0..{NUM_FINE_CLASSES - 1}]: "
            f"min={min_id}, max={max_id}"
        )


def fine_to_level0(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Maps fine labels [0..6] to level-0 labels: 0=background, 1=body."""
    if isinstance(mask, np.ndarray):
        return (mask != BACKGROUND_ID).astype(np.int64)
    return (mask != BACKGROUND_ID).to(torch.long)


def _torch_isin(mask: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    out = torch.zeros_like(mask, dtype=torch.bool)
    for class_id in ids:
        out |= mask == class_id
    return out


def fine_to_level1(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Maps fine labels [0..6] to level-1 labels.

    Mapping:
      - 0 -> 0 (background)
      - {1,2,4,6} -> 1 (upper_body)
      - {3,5} -> 2 (lower_body)
    """
    if isinstance(mask, np.ndarray):
        out = np.zeros_like(mask, dtype=np.int64)
        out[np.isin(mask, UPPER_FINE_IDS)] = 1
        out[np.isin(mask, LOWER_FINE_IDS)] = 2
        return out

    out = torch.zeros_like(mask, dtype=torch.long)
    out[_torch_isin(mask, UPPER_FINE_IDS)] = 1
    out[_torch_isin(mask, LOWER_FINE_IDS)] = 2
    return out


def aggregate_fine_probabilities(
    probs_l2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregates fine-level probabilities to level-0 and level-1 probabilities.

    Args:
        probs_l2: Tensor shaped [B, 7, H, W], probabilities for fine classes.

    Returns:
        probs_l0: Tensor shaped [B, 2, H, W] (background, body)
        probs_l1: Tensor shaped [B, 3, H, W] (background, upper_body, lower_body)
    """
    if probs_l2.dim() != 4 or probs_l2.size(1) != NUM_FINE_CLASSES:
        raise ValueError(
            "Expected probs_l2 with shape [B, 7, H, W], "
            f"got {tuple(probs_l2.shape)}"
        )

    bg = probs_l2[:, 0:1]
    upper = probs_l2[:, list(UPPER_FINE_IDS)].sum(dim=1, keepdim=True)
    lower = probs_l2[:, list(LOWER_FINE_IDS)].sum(dim=1, keepdim=True)
    body = upper + lower

    probs_l0 = torch.cat([bg, body], dim=1)
    probs_l1 = torch.cat([bg, upper, lower], dim=1)
    return probs_l0, probs_l1
