#!/usr/bin/env python3
"""Run inference on a single image and save predicted masks."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hseg.hierarchy import (  # noqa: E402
    LEVEL0_COLOR_PALETTE,
    LEVEL1_COLOR_PALETTE,
    LEVEL2_COLOR_PALETTE,
)
from hseg.model import build_model  # noqa: E402
from hseg.utils import ensure_dir, load_yaml, select_device  # noqa: E402

LOGGER = logging.getLogger(__name__)

CRITICAL_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("model",),
    ("data", "image_size"),
    ("data", "mean"),
    ("data", "std"),
)


def configure_logging() -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_PALETTE_L0 = [list(rgb) for rgb in LEVEL0_COLOR_PALETTE]
DEFAULT_PALETTE_L1 = [list(rgb) for rgb in LEVEL1_COLOR_PALETTE]
DEFAULT_PALETTE_L2 = [list(rgb) for rgb in LEVEL2_COLOR_PALETTE]


@dataclass(frozen=True)
class InferCliArgs:
    """Command line arguments for inference."""

    config: str | None
    checkpoint: str
    image: str
    output_dir: str
    allow_config_mismatch: bool


def parse_args() -> InferCliArgs:
    """Parse command line arguments.

    Returns:
        InferCliArgs: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Infer hierarchical segmentation for one image")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional external config. If omitted, uses config embedded in checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/infer")
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow external config values to differ from checkpoint embedded config.",
    )
    ns = parser.parse_args()

    return InferCliArgs(
        config=ns.config,
        checkpoint=ns.checkpoint,
        image=ns.image,
        output_dir=ns.output_dir,
        allow_config_mismatch=bool(ns.allow_config_mismatch),
    )


def colorize(mask: np.ndarray, palette: Sequence[Sequence[int]]) -> np.ndarray:
    """Convert class-id mask to RGB palette image.

    Args:
        mask (np.ndarray): HxW integer class-id mask.
        palette (Sequence[Sequence[int]]): Class color palette.

    Returns:
        np.ndarray: HxWx3 RGB mask.
    """
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        rgb[mask == idx] = np.array(color, dtype=np.uint8)
    return rgb


def _palette_from_config(
    palette_cfg: Mapping[str, Any],
    level_name: str,
    fallback: list[list[int]],
) -> list[list[int]]:
    """Load and validate one palette from config with fallback.

    Returns:
        list[list[int]]: Normalized RGB palette entries.
    """
    palette = palette_cfg.get(level_name, fallback)
    if not isinstance(palette, list):
        return fallback

    normalized: list[list[int]] = []
    for item in palette:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return fallback
        channels: list[int] = []
        for channel in item:
            if not isinstance(channel, (int, float)):
                return fallback
            value = int(channel)
            if value < 0 or value > 255:
                return fallback
            channels.append(value)
        normalized.append(channels)

    return normalized


def _nested_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> tuple[object, bool]:
    """Read nested mapping value by path and report whether it exists."""
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


def _format_path(path: tuple[str, ...]) -> str:
    """Convert tuple path to dotted representation."""
    return ".".join(path)


class InferenceApp:
    """Run end-to-end single-image inference workflow."""

    def __init__(self, args: InferCliArgs) -> None:
        """Initialize inference runtime state.

        Args:
            args (InferCliArgs): Parsed CLI arguments.
        """
        self.args = args

        self.cfg: dict[str, Any] = {}
        self.checkpoint: dict[str, Any] = {}
        self.model_cfg: dict[str, Any] = {}
        self.data_cfg: dict[str, Any] = {}
        self.inf_cfg: dict[str, Any] = {}

        self.device = torch.device("cpu")
        self.model: torch.nn.Module

        self.checkpoint_path = Path(".")
        self.image_path = Path(".")
        self.output_dir = Path(".")

        self.image_size = (384, 384)
        self.mean: tuple[float, ...] = DEFAULT_MEAN
        self.std: tuple[float, ...] = DEFAULT_STD
        self.palette_l0 = DEFAULT_PALETTE_L0
        self.palette_l1 = DEFAULT_PALETTE_L1
        self.palette_l2 = DEFAULT_PALETTE_L2

    def run(self) -> None:
        """Execute inference workflow and persist outputs."""
        self._prepare()
        image_tensor = self._preprocess_image()
        pred_l0, pred_l1, pred_l2 = self._predict(image_tensor)
        self._save_outputs(pred_l0=pred_l0, pred_l1=pred_l1, pred_l2=pred_l2)
        LOGGER.info("Saved outputs to: %s", self.output_dir)

    def _prepare(self) -> None:
        """Prepare config, device, paths, palettes, and model."""
        self.checkpoint_path = Path(self.args.checkpoint).resolve()
        self.checkpoint = self._load_checkpoint()
        self.cfg = self._resolve_config(self.checkpoint)

        self.model_cfg = self._as_mapping(self.cfg.get("model"), "model")
        self.data_cfg = self._as_mapping(self.cfg.get("data"), "data")
        self.inf_cfg = self._as_mapping(self.cfg.get("inference", {}), "inference")

        self.device = select_device(str(self.cfg.get("device", "cuda")))
        LOGGER.info("Using device: %s", self.device)

        self.image_path = Path(self.args.image).resolve()
        self.output_dir = ensure_dir(self.args.output_dir)

        self.image_size = tuple(self.data_cfg.get("image_size", [384, 384]))
        self.mean = tuple(self.data_cfg.get("mean", DEFAULT_MEAN))
        self.std = tuple(self.data_cfg.get("std", DEFAULT_STD))
        self.palette_l0, self.palette_l1, self.palette_l2 = self._resolve_palettes()

        self.model = self._load_model()

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint payload once and validate required fields."""
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint must contain a dictionary payload: {self.checkpoint_path}")
        if "model_state" not in checkpoint:
            raise KeyError(f"Checkpoint is missing 'model_state': {self.checkpoint_path}")
        return checkpoint

    def _collect_config_mismatches(
        self,
        checkpoint_cfg: Mapping[str, Any],
        external_cfg: Mapping[str, Any],
    ) -> list[str]:
        """Collect critical config differences between checkpoint and external config."""
        mismatches: list[str] = []
        for path in CRITICAL_CONFIG_PATHS:
            checkpoint_value, checkpoint_present = _nested_get(checkpoint_cfg, path)
            external_value, external_present = _nested_get(external_cfg, path)
            if not checkpoint_present and not external_present:
                continue
            if checkpoint_present != external_present or checkpoint_value != external_value:
                mismatches.append(
                    f"{_format_path(path)}: checkpoint={checkpoint_value!r} external={external_value!r}"
                )
        return mismatches

    def _resolve_config(self, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve runtime config with checkpoint config as default source of truth."""
        checkpoint_cfg_raw = checkpoint.get("config")
        checkpoint_cfg = None
        if checkpoint_cfg_raw is not None:
            checkpoint_cfg = self._as_mapping(checkpoint_cfg_raw, "checkpoint.config")

        if self.args.config is None:
            if checkpoint_cfg is None:
                raise ValueError(
                    "Checkpoint does not contain embedded config. "
                    "Provide --config for inference."
                )
            LOGGER.info("Using config embedded in checkpoint: %s", self.checkpoint_path)
            return checkpoint_cfg

        external_cfg = load_yaml(self.args.config)
        if checkpoint_cfg is None:
            LOGGER.warning(
                "Checkpoint has no embedded config; using external config: %s",
                self.args.config,
            )
            return external_cfg

        mismatches = self._collect_config_mismatches(checkpoint_cfg, external_cfg)
        if mismatches and not self.args.allow_config_mismatch:
            preview = "\n".join(f"- {item}" for item in mismatches[:8])
            raise ValueError(
                "External config differs from checkpoint embedded config for critical fields.\n"
                "Use --allow-config-mismatch to override.\n"
                f"{preview}"
            )

        if mismatches:
            LOGGER.warning(
                "Using external config with %s critical mismatch(es) due to --allow-config-mismatch.",
                len(mismatches),
            )
        else:
            LOGGER.info("External config matches checkpoint critical fields: %s", self.args.config)
        return external_cfg

    @staticmethod
    def _as_mapping(payload: object, key_name: str) -> dict[str, Any]:
        """Normalize optional config section into mapping."""
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError(f"Config key '{key_name}' must be a mapping when provided.")

    def _resolve_palettes(self) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """Resolve L0/L1/L2 palettes from config."""
        palette_cfg_raw = self.inf_cfg.get("palette", {})
        palette_cfg = palette_cfg_raw if isinstance(palette_cfg_raw, dict) else {}

        palette_l0 = _palette_from_config(palette_cfg, "level0", DEFAULT_PALETTE_L0)
        palette_l1 = _palette_from_config(palette_cfg, "level1", DEFAULT_PALETTE_L1)
        palette_l2 = _palette_from_config(palette_cfg, "level2", DEFAULT_PALETTE_L2)
        return palette_l0, palette_l1, palette_l2

    def _load_model(self) -> torch.nn.Module:
        """Build model and load checkpoint weights.

        Returns:
            torch.nn.Module: Ready-to-run model in eval mode.
        """
        model = build_model(self.model_cfg).to(self.device)
        model.load_state_dict(self.checkpoint["model_state"])
        model.eval()
        return model

    def _preprocess_image(self) -> torch.Tensor:
        """Load and preprocess input image tensor.

        Returns:
            torch.Tensor: Batched normalized input tensor.
        """
        image = Image.open(self.image_path).convert("RGB")
        image_resized = F.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)

        image_tensor = F.to_tensor(image_resized)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)
        return image_tensor.unsqueeze(0).to(self.device)

    def _predict(self, image_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model forward pass and decode hierarchical predictions."""
        with torch.no_grad():
            outputs = self.model(image_tensor)

        pred_l0 = outputs["logits_l0"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_l1 = outputs["logits_l1"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_l2 = outputs["logits_l2"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred_l0, pred_l1, pred_l2

    def _save_outputs(self, pred_l0: np.ndarray, pred_l1: np.ndarray, pred_l2: np.ndarray) -> None:
        """Save predicted masks as NPY and PNG outputs."""
        stem = self.image_path.stem

        np.save(self.output_dir / f"{stem}_pred_l0.npy", pred_l0)
        np.save(self.output_dir / f"{stem}_pred_l1.npy", pred_l1)
        np.save(self.output_dir / f"{stem}_pred_l2.npy", pred_l2)

        Image.fromarray(colorize(pred_l0, self.palette_l0)).save(self.output_dir / f"{stem}_pred_l0.png")
        Image.fromarray(colorize(pred_l1, self.palette_l1)).save(self.output_dir / f"{stem}_pred_l1.png")
        Image.fromarray(colorize(pred_l2, self.palette_l2)).save(self.output_dir / f"{stem}_pred_l2.png")


def main() -> None:
    """Entrypoint for CLI execution."""
    configure_logging()
    app = InferenceApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
