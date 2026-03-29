#!/usr/bin/env python3
"""Visualize model predictions vs ground truth masks for a dataset split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hseg.dataset import SampleRecord, discover_records, load_splits  # noqa: E402
from hseg.hierarchy import (  # noqa: E402
    LEVEL0_COLOR_PALETTE,
    LEVEL1_COLOR_PALETTE,
    LEVEL2_COLOR_PALETTE,
    fine_to_level0,
    fine_to_level1,
    validate_mask_ids,
)
from hseg.model import build_model  # noqa: E402
from hseg.utils import ensure_dir, load_yaml, select_device  # noqa: E402

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )

FINE_PALETTE = np.array(LEVEL2_COLOR_PALETTE, dtype=np.uint8)
L1_PALETTE = np.array(LEVEL1_COLOR_PALETTE, dtype=np.uint8)
L0_PALETTE = np.array(LEVEL0_COLOR_PALETTE, dtype=np.uint8)

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VisualizePredArgs:
    """Command line arguments for prediction visualization."""

    config: str
    checkpoint: str
    data_root: str | None
    split: str
    output_dir: str
    max_samples: int
    alpha: float


def parse_args() -> VisualizePredArgs:
    """Parse command line arguments.

    Returns:
        VisualizePredArgs: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="Visualize model predictions on a split")
    parser.add_argument("--config", type=str, default="configs/deeplabv3_mnv3_ft.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/test_predictions",
        help="Directory to save panels and masks",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--alpha", type=float, default=0.55)
    ns = parser.parse_args()

    return VisualizePredArgs(
        config=ns.config,
        checkpoint=ns.checkpoint,
        data_root=ns.data_root,
        split=ns.split,
        output_dir=ns.output_dir,
        max_samples=ns.max_samples,
        alpha=ns.alpha,
    )


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert class-id mask to RGB image with palette."""
    if mask.min() < 0 or mask.max() >= len(palette):
        raise ValueError(f"Mask class id out of palette range [0, {len(palette) - 1}]")
    return palette[mask]


def _sanitize_path_for_report(path_value: str | Path, workspace_root: Path = ROOT) -> str:
    """Return repo-relative path string for report summaries."""
    raw = str(path_value).strip()
    if not raw:
        return ""

    file_uri = raw.startswith("file:")
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    elif raw.startswith("file:"):
        raw = raw[len("file:") :]

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        rel_text = candidate.as_posix()
        if not file_uri:
            return rel_text
        if rel_text in {"", "."}:
            return "file:."
        if rel_text.startswith(("./", "../")):
            return f"file:{rel_text}"
        return f"file:./{rel_text}"

    try:
        rel = candidate.resolve().relative_to(workspace_root.resolve())
        rel_text = rel.as_posix() if str(rel) else "."
    except ValueError:
        tail = candidate.name or "path"
        rel_text = f"external/{tail}"

    if not file_uri:
        return rel_text
    if rel_text in {"", "."}:
        return "file:."
    if rel_text.startswith(("./", "../")):
        return f"file:{rel_text}"
    return f"file:./{rel_text}"


def overlay(image: np.ndarray, color_mask: np.ndarray, fg: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay class mask on original image."""
    out = image.astype(np.float32).copy()
    fg3 = np.repeat(fg[..., None], 3, axis=2)
    out[fg3] = (1.0 - alpha) * out[fg3] + alpha * color_mask.astype(np.float32)[fg3]
    return np.clip(out, 0, 255).astype(np.uint8)


def with_title(img: np.ndarray, title: str, header_h: int = 28) -> Image.Image:
    """Attach title header to image panel."""
    panel = Image.new("RGB", (img.shape[1], img.shape[0] + header_h), color=(20, 20, 20))
    panel.paste(Image.fromarray(img), (0, header_h))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill=(235, 235, 235), font=ImageFont.load_default())
    return panel


def make_grid(rows: list[list[Image.Image]], spacer: int = 8) -> Image.Image:
    """Compose tiled grid from image rows."""
    row_images: list[Image.Image] = []
    spacer_img_w: Image.Image | None = None
    spacer_img_h: Image.Image | None = None

    for row in rows:
        row_h = max(im.height for im in row)
        row_w = sum(im.width for im in row) + spacer * (len(row) - 1)
        canvas = Image.new("RGB", (row_w, row_h), color=(10, 10, 10))
        x = 0
        for i, im in enumerate(row):
            canvas.paste(im, (x, 0))
            x += im.width
            if i + 1 < len(row):
                if spacer_img_w is None or spacer_img_w.height != row_h:
                    spacer_img_w = Image.new("RGB", (spacer, row_h), color=(30, 30, 30))
                canvas.paste(spacer_img_w, (x, 0))
                x += spacer
        row_images.append(canvas)

    full_w = max(im.width for im in row_images)
    full_h = sum(im.height for im in row_images) + spacer * (len(row_images) - 1)
    out = Image.new("RGB", (full_w, full_h), color=(10, 10, 10))
    y = 0
    for i, row_im in enumerate(row_images):
        out.paste(row_im, (0, y))
        y += row_im.height
        if i + 1 < len(row_images):
            if spacer_img_h is None or spacer_img_h.width != full_w:
                spacer_img_h = Image.new("RGB", (full_w, spacer), color=(30, 30, 30))
            out.paste(spacer_img_h, (0, y))
            y += spacer

    return out


def write_readme(path: Path, num_samples: int) -> None:
    """Write README legend for generated prediction visuals."""
    lines = [
        "# Test Prediction Visualizations",
        "",
        f"Generated samples: **{num_samples}**",
        "",
        "Each panel contains two rows:",
        "- Row 1: `Original | GT L2 Mask | Pred L2 Mask | GT L2 Overlay | Pred L2 Overlay`",
        "- Row 2: `GT L1 Mask | Pred L1 Mask | GT L0 Mask | Pred L0 Mask`",
        "",
        "## Level2 Colors",
        "- `0 background`: `(0, 0, 0)`",
        "- `1 low_hand`: `(255, 120, 60)`",
        "- `2 torso`: `(255, 70, 70)`",
        "- `3 low_leg`: `(50, 135, 255)`",
        "- `4 head`: `(255, 225, 50)`",
        "- `5 up_leg`: `(70, 235, 120)`",
        "- `6 up_hand`: `(185, 80, 255)`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


class TestPredictionVisualizerApp:
    """Run end-to-end prediction visualization workflow."""

    def __init__(self, args: VisualizePredArgs) -> None:
        """Initialize visualizer runtime state."""
        self.args = args

        self.cfg: dict[str, Any] = {}
        self.data_cfg: dict[str, Any] = {}
        self.model_cfg: dict[str, Any] = {}
        self.device = torch.device("cpu")

        self.data_root = Path(".")
        self.split_file = Path(".")
        self.records: list[SampleRecord] = []

        self.model: torch.nn.Module
        self.image_size = (384, 384)
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD

        self.output_dir = Path(".")
        self.panels_dir = Path(".")
        self.l2_gt_dir = Path(".")
        self.l2_pred_dir = Path(".")
        self.l1_gt_dir = Path(".")
        self.l1_pred_dir = Path(".")
        self.l0_gt_dir = Path(".")
        self.l0_pred_dir = Path(".")

    def run(self) -> None:
        """Render and save visualizations for configured split."""
        self._prepare()

        progress = tqdm(self.records, desc=f"visualize-{self.args.split}")
        for index, record in enumerate(progress, start=1):
            self._process_record(index=index, record=record)

        write_readme(self.output_dir / "README.md", num_samples=len(self.records))

        summary = {
            "config": _sanitize_path_for_report(self.args.config),
            "checkpoint": _sanitize_path_for_report(self.args.checkpoint),
            "split": self.args.split,
            "num_samples": len(self.records),
            "output_dir": _sanitize_path_for_report(self.output_dir),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("%s", json.dumps(summary, indent=2))

    def _prepare(self) -> None:
        """Prepare config, records, model, and output directories."""
        self.cfg = load_yaml(self.args.config)

        self.data_cfg = self._as_mapping(self.cfg.get("data"), "data")
        self.model_cfg = self._as_mapping(self.cfg.get("model"), "model")
        self.device = select_device(str(self.cfg.get("device", "cuda")))

        self.data_root = Path(self.args.data_root or self.data_cfg["root"]).resolve()
        self.split_file = self.data_root / self.data_cfg.get("split_file", "splits.json")
        self.records = self._load_split_records(self.args.split)

        self.image_size = tuple(self.data_cfg.get("image_size", [384, 384]))
        self.mean = tuple(self.data_cfg.get("mean", DEFAULT_MEAN))
        self.std = tuple(self.data_cfg.get("std", DEFAULT_STD))

        self.model = self._load_model()
        self._prepare_output_dirs()

    @staticmethod
    def _as_mapping(payload: object, key_name: str) -> dict[str, Any]:
        """Validate and normalize config section as mapping."""
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError(f"Config key '{key_name}' must be a mapping when provided.")

    def _load_split_records(self, split_name: str) -> list[SampleRecord]:
        """Load records for requested dataset split."""
        split_map = load_splits(self.split_file)
        if split_name not in split_map:
            raise KeyError(
                f"Split '{split_name}' is missing in {self.split_file}. "
                f"Available splits: {sorted(split_map.keys())}"
            )

        all_records = discover_records(
            self.data_root,
            image_dir=self.data_cfg.get("image_dir", "JPEGImages"),
            mask_dir=self.data_cfg.get("mask_dir", "gt_masks"),
        )
        split_stems = set(split_map[split_name])
        split_records = [record for record in all_records if record.stem in split_stems]
        if not split_records:
            raise RuntimeError(f"No records found for split '{split_name}'")

        if self.args.max_samples > 0:
            return split_records[: self.args.max_samples]
        return split_records

    def _load_model(self) -> torch.nn.Module:
        """Build model and load checkpoint state."""
        model = build_model(self.model_cfg).to(self.device)

        checkpoint_path = Path(self.args.checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state" not in checkpoint:
            raise KeyError(f"Checkpoint is missing 'model_state': {checkpoint_path}")

        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    def _prepare_output_dirs(self) -> None:
        """Create output directories for all artifact groups."""
        self.output_dir = ensure_dir(self.args.output_dir)
        self.panels_dir = ensure_dir(self.output_dir / "panels")
        self.l2_gt_dir = ensure_dir(self.output_dir / "masks_level2_gt")
        self.l2_pred_dir = ensure_dir(self.output_dir / "masks_level2_pred")
        self.l1_gt_dir = ensure_dir(self.output_dir / "masks_level1_gt")
        self.l1_pred_dir = ensure_dir(self.output_dir / "masks_level1_pred")
        self.l0_gt_dir = ensure_dir(self.output_dir / "masks_level0_gt")
        self.l0_pred_dir = ensure_dir(self.output_dir / "masks_level0_pred")

    def _process_record(self, index: int, record: SampleRecord) -> None:
        """Process one sample and save panel plus masks."""
        image = Image.open(record.image_path).convert("RGB")
        mask_l2 = np.load(record.mask_path)
        if mask_l2.ndim != 2:
            raise ValueError(f"Expected 2D mask at {record.mask_path}, got shape {mask_l2.shape}")
        validate_mask_ids(mask_l2)
        if image.width != int(mask_l2.shape[1]) or image.height != int(mask_l2.shape[0]):
            raise ValueError(
                f"Image/mask size mismatch for '{record.stem}': "
                f"image={(image.height, image.width)}, mask={mask_l2.shape}"
            )

        image_resized = F.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask_l2_pil = Image.fromarray(mask_l2.astype(np.uint8))
        mask_l2_resized = np.array(
            F.resize(mask_l2_pil, self.image_size, interpolation=InterpolationMode.NEAREST),
            dtype=np.uint8,
        )

        x = F.to_tensor(image_resized)
        x = F.normalize(x, mean=self.mean, std=self.std).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

        pred_l0 = outputs["logits_l0"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_l1 = outputs["logits_l1"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_l2 = outputs["logits_l2"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        gt_l0 = fine_to_level0(mask_l2_resized).astype(np.uint8)
        gt_l1 = fine_to_level1(mask_l2_resized).astype(np.uint8)
        gt_l2 = mask_l2_resized.astype(np.uint8)

        gt_l2_color = colorize(gt_l2, FINE_PALETTE)
        pred_l2_color = colorize(pred_l2, FINE_PALETTE)
        gt_l1_color = colorize(gt_l1, L1_PALETTE)
        pred_l1_color = colorize(pred_l1, L1_PALETTE)
        gt_l0_color = colorize(gt_l0, L0_PALETTE)
        pred_l0_color = colorize(pred_l0, L0_PALETTE)

        image_np = np.array(image_resized, dtype=np.uint8)
        gt_overlay = overlay(image_np, gt_l2_color, gt_l2 > 0, alpha=self.args.alpha)
        pred_overlay = overlay(image_np, pred_l2_color, pred_l2 > 0, alpha=self.args.alpha)

        row1 = [
            with_title(image_np, "Original"),
            with_title(gt_l2_color, "GT L2 Mask"),
            with_title(pred_l2_color, "Pred L2 Mask"),
            with_title(gt_overlay, "GT L2 Overlay"),
            with_title(pred_overlay, "Pred L2 Overlay"),
        ]
        row2 = [
            with_title(gt_l1_color, "GT L1 Mask"),
            with_title(pred_l1_color, "Pred L1 Mask"),
            with_title(gt_l0_color, "GT L0 Mask"),
            with_title(pred_l0_color, "Pred L0 Mask"),
        ]
        panel = make_grid([row1, row2], spacer=8)

        stem = record.stem
        panel.save(self.panels_dir / f"{index:03d}_{stem}_panel.jpg", quality=95)
        Image.fromarray(gt_l2_color).save(self.l2_gt_dir / f"{index:03d}_{stem}_gt_l2.png")
        Image.fromarray(pred_l2_color).save(self.l2_pred_dir / f"{index:03d}_{stem}_pred_l2.png")
        Image.fromarray(gt_l1_color).save(self.l1_gt_dir / f"{index:03d}_{stem}_gt_l1.png")
        Image.fromarray(pred_l1_color).save(self.l1_pred_dir / f"{index:03d}_{stem}_pred_l1.png")
        Image.fromarray(gt_l0_color).save(self.l0_gt_dir / f"{index:03d}_{stem}_gt_l0.png")
        Image.fromarray(pred_l0_color).save(self.l0_pred_dir / f"{index:03d}_{stem}_pred_l0.png")


def main() -> None:
    """Entrypoint for CLI execution."""
    configure_logging()
    app = TestPredictionVisualizerApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
