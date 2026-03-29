#!/usr/bin/env python3
"""Create colorized mask visualizations for dataset inspection."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hseg.hierarchy import (  # noqa: E402
    FINE_CLASS_NAMES,
    LEVEL0_CLASS_NAMES,
    LEVEL0_COLOR_PALETTE,
    LEVEL1_CLASS_NAMES,
    LEVEL1_COLOR_PALETTE,
    LEVEL2_COLOR_PALETTE,
    fine_to_level0,
    fine_to_level1,
    validate_mask_ids,
)

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

FINE_NAMES = list(FINE_CLASS_NAMES)
L1_NAMES = list(LEVEL1_CLASS_NAMES)
L0_NAMES = list(LEVEL0_CLASS_NAMES)


@dataclass(frozen=True)
class VisualizeMasksArgs:
    """Command line arguments for mask visualization."""

    data_root: str
    output_dir: str
    num_samples: int
    seed: int
    alpha: float


def parse_args() -> VisualizeMasksArgs:
    """Parse command line arguments.

    Returns:
        VisualizeMasksArgs: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="Visualize Pascal-Part masks on top of images")
    parser.add_argument("--data-root", type=str, default="data/Pascal-part")
    parser.add_argument("--output-dir", type=str, default="reports/visual_analysis")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.55)
    ns = parser.parse_args()
    return VisualizeMasksArgs(
        data_root=ns.data_root,
        output_dir=ns.output_dir,
        num_samples=ns.num_samples,
        seed=ns.seed,
        alpha=ns.alpha,
    )


def discover_pairs(data_root: Path) -> list[tuple[str, Path, Path]]:
    """Discover aligned image/mask file pairs."""
    image_dir = data_root / "JPEGImages"
    mask_dir = data_root / "gt_masks"

    images: dict[str, Path] = {}
    for p in sorted(image_dir.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images[p.stem] = p

    pairs: list[tuple[str, Path, Path]] = []
    for m in sorted(mask_dir.glob("*.npy")):
        if m.stem in images:
            pairs.append((m.stem, images[m.stem], m))
    return pairs


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert a class-id mask into an RGB image using palette."""
    if mask.min() < 0 or mask.max() >= len(palette):
        raise ValueError(f"Mask class id out of palette range [0, {len(palette) - 1}]")
    return palette[mask]


def overlay(image: np.ndarray, color_mask: np.ndarray, fg: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay color mask over original image on foreground pixels."""
    out = image.astype(np.float32).copy()
    fg3 = np.repeat(fg[..., None], 3, axis=2)
    out[fg3] = (1.0 - alpha) * out[fg3] + alpha * color_mask.astype(np.float32)[fg3]
    return np.clip(out, 0, 255).astype(np.uint8)


def with_title(img: np.ndarray, title: str, height: int = 28) -> Image.Image:
    """Attach a title header to an image panel."""
    panel = Image.new("RGB", (img.shape[1], img.shape[0] + height), color=(20, 20, 20))
    panel.paste(Image.fromarray(img), (0, height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill=(235, 235, 235), font=ImageFont.load_default())
    return panel


def make_panel(
    image: np.ndarray,
    mask_l2: np.ndarray,
    mask_l1: np.ndarray,
    mask_l0: np.ndarray,
    alpha: float,
) -> Image.Image:
    """Build a single multi-view visualization panel."""
    color_l2 = colorize(mask_l2, FINE_PALETTE)
    color_l1 = colorize(mask_l1, L1_PALETTE)
    color_l0 = colorize(mask_l0, L0_PALETTE)

    panel_original = with_title(image, "Original")
    panel_l2 = with_title(color_l2, "Level2 Mask")
    panel_l2_overlay = with_title(overlay(image, color_l2, mask_l2 > 0, alpha), "Level2 Overlay")
    panel_l1_overlay = with_title(overlay(image, color_l1, mask_l1 > 0, alpha), "Level1 Overlay")
    panel_l0_overlay = with_title(overlay(image, color_l0, mask_l0 > 0, alpha), "Level0 Overlay")

    spacer = Image.new("RGB", (8, panel_original.height), color=(30, 30, 30))
    width = (
        panel_original.width
        + panel_l2.width
        + panel_l2_overlay.width
        + panel_l1_overlay.width
        + panel_l0_overlay.width
        + 4 * spacer.width
    )
    out = Image.new("RGB", (width, panel_original.height), color=(10, 10, 10))

    x = 0
    for panel in [panel_original, panel_l2, panel_l2_overlay, panel_l1_overlay, panel_l0_overlay]:
        out.paste(panel, (x, 0))
        x += panel.width
        if panel is not panel_l0_overlay:
            out.paste(spacer, (x, 0))
            x += spacer.width
    return out


def write_legend(path: Path) -> None:
    """Write README legend for generated visual artifacts."""
    lines = [
        "# Visual Analysis Outputs",
        "",
        "Each panel has 5 views: `Original | Level2 Mask | Level2 Overlay | Level1 Overlay | Level0 Overlay`.",
        "",
        "## Class Color Legend",
        "",
        "### Level2",
    ]
    for idx, name in enumerate(FINE_NAMES):
        rgb = tuple(int(v) for v in FINE_PALETTE[idx])
        lines.append(f"- `{idx}: {name}` -> `{rgb}`")
    lines.append("")
    lines.append("### Level1")
    for idx, name in enumerate(L1_NAMES):
        rgb = tuple(int(v) for v in L1_PALETTE[idx])
        lines.append(f"- `{idx}: {name}` -> `{rgb}`")
    lines.append("")
    lines.append("### Level0")
    for idx, name in enumerate(L0_NAMES):
        rgb = tuple(int(v) for v in L0_PALETTE[idx])
        lines.append(f"- `{idx}: {name}` -> `{rgb}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


class VisualizeMasksApp:
    """Run end-to-end dataset mask visualization workflow."""

    def __init__(self, args: VisualizeMasksArgs) -> None:
        """Initialize workflow paths and options."""
        self.args = args
        self.data_root = Path(args.data_root).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.panels_dir = self.output_dir / "panels"
        self.masks_dir = self.output_dir / "masks_level2"

    def run(self) -> None:
        """Generate visualization panels and colorized masks."""
        random.seed(self.args.seed)
        self._ensure_dirs()

        pairs = discover_pairs(self.data_root)
        if not pairs:
            raise RuntimeError(f"No image/mask pairs found under {self.data_root}")

        sample_count = min(self.args.num_samples, len(pairs))
        selected = random.sample(pairs, k=sample_count)

        for idx, (stem, image_path, mask_path) in enumerate(selected, start=1):
            image, mask_l2 = self._load_sample(stem, image_path, mask_path)
            mask_l1 = fine_to_level1(mask_l2).astype(np.uint8)
            mask_l0 = fine_to_level0(mask_l2).astype(np.uint8)

            panel = make_panel(
                image=image,
                mask_l2=mask_l2,
                mask_l1=mask_l1,
                mask_l0=mask_l0,
                alpha=self.args.alpha,
            )
            panel.save(self.panels_dir / f"{idx:03d}_{stem}_panel.jpg", quality=95)

            color_l2 = colorize(mask_l2, FINE_PALETTE)
            Image.fromarray(color_l2).save(self.masks_dir / f"{idx:03d}_{stem}_l2_mask.png")

        write_legend(self.output_dir / "README.md")
        LOGGER.info("Generated %s samples", sample_count)
        LOGGER.info("Panels: %s", self.panels_dir)
        LOGGER.info("Color masks: %s", self.masks_dir)

    def _ensure_dirs(self) -> None:
        """Ensure output directory structure exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_sample(stem: str, image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load and validate one image/mask sample.

        Returns:
            tuple[np.ndarray, np.ndarray]: RGB image array and validated L2 mask.
        """
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask_l2 = np.load(mask_path)

        if mask_l2.ndim != 2:
            raise ValueError(f"Mask {mask_path} is not 2D: {mask_l2.shape}")
        if image.shape[0] != mask_l2.shape[0] or image.shape[1] != mask_l2.shape[1]:
            raise ValueError(
                f"Image/mask size mismatch for {stem}: image={image.shape[:2]}, mask={mask_l2.shape}"
            )

        validate_mask_ids(mask_l2)
        return image, mask_l2.astype(np.uint8, copy=False)


def main() -> None:
    """Entrypoint for CLI execution."""
    configure_logging()
    app = VisualizeMasksApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
