#!/usr/bin/env python3
"""Dataset analysis for Pascal-Part hierarchical segmentation task."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def configure_logging() -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )

FINE_CLASSES = [
    "background",
    "low_hand",
    "torso",
    "low_leg",
    "head",
    "up_leg",
    "up_hand",
]
L0_CLASSES = ["background", "body"]
L1_CLASSES = ["background", "upper_body", "lower_body"]
L0_MAP = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
L1_MAP = np.array([0, 1, 1, 2, 1, 2, 1], dtype=np.int64)


@dataclass(frozen=True)
class AnalyzeCliArgs:
    """Command line arguments for dataset analysis."""

    data_root: str
    output_json: str
    output_md: str
    split_file: str | None


def parse_args() -> AnalyzeCliArgs:
    """Parse command line arguments.

    Returns:
        AnalyzeCliArgs: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="Analyze Pascal-Part dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Pascal-part",
        help="Path containing JPEGImages and gt_masks",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/dataset_stats.json",
        help="Path to write raw statistics JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/dataset_report.md",
        help="Path to write markdown report",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help=(
            "Optional split JSON path. "
            "If omitted, split stats are inferred from train_id.txt/val_id.txt."
        ),
    )
    ns = parser.parse_args()
    return AnalyzeCliArgs(
        data_root=ns.data_root,
        output_json=ns.output_json,
        output_md=ns.output_md,
        split_file=ns.split_file,
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for numeric values."""
    if values.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def load_id_file(path: Path) -> list[str]:
    """Load split IDs from file, skipping empty lines."""
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def split_name_for_stem(stem: str, split_stems: dict[str, set[str]]) -> str:
    """Resolve split name for a sample stem."""
    for split_name, members in split_stems.items():
        if stem in members:
            return split_name
    return "unassigned"


def load_split_file(path: Path) -> dict[str, list[str]]:
    """Load and validate split JSON from file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Split file must contain a non-empty JSON object: {path}")

    splits: dict[str, list[str]] = {}
    seen_stems: dict[str, str] = {}
    for split_name, stems_raw in payload.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError(f"Invalid split name in {path}: {split_name!r}")
        if not isinstance(stems_raw, list):
            raise ValueError(f"Split '{split_name}' must be a list in {path}")

        split_stems: list[str] = []
        split_seen: set[str] = set()
        for item in stems_raw:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"Split '{split_name}' contains non-string/empty stem in {path}: {item!r}"
                )
            if item in split_seen:
                raise ValueError(f"Split '{split_name}' contains duplicate stem '{item}' in {path}")
            if item in seen_stems:
                raise ValueError(
                    f"Stem '{item}' appears in multiple splits: "
                    f"'{seen_stems[item]}' and '{split_name}' in {path}"
                )
            split_seen.add(item)
            seen_stems[item] = split_name
            split_stems.append(item)
        splits[split_name] = split_stems
    return splits


def to_percent(numerator: int | float, denominator: int | float) -> float:
    """Convert ratio to percent safely."""
    if denominator == 0:
        return 0.0
    return float(numerator) * 100.0 / float(denominator)


def sanitize_path_for_report(path_value: str | Path, workspace_root: Path = ROOT) -> str:
    """Return repo-relative path string for report payloads."""
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


def compute_stats(data_root: Path, split_file: Path | None = None) -> dict[str, Any]:
    """Compute full dataset statistics for reporting.

    Args:
        data_root (Path): Dataset root directory.
        split_file (Path | None): Optional split JSON file path.

    Returns:
        dict[str, Any]: Nested dataset statistics payload.
    """
    image_dir = data_root / "JPEGImages"
    mask_dir = data_root / "gt_masks"
    if split_file is not None and not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    image_files = sorted(
        [
            p
            for p in image_dir.glob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    mask_files = sorted([p for p in mask_dir.glob("*.npy") if p.is_file()])

    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in image_files:
        images_by_stem[path.stem].append(path)

    masks_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in mask_files:
        masks_by_stem[path.stem].append(path)

    duplicate_image_stems = sorted([stem for stem, lst in images_by_stem.items() if len(lst) > 1])
    duplicate_mask_stems = sorted([stem for stem, lst in masks_by_stem.items() if len(lst) > 1])

    image_stems = set(images_by_stem.keys())
    mask_stems = set(masks_by_stem.keys())
    matched_stems = sorted(image_stems & mask_stems)
    image_only_stems = sorted(image_stems - mask_stems)
    mask_only_stems = sorted(mask_stems - image_stems)

    split_stems: dict[str, set[str]]
    split_definition: dict[str, Any]
    if split_file is not None:
        split_payload = load_split_file(split_file)
        split_stems = {name: set(stems) for name, stems in split_payload.items()}
        split_definition = {
            "source": "split_file",
            "split_file": sanitize_path_for_report(split_file),
            "split_counts": {name: int(len(stems)) for name, stems in split_payload.items()},
        }
    else:
        train_ids = set(load_id_file(data_root / "train_id.txt"))
        val_ids = set(load_id_file(data_root / "val_id.txt"))
        split_stems = {"train": train_ids, "val": val_ids}
        split_definition = {
            "source": "id_files",
            "train_ids_file": int(len(train_ids)),
            "val_ids_file": int(len(val_ids)),
            "intersection_ids": int(len(train_ids & val_ids)),
        }

    widths: list[int] = []
    heights: list[int] = []
    areas: list[int] = []
    aspect_ratios: list[float] = []
    orientations = Counter()
    image_sizes_counter = Counter()
    ext_counter = Counter()

    mask_shapes_counter = Counter()
    mask_dtype_counter = Counter()
    invalid_id_files: list[str] = []
    mismatched_size_files: list[str] = []
    empty_fg_files: list[str] = []

    pixel_counts_l2 = np.zeros(7, dtype=np.int64)
    pixel_counts_l1 = np.zeros(3, dtype=np.int64)
    pixel_counts_l0 = np.zeros(2, dtype=np.int64)

    image_presence_l2 = np.zeros(7, dtype=np.int64)
    image_presence_l1 = np.zeros(3, dtype=np.int64)
    image_presence_l0 = np.zeros(2, dtype=np.int64)

    area_ratio_present_l2: dict[int, list[float]] = defaultdict(list)
    area_ratio_present_l1: dict[int, list[float]] = defaultdict(list)
    area_ratio_present_l0: dict[int, list[float]] = defaultdict(list)

    fg_class_count_per_image: list[int] = []
    body_coverage_per_image: list[float] = []
    upper_coverage_per_image: list[float] = []
    lower_coverage_per_image: list[float] = []

    upper_lower_pattern = Counter()
    split_sample_counts = Counter()
    split_order = list(split_stems.keys())
    split_pixel_counts_l2: dict[str, np.ndarray] = {
        split_name: np.zeros(7, dtype=np.int64) for split_name in split_order
    }
    split_pixel_counts_l2["unassigned"] = np.zeros(7, dtype=np.int64)

    # Pairwise co-occurrence across foreground classes (1..6), image-level.
    cooc_l2_fg = np.zeros((6, 6), dtype=np.int64)
    valid_mask_samples = 0

    for stem in matched_stems:
        image_path = images_by_stem[stem][0]
        mask_path = masks_by_stem[stem][0]

        with Image.open(image_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            aspect = float(w) / float(h)
            aspect_ratios.append(aspect)
            image_sizes_counter[f"{w}x{h}"] += 1
            ext_counter[image_path.suffix.lower()] += 1
            if w > h:
                orientations["landscape"] += 1
            elif h > w:
                orientations["portrait"] += 1
            else:
                orientations["square"] += 1

        mask = np.load(mask_path)
        mask_dtype_counter[str(mask.dtype)] += 1
        if mask.ndim != 2:
            invalid_id_files.append(stem)
            continue

        mh, mw = int(mask.shape[0]), int(mask.shape[1])
        mask_shapes_counter[f"{mw}x{mh}"] += 1
        if mw != w or mh != h:
            mismatched_size_files.append(stem)

        min_id = int(mask.min())
        max_id = int(mask.max())
        if min_id < 0 or max_id > 6:
            invalid_id_files.append(stem)
            continue

        valid_mask_samples += 1
        l2 = mask.astype(np.int64, copy=False)
        l1 = L1_MAP[l2]
        l0 = L0_MAP[l2]

        total_pixels = int(l2.size)
        counts_l2 = np.bincount(l2.reshape(-1), minlength=7)
        counts_l1 = np.bincount(l1.reshape(-1), minlength=3)
        counts_l0 = np.bincount(l0.reshape(-1), minlength=2)

        pixel_counts_l2 += counts_l2
        pixel_counts_l1 += counts_l1
        pixel_counts_l0 += counts_l0

        present_l2 = counts_l2 > 0
        present_l1 = counts_l1 > 0
        present_l0 = counts_l0 > 0

        image_presence_l2 += present_l2.astype(np.int64)
        image_presence_l1 += present_l1.astype(np.int64)
        image_presence_l0 += present_l0.astype(np.int64)

        for class_id in range(7):
            if present_l2[class_id]:
                area_ratio_present_l2[class_id].append(float(counts_l2[class_id]) / float(total_pixels))
        for class_id in range(3):
            if present_l1[class_id]:
                area_ratio_present_l1[class_id].append(float(counts_l1[class_id]) / float(total_pixels))
        for class_id in range(2):
            if present_l0[class_id]:
                area_ratio_present_l0[class_id].append(float(counts_l0[class_id]) / float(total_pixels))

        fg_count = int(np.count_nonzero(present_l2[1:]))
        fg_class_count_per_image.append(fg_count)
        body_coverage = float(counts_l0[1]) / float(total_pixels)
        upper_coverage = float(counts_l1[1]) / float(total_pixels)
        lower_coverage = float(counts_l1[2]) / float(total_pixels)
        body_coverage_per_image.append(body_coverage)
        upper_coverage_per_image.append(upper_coverage)
        lower_coverage_per_image.append(lower_coverage)

        if fg_count == 0:
            empty_fg_files.append(stem)

        has_upper = bool(present_l1[1])
        has_lower = bool(present_l1[2])
        pattern = (
            "both"
            if has_upper and has_lower
            else "upper_only"
            if has_upper
            else "lower_only"
            if has_lower
            else "none"
        )
        upper_lower_pattern[pattern] += 1

        # Co-occurrence for foreground classes (1..6).
        fg_present = np.where(present_l2[1:])[0]  # 0..5 local index
        for i in fg_present:
            for j in fg_present:
                cooc_l2_fg[i, j] += 1

        split_name = split_name_for_stem(stem, split_stems)
        split_sample_counts[split_name] += 1
        split_pixel_counts_l2[split_name] += counts_l2

    num_samples = len(matched_stems)
    num_valid_samples = valid_mask_samples
    total_pixels_dataset = int(pixel_counts_l2.sum())

    foreground_pixels = int(pixel_counts_l2[1:].sum())
    upper_pixels = int(pixel_counts_l1[1])
    lower_pixels = int(pixel_counts_l1[2])

    fg_count_hist = Counter(fg_class_count_per_image)
    fg_count_distribution = {
            str(k): {
                "images": int(v),
                "ratio_percent": to_percent(v, num_valid_samples),
            }
        for k, v in sorted(fg_count_hist.items())
    }

    fine_rows = []
    for class_id, class_name in enumerate(FINE_CLASSES):
        pixel_count = int(pixel_counts_l2[class_id])
        image_count = int(image_presence_l2[class_id])
        fine_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "pixel_ratio_percent": to_percent(pixel_count, total_pixels_dataset),
                "image_count": image_count,
                "image_ratio_percent": to_percent(image_count, num_valid_samples),
                "present_area_ratio_summary": summarize(
                    np.array(area_ratio_present_l2[class_id], dtype=np.float64)
                ),
            }
        )

    level1_rows = []
    for class_id, class_name in enumerate(L1_CLASSES):
        pixel_count = int(pixel_counts_l1[class_id])
        image_count = int(image_presence_l1[class_id])
        level1_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "pixel_ratio_percent": to_percent(pixel_count, total_pixels_dataset),
                "image_count": image_count,
                "image_ratio_percent": to_percent(image_count, num_valid_samples),
                "present_area_ratio_summary": summarize(
                    np.array(area_ratio_present_l1[class_id], dtype=np.float64)
                ),
            }
        )

    level0_rows = []
    for class_id, class_name in enumerate(L0_CLASSES):
        pixel_count = int(pixel_counts_l0[class_id])
        image_count = int(image_presence_l0[class_id])
        level0_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "pixel_ratio_percent": to_percent(pixel_count, total_pixels_dataset),
                "image_count": image_count,
                "image_ratio_percent": to_percent(image_count, num_valid_samples),
                "present_area_ratio_summary": summarize(
                    np.array(area_ratio_present_l0[class_id], dtype=np.float64)
                ),
            }
        )

    split_stats: dict[str, dict[str, Any]] = {}
    for split_name, pixels_l2 in split_pixel_counts_l2.items():
        sample_count = int(split_sample_counts.get(split_name, 0))
        split_total_pixels = int(np.sum(pixels_l2))
        split_fg_pixels = int(np.sum(pixels_l2[1:]))
        split_stats[split_name] = {
            "samples": sample_count,
            "sample_ratio_percent": to_percent(sample_count, num_valid_samples),
            "total_pixels": split_total_pixels,
            "foreground_ratio_percent": to_percent(split_fg_pixels, split_total_pixels),
            "fine_class_pixel_ratio_percent": {
                FINE_CLASSES[i]: to_percent(int(pixels_l2[i]), split_total_pixels) for i in range(7)
            },
        }

    cooc_table = {
        FINE_CLASSES[i + 1]: {FINE_CLASSES[j + 1]: int(cooc_l2_fg[i, j]) for j in range(6)}
        for i in range(6)
    }

    stats: dict[str, Any] = {
        "dataset_root": sanitize_path_for_report(data_root),
        "inventory": {
            "num_image_files": int(len(image_files)),
            "num_mask_files": int(len(mask_files)),
            "num_matched_pairs": int(num_samples),
            "num_valid_masks_for_statistics": int(num_valid_samples),
            "image_extension_counts": dict(sorted(ext_counter.items())),
            "duplicate_image_stems_count": int(len(duplicate_image_stems)),
            "duplicate_mask_stems_count": int(len(duplicate_mask_stems)),
            "image_only_stems_count": int(len(image_only_stems)),
            "mask_only_stems_count": int(len(mask_only_stems)),
            "image_only_stems_examples": image_only_stems[:20],
            "mask_only_stems_examples": mask_only_stems[:20],
            "split_definition": split_definition,
        },
        "image_statistics": {
            "width_summary": summarize(np.array(widths, dtype=np.float64)),
            "height_summary": summarize(np.array(heights, dtype=np.float64)),
            "area_summary": summarize(np.array(areas, dtype=np.float64)),
            "aspect_ratio_w_over_h_summary": summarize(np.array(aspect_ratios, dtype=np.float64)),
            "orientation_counts": dict(orientations),
            "top_15_image_sizes": [
                {"size": size, "count": int(count)} for size, count in image_sizes_counter.most_common(15)
            ],
        },
        "mask_integrity": {
            "mask_shape_unique_count": int(len(mask_shapes_counter)),
            "top_15_mask_shapes": [
                {"shape": shape, "count": int(count)} for shape, count in mask_shapes_counter.most_common(15)
            ],
            "mask_dtype_counts": dict(mask_dtype_counter),
            "invalid_or_out_of_range_mask_count": int(len(invalid_id_files)),
            "invalid_or_out_of_range_mask_examples": invalid_id_files[:20],
            "image_mask_size_mismatch_count": int(len(mismatched_size_files)),
            "image_mask_size_mismatch_examples": mismatched_size_files[:20],
        },
        "class_balance": {
            "total_pixels": total_pixels_dataset,
            "foreground_pixels": foreground_pixels,
            "foreground_ratio_percent": to_percent(foreground_pixels, total_pixels_dataset),
            "empty_foreground_images_count": int(len(empty_fg_files)),
            "empty_foreground_images_examples": empty_fg_files[:20],
            "level0": level0_rows,
            "level1": level1_rows,
            "level2": fine_rows,
        },
        "hierarchy_specific": {
            "upper_pixels": upper_pixels,
            "lower_pixels": lower_pixels,
            "upper_ratio_within_body_percent": to_percent(upper_pixels, max(foreground_pixels, 1)),
            "lower_ratio_within_body_percent": to_percent(lower_pixels, max(foreground_pixels, 1)),
            "body_coverage_per_image_summary": summarize(
                np.array(body_coverage_per_image, dtype=np.float64)
            ),
            "upper_coverage_per_image_summary": summarize(
                np.array(upper_coverage_per_image, dtype=np.float64)
            ),
            "lower_coverage_per_image_summary": summarize(
                np.array(lower_coverage_per_image, dtype=np.float64)
            ),
            "upper_lower_presence_pattern": {
                name: {
                    "images": int(count),
                    "ratio_percent": to_percent(count, num_valid_samples),
                }
                for name, count in sorted(upper_lower_pattern.items())
            },
            "foreground_class_count_per_image_distribution": fg_count_distribution,
        },
        "cooccurrence": {
            "foreground_fine_class_image_cooccurrence": cooc_table,
        },
        "split_statistics": split_stats,
    }
    return stats


def _fmt(value: float, precision: int = 3) -> str:
    """Format float for markdown tables."""
    return f"{value:.{precision}f}"


def render_markdown(stats: dict[str, Any]) -> str:
    """Render markdown report from computed stats."""
    inv = stats["inventory"]
    img = stats["image_statistics"]
    mask = stats["mask_integrity"]
    balance = stats["class_balance"]
    hs = stats["hierarchy_specific"]
    split = stats["split_statistics"]
    cooc = stats["cooccurrence"]["foreground_fine_class_image_cooccurrence"]

    lines: list[str] = []
    lines.append("# Pascal-Part Dataset Report")
    lines.append("")
    lines.append("## 1) Dataset Inventory")
    lines.append("")
    lines.append(f"- Dataset root: `{stats['dataset_root']}`")
    lines.append(f"- Matched image/mask pairs: **{inv['num_matched_pairs']}**")
    lines.append(f"- Valid masks used for class/split statistics: **{inv['num_valid_masks_for_statistics']}**")
    lines.append(f"- Image files: {inv['num_image_files']}")
    lines.append(f"- Mask files: {inv['num_mask_files']}")
    lines.append(f"- Duplicate image stems: {inv['duplicate_image_stems_count']}")
    lines.append(f"- Duplicate mask stems: {inv['duplicate_mask_stems_count']}")
    lines.append(f"- Image-only stems: {inv['image_only_stems_count']}")
    lines.append(f"- Mask-only stems: {inv['mask_only_stems_count']}")
    split_definition = inv["split_definition"]
    if split_definition["source"] == "split_file":
        lines.append(f"- Split source file: `{split_definition['split_file']}`")
        split_counts = split_definition.get("split_counts", {})
        if split_counts:
            summary = ", ".join(f"{name}={count}" for name, count in split_counts.items())
            lines.append(f"- Declared split sizes: {summary}")
    else:
        lines.append(
            "- Split id files: "
            f"train={split_definition['train_ids_file']}, "
            f"val={split_definition['val_ids_file']}, "
            f"intersection={split_definition['intersection_ids']}"
        )
    lines.append("")
    lines.append("## 2) Image Geometry")
    lines.append("")
    ws = img["width_summary"]
    hs_summary = img["height_summary"]
    ar = img["aspect_ratio_w_over_h_summary"]
    lines.append(
        "- Width: "
        f"min={_fmt(ws['min'], 0)}, max={_fmt(ws['max'], 0)}, "
        f"mean={_fmt(ws['mean'])}, median={_fmt(ws['median'])}, p05={_fmt(ws['p05'])}, p95={_fmt(ws['p95'])}"
    )
    lines.append(
        "- Height: "
        f"min={_fmt(hs_summary['min'], 0)}, max={_fmt(hs_summary['max'], 0)}, "
        f"mean={_fmt(hs_summary['mean'])}, median={_fmt(hs_summary['median'])}, "
        f"p05={_fmt(hs_summary['p05'])}, p95={_fmt(hs_summary['p95'])}"
    )
    lines.append(
        "- Aspect ratio (W/H): "
        f"mean={_fmt(ar['mean'])}, median={_fmt(ar['median'])}, p05={_fmt(ar['p05'])}, p95={_fmt(ar['p95'])}"
    )
    orient = img["orientation_counts"]
    lines.append(
        "- Orientation counts: "
        f"landscape={orient.get('landscape', 0)}, portrait={orient.get('portrait', 0)}, square={orient.get('square', 0)}"
    )
    lines.append("")
    lines.append("Top image sizes (W x H):")
    lines.append("")
    lines.append("| Size | Count |")
    lines.append("|---|---:|")
    for row in img["top_15_image_sizes"]:
        lines.append(f"| {row['size']} | {row['count']} |")
    lines.append("")
    lines.append("## 3) Mask Integrity")
    lines.append("")
    lines.append(f"- Unique mask shapes: {mask['mask_shape_unique_count']}")
    lines.append(f"- Mask dtypes: {mask['mask_dtype_counts']}")
    lines.append(f"- Invalid/out-of-range masks: {mask['invalid_or_out_of_range_mask_count']}")
    lines.append(f"- Image-mask size mismatches: {mask['image_mask_size_mismatch_count']}")
    lines.append("")
    lines.append("## 4) Class Balance")
    lines.append("")
    lines.append(
        f"- Foreground pixel ratio (all non-background): **{_fmt(balance['foreground_ratio_percent'])}%**"
    )
    lines.append(f"- Empty-foreground images: {balance['empty_foreground_images_count']}")
    lines.append("")
    lines.append("### Level 2 (fine classes)")
    lines.append("")
    lines.append("| Class | Pixel % | Image Presence % | Median Area % (when present) |")
    lines.append("|---|---:|---:|---:|")
    for row in balance["level2"]:
        med = row["present_area_ratio_summary"]["median"] * 100.0
        lines.append(
            f"| {row['class_name']} | {_fmt(row['pixel_ratio_percent'])} | "
            f"{_fmt(row['image_ratio_percent'])} | {_fmt(med)} |"
        )
    lines.append("")
    lines.append("### Level 1 (upper/lower)")
    lines.append("")
    lines.append("| Class | Pixel % | Image Presence % | Median Area % (when present) |")
    lines.append("|---|---:|---:|---:|")
    for row in balance["level1"]:
        med = row["present_area_ratio_summary"]["median"] * 100.0
        lines.append(
            f"| {row['class_name']} | {_fmt(row['pixel_ratio_percent'])} | "
            f"{_fmt(row['image_ratio_percent'])} | {_fmt(med)} |"
        )
    lines.append("")
    lines.append("### Level 0 (body/background)")
    lines.append("")
    lines.append("| Class | Pixel % | Image Presence % | Median Area % (when present) |")
    lines.append("|---|---:|---:|---:|")
    for row in balance["level0"]:
        med = row["present_area_ratio_summary"]["median"] * 100.0
        lines.append(
            f"| {row['class_name']} | {_fmt(row['pixel_ratio_percent'])} | "
            f"{_fmt(row['image_ratio_percent'])} | {_fmt(med)} |"
        )
    lines.append("")
    lines.append("## 5) Hierarchy-Relevant Findings")
    lines.append("")
    lines.append(
        "- Body composition by pixels (within foreground): "
        f"upper={_fmt(hs['upper_ratio_within_body_percent'])}%, "
        f"lower={_fmt(hs['lower_ratio_within_body_percent'])}%"
    )
    lines.append(
        "- Body coverage per image: "
        f"median={_fmt(hs['body_coverage_per_image_summary']['median'] * 100.0)}%, "
        f"p05={_fmt(hs['body_coverage_per_image_summary']['p05'] * 100.0)}%, "
        f"p95={_fmt(hs['body_coverage_per_image_summary']['p95'] * 100.0)}%"
    )
    lines.append("")
    lines.append("Upper/lower presence patterns:")
    lines.append("")
    lines.append("| Pattern | Images | Ratio % |")
    lines.append("|---|---:|---:|")
    for name, payload in hs["upper_lower_presence_pattern"].items():
        lines.append(f"| {name} | {payload['images']} | {_fmt(payload['ratio_percent'])} |")
    lines.append("")
    lines.append("Foreground class count per image:")
    lines.append("")
    lines.append("| #FG classes | Images | Ratio % |")
    lines.append("|---:|---:|---:|")
    for num_classes, payload in hs["foreground_class_count_per_image_distribution"].items():
        lines.append(f"| {num_classes} | {payload['images']} | {_fmt(payload['ratio_percent'])} |")
    lines.append("")
    lines.append("## 6) Fine-Class Co-occurrence (Image-level)")
    lines.append("")
    header = "| class | " + " | ".join(cooc.keys()) + " |"
    sep = "|" + "---|" * (len(cooc) + 1)
    lines.append(header)
    lines.append(sep)
    for row_name, row_vals in cooc.items():
        row = "| " + row_name + " | " + " | ".join(str(row_vals[col]) for col in cooc.keys()) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## 7) Split Statistics")
    lines.append("")
    lines.append("| Split | Samples | Sample % | Foreground Pixel % |")
    lines.append("|---|---:|---:|---:|")
    for split_name, payload in split.items():
        lines.append(
            f"| {split_name} | {payload['samples']} | {_fmt(payload['sample_ratio_percent'])} | "
            f"{_fmt(payload['foreground_ratio_percent'])} |"
        )
    lines.append("")
    if "train" in split and "val" in split:
        train_ratios = split["train"]["fine_class_pixel_ratio_percent"]
        val_ratios = split["val"]["fine_class_pixel_ratio_percent"]
        lines.append("Fine-class pixel ratio by split (%):")
        lines.append("")
        lines.append("| Class | Train % | Val % | Delta (Val-Train, pp) |")
        lines.append("|---|---:|---:|---:|")
        for class_name in FINE_CLASSES:
            tr = float(train_ratios[class_name])
            vr = float(val_ratios[class_name])
            lines.append(
                f"| {class_name} | {_fmt(tr)} | {_fmt(vr)} | {_fmt(vr - tr)} |"
            )
        lines.append("")
    lines.append("## 8) Training Implications")
    lines.append("")
    lines.append("- Strong class imbalance at fine level is expected; use weighted CE or focal-like alternatives for level-2 head.")
    lines.append("- If upper/lower balance is skewed, monitor `mIoU^1` components separately (upper vs lower), not only mean.")
    lines.append("- Wide image-size diversity implies resize policy impacts small-part visibility; avoid overly aggressive downscaling.")
    lines.append("- Use hierarchy-consistency loss to stabilize coarse heads when fine classes are rare.")
    lines.append("- Consider split-stratified checks to ensure train/val class distributions are aligned.")
    lines.append("")
    lines.append(
        "_This report is auto-generated from local files. Raw values are available in `reports/dataset_stats.json`._"
    )
    lines.append("")
    return "\n".join(lines)


class DatasetAnalysisApp:
    """Run dataset analysis and write JSON/Markdown reports."""

    def __init__(self, args: AnalyzeCliArgs) -> None:
        """Initialize analysis paths from CLI arguments."""
        self.args = args

        self.data_root = Path(args.data_root).resolve()
        self.output_json = Path(args.output_json).resolve()
        self.output_md = Path(args.output_md).resolve()
        self.split_file = None
        if args.split_file is not None:
            candidate = Path(args.split_file).expanduser()
            if not candidate.is_absolute() and not candidate.exists():
                candidate = self.data_root / candidate
            self.split_file = candidate.resolve()

    def run(self) -> None:
        """Execute analysis and persist outputs."""
        stats = compute_stats(self.data_root, split_file=self.split_file)
        markdown = render_markdown(stats)

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_md.parent.mkdir(parents=True, exist_ok=True)

        self.output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        self.output_md.write_text(markdown, encoding="utf-8")

        LOGGER.info("Wrote JSON stats: %s", self.output_json)
        LOGGER.info("Wrote Markdown report: %s", self.output_md)


def main() -> None:
    """Entrypoint for CLI execution."""
    configure_logging()
    app = DatasetAnalysisApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
