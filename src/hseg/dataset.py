"""Dataset and split utilities for Pascal-Part subset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .hierarchy import fine_to_level0, fine_to_level1, validate_mask_ids
from .transforms import SegmentationTransforms


@dataclass(frozen=True)
class SampleRecord:
    stem: str
    image_path: Path
    mask_path: Path


def discover_records(
    data_root: str | Path,
    image_dir: str = "JPEGImages",
    mask_dir: str = "gt_masks",
) -> list[SampleRecord]:
    """Finds aligned image/mask pairs by file stem."""
    root = Path(data_root)
    image_root = root / image_dir
    mask_root = root / mask_dir

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")

    records: list[SampleRecord] = []
    for mask_path in sorted(mask_root.glob("*.npy")):
        stem = mask_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = image_root / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        records.append(SampleRecord(stem=stem, image_path=image_path, mask_path=mask_path))

    if not records:
        raise RuntimeError(
            f"No image/mask pairs found in {image_root} and {mask_root}. "
            "Check dataset extraction paths."
        )
    return records


def create_splits(
    stems: Iterable[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Creates deterministic train/val/test splits from sample stems."""
    stems = sorted(set(stems))
    if not stems:
        raise ValueError("No stems provided for split creation")

    rng = random.Random(seed)
    rng.shuffle(stems)

    n_total = len(stems)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    # Keep at least one training sample.
    if n_test + n_val >= n_total:
        n_test = max(1, n_total // 5)
        n_val = max(1, n_total // 5)

    test = stems[:n_test]
    val = stems[n_test : n_test + n_val]
    train = stems[n_test + n_val :]

    if not train:
        raise ValueError("Split creation left no training samples")

    return {"train": train, "val": val, "test": test}


def load_ids(path: str | Path) -> list[str]:
    """Loads non-empty sample ids (stems), one per line."""
    id_path = Path(path)
    if not id_path.exists():
        raise FileNotFoundError(f"ID file not found: {id_path}")
    lines = [line.strip() for line in id_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def create_splits_from_id_files(
    data_root: str | Path,
    image_dir: str = "JPEGImages",
    mask_dir: str = "gt_masks",
    train_ids_file: str = "train_id.txt",
    val_ids_file: str = "val_id.txt",
    create_test_split: bool = False,
    test_ratio_from_train: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Creates dataset splits from predefined id files.

    This follows the provided Pascal-Part split files in the dataset root.
    """
    root = Path(data_root)
    records = discover_records(root, image_dir=image_dir, mask_dir=mask_dir)
    available_stems = {record.stem for record in records}

    train_ids = sorted(set(load_ids(root / train_ids_file)))
    val_ids = sorted(set(load_ids(root / val_ids_file)))

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(
            f"train/val id files overlap for {len(overlap)} stems. "
            f"Examples: {overlap[:5]}"
        )

    referenced_stems = set(train_ids) | set(val_ids)
    missing_in_dataset = sorted(referenced_stems - available_stems)
    if missing_in_dataset:
        raise ValueError(
            f"ID files reference {len(missing_in_dataset)} stems missing in dataset. "
            f"Examples: {missing_in_dataset[:5]}"
        )

    unassigned = sorted(available_stems - referenced_stems)
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} dataset stems are not assigned by id files. "
            f"Examples: {unassigned[:5]}"
        )

    if not train_ids or not val_ids:
        raise ValueError("ID-based splits are empty; check train_id.txt and val_id.txt")

    if not create_test_split:
        return {"train": train_ids, "val": val_ids}

    rng = random.Random(seed)
    shuffled_train = list(train_ids)
    rng.shuffle(shuffled_train)

    n_test = max(1, int(len(shuffled_train) * float(test_ratio_from_train)))
    n_test = min(n_test, len(shuffled_train) - 1)
    if n_test <= 0:
        raise ValueError(
            "Unable to create test split from train ids; "
            f"train ids count={len(shuffled_train)}, requested ratio={test_ratio_from_train}"
        )

    test_ids = sorted(shuffled_train[:n_test])
    final_train_ids = sorted(shuffled_train[n_test:])
    if not final_train_ids:
        raise ValueError("Creating test split left no training samples")

    return {"train": final_train_ids, "val": val_ids, "test": test_ids}


def save_splits(split_path: str | Path, splits: dict[str, list[str]]) -> None:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as fp:
        json.dump(splits, fp, indent=2)


def load_splits(split_path: str | Path) -> dict[str, list[str]]:
    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Split file must contain a non-empty JSON object: {split_path}")

    splits: dict[str, list[str]] = {}
    seen_stems: dict[str, str] = {}
    for split_name, stems_raw in payload.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError(f"Invalid split name in {split_path}: {split_name!r}")
        if not isinstance(stems_raw, list):
            raise ValueError(f"Split '{split_name}' must be a list in {split_path}")

        split_stems: list[str] = []
        split_seen: set[str] = set()
        for item in stems_raw:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"Split '{split_name}' contains non-string/empty stem in {split_path}: {item!r}"
                )
            if item in split_seen:
                raise ValueError(f"Split '{split_name}' contains duplicate stem '{item}' in {split_path}")
            split_seen.add(item)
            if item in seen_stems:
                raise ValueError(
                    f"Stem '{item}' appears in multiple splits: "
                    f"'{seen_stems[item]}' and '{split_name}' in {split_path}"
                )
            seen_stems[item] = split_name
            split_stems.append(item)
        splits[split_name] = split_stems

    return splits


class PascalPartDataset(Dataset):
    """Pascal-Part subset dataset for hierarchical segmentation."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        split_file: str | Path,
        image_dir: str = "JPEGImages",
        mask_dir: str = "gt_masks",
        image_size: tuple[int, int] = (384, 384),
        train: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        class_aware_crop_cfg: Mapping[str, object] | None = None,
        augmentation_cfg: Mapping[str, object] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.records = discover_records(self.data_root, image_dir=image_dir, mask_dir=mask_dir)

        splits = load_splits(split_file)
        if split not in splits:
            raise KeyError(f"Split '{split}' is missing in {split_file}")
        split_stems = splits[split]

        records_by_stem = {record.stem: record for record in self.records}
        missing_split_stems = sorted(stem for stem in split_stems if stem not in records_by_stem)
        if missing_split_stems:
            raise RuntimeError(
                f"Split '{split}' references {len(missing_split_stems)} stems missing on disk. "
                f"Examples: {missing_split_stems[:5]}"
            )

        self.records = [records_by_stem[stem] for stem in split_stems]
        if not self.records:
            raise RuntimeError(f"No samples found for split '{split}'")

        self.transforms = SegmentationTransforms(
            image_size=image_size,
            train=train,
            mean=mean,
            std=std,
            class_aware_crop_cfg=class_aware_crop_cfg,
            augmentation_cfg=augmentation_cfg,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[idx]

        image = Image.open(rec.image_path).convert("RGB")
        mask = np.load(rec.mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask at {rec.mask_path}, got shape {mask.shape}")
        if image.width != int(mask.shape[1]) or image.height != int(mask.shape[0]):
            raise ValueError(
                f"Image/mask size mismatch for '{rec.stem}': "
                f"image={(image.height, image.width)}, mask={mask.shape}"
            )
        validate_mask_ids(mask)

        image_tensor, mask_l2 = self.transforms(image, mask)
        mask_l0 = fine_to_level0(mask_l2)
        mask_l1 = fine_to_level1(mask_l2)

        return {
            "image": image_tensor,
            "mask_l0": mask_l0,
            "mask_l1": mask_l1,
            "mask_l2": mask_l2,
            "id": rec.stem,
        }
