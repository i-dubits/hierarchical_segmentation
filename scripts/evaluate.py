#!/usr/bin/env python3
"""Evaluate hierarchical segmentation checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hseg.dataset import (  # noqa: E402
    PascalPartDataset,
    create_splits,
    create_splits_from_id_files,
    discover_records,
    load_splits,
    save_splits,
)
from hseg.model import build_model  # noqa: E402
from hseg.trainer import run_epoch  # noqa: E402
from hseg.utils import load_yaml, parse_bool, select_device  # noqa: E402

LOGGER = logging.getLogger(__name__)

CRITICAL_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("model",),
    ("data", "image_size"),
    ("data", "mean"),
    ("data", "std"),
    ("data", "image_dir"),
    ("data", "mask_dir"),
    ("training", "loss_weights"),
    ("training", "consistency"),
    ("training", "class_weights_l2"),
    ("training", "dice"),
    ("training", "lovasz"),
    ("training", "tversky"),
)


def configure_logging() -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


@dataclass(frozen=True)
class EvalCliArgs:
    """Command line arguments for evaluation."""

    config: str | None
    checkpoint: str
    split: str
    data_root: str | None
    output_json: str | None
    allow_generate_split: bool
    allow_config_mismatch: bool


def parse_args() -> EvalCliArgs:
    """Parse command line arguments.

    Returns:
        EvalCliArgs: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="Evaluate hierarchical segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional external config path. "
            "If omitted, config embedded in checkpoint is used."
        ),
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--allow-generate-split",
        action="store_true",
        help="Allow generating split file when missing (disabled by default for strict evaluation).",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow external config values to differ from checkpoint embedded config.",
    )
    ns = parser.parse_args()

    return EvalCliArgs(
        config=ns.config,
        checkpoint=ns.checkpoint,
        split=ns.split,
        data_root=ns.data_root,
        output_json=ns.output_json,
        allow_generate_split=bool(ns.allow_generate_split),
        allow_config_mismatch=bool(ns.allow_config_mismatch),
    )


def _as_mapping(payload: object, key_name: str) -> dict[str, Any]:
    """Validate and normalize a config value as mapping."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Config key '{key_name}' must be a mapping when provided.")


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


def make_eval_loader(dataset: PascalPartDataset, batch_size: int, num_workers: int) -> DataLoader:
    """Build evaluation dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


class EvaluationApp:
    """Run full evaluation workflow for one checkpoint/split."""

    def __init__(self, args: EvalCliArgs) -> None:
        """Initialize evaluation runtime fields."""
        self.args = args

        self.cfg: dict[str, Any] = {}
        self.seed = 42
        self.device = torch.device("cpu")
        self.checkpoint_path = Path(".")
        self.checkpoint: dict[str, Any] = {}
        self.data_cfg: dict[str, Any] = {}
        self.train_cfg: dict[str, Any] = {}
        self.model_cfg: dict[str, Any] = {}

        self.data_root = Path(".")
        self.split_file = Path(".")
        self.image_dir = "JPEGImages"
        self.mask_dir = "gt_masks"
        self.image_size = (384, 384)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.model: torch.nn.Module
        self.loader: DataLoader

        self.amp = True
        self.loss_weights: dict[str, float] = {}
        self.consistency = "mse"
        self.dice_cfg: dict[str, Any] = {}
        self.lovasz_cfg: dict[str, Any] = {}
        self.tversky_cfg: dict[str, Any] = {}
        self.class_weights_l2: torch.Tensor | None = None

    def run(self) -> None:
        """Execute evaluation and save optional JSON output."""
        self._prepare()
        metrics = self._evaluate()

        LOGGER.info("%s", json.dumps(metrics, indent=2))
        self._save_metrics(metrics)

    def _prepare(self) -> None:
        """Prepare config, split, loader, model, and loss settings."""
        self.checkpoint_path = Path(self.args.checkpoint).resolve()
        self.checkpoint = self._load_checkpoint()
        self.cfg = self._resolve_config(self.checkpoint)

        self.seed = int(self.cfg.get("seed", 42))
        self.device = select_device(str(self.cfg.get("device", "cuda")))
        LOGGER.info("Using device: %s", self.device)

        self.data_cfg = _as_mapping(self.cfg.get("data"), "data")
        self.train_cfg = _as_mapping(self.cfg.get("training"), "training")
        self.model_cfg = _as_mapping(self.cfg.get("model"), "model")

        self.data_root = Path(self.args.data_root or self.data_cfg["root"]).resolve()
        self.image_dir = str(self.data_cfg.get("image_dir", "JPEGImages"))
        self.mask_dir = str(self.data_cfg.get("mask_dir", "gt_masks"))
        self.image_size = tuple(self.data_cfg.get("image_size", [384, 384]))
        self.mean = tuple(self.data_cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = tuple(self.data_cfg.get("std", [0.229, 0.224, 0.225]))

        self.split_file = self._prepare_split_file()
        self._validate_split(self.args.split)

        self.loader = self._build_loader(self.args.split)
        self.model = self._load_model()

        self.amp = parse_bool(self.train_cfg.get("amp", True), "training.amp")
        self.loss_weights = dict(
            self.train_cfg.get("loss_weights", {"l2": 1.0, "l1": 0.5, "l0": 0.25, "consistency": 0.2})
        )
        self.consistency = str(self.train_cfg.get("consistency", "mse"))
        dice_cfg_raw = _as_mapping(self.train_cfg.get("dice", {}), "training.dice")
        self.dice_cfg = {
            "enabled": parse_bool(dice_cfg_raw.get("enabled", False), "training.dice.enabled"),
            "weight": float(dice_cfg_raw.get("weight", 0.0)),
            "smooth": float(dice_cfg_raw.get("smooth", 1.0)),
            "eps": float(dice_cfg_raw.get("eps", 1e-7)),
            "exclude_background": parse_bool(
                dice_cfg_raw.get("exclude_background", True),
                "training.dice.exclude_background",
            ),
        }
        lovasz_cfg_raw = _as_mapping(self.train_cfg.get("lovasz", {}), "training.lovasz")
        lovasz_ignore_index_raw = lovasz_cfg_raw.get("ignore_index", None)
        lovasz_ignore_index: int | None
        if lovasz_ignore_index_raw is None:
            lovasz_ignore_index = None
        else:
            lovasz_ignore_index = int(lovasz_ignore_index_raw)
        self.lovasz_cfg = {
            "enabled": parse_bool(lovasz_cfg_raw.get("enabled", False), "training.lovasz.enabled"),
            "weight": float(lovasz_cfg_raw.get("weight", 0.0)),
            "per_image": parse_bool(lovasz_cfg_raw.get("per_image", True), "training.lovasz.per_image"),
            "ignore_index": lovasz_ignore_index,
            "exclude_background": parse_bool(
                lovasz_cfg_raw.get("exclude_background", True),
                "training.lovasz.exclude_background",
            ),
            "present_only": parse_bool(
                lovasz_cfg_raw.get("present_only", True),
                "training.lovasz.present_only",
            ),
        }
        tversky_cfg_raw = _as_mapping(self.train_cfg.get("tversky", {}), "training.tversky")
        self.tversky_cfg = {
            "enabled": parse_bool(tversky_cfg_raw.get("enabled", False), "training.tversky.enabled"),
            "weight": float(tversky_cfg_raw.get("weight", 0.0)),
            "alpha": float(tversky_cfg_raw.get("alpha", 0.5)),
            "beta": float(tversky_cfg_raw.get("beta", 0.5)),
            "smooth": float(tversky_cfg_raw.get("smooth", 1.0)),
            "eps": float(tversky_cfg_raw.get("eps", 1e-7)),
            "focal_gamma": float(tversky_cfg_raw.get("focal_gamma", 1.0)),
            "exclude_background": parse_bool(
                tversky_cfg_raw.get("exclude_background", True),
                "training.tversky.exclude_background",
            ),
        }
        if (
            self.dice_cfg["enabled"]
            and self.dice_cfg["weight"] > 0.0
            and self.tversky_cfg["enabled"]
            and self.tversky_cfg["weight"] > 0.0
        ):
            raise ValueError("Config enables both Dice and Tversky for L2; enable only one.")

        class_weights = self.train_cfg.get("class_weights_l2")
        self.class_weights_l2 = None
        if class_weights is not None:
            self.class_weights_l2 = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

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
            checkpoint_cfg = _as_mapping(checkpoint_cfg_raw, "checkpoint.config")

        if self.args.config is None:
            if checkpoint_cfg is None:
                raise ValueError(
                    "Checkpoint does not contain embedded config. "
                    "Provide --config for evaluation."
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

    def _prepare_split_file(self) -> Path:
        """Create split file if needed and return its path."""
        split_file = self.data_root / self.data_cfg.get("split_file", "splits.json")
        if split_file.exists():
            LOGGER.info("Using existing split file: %s", split_file)
            return split_file

        if not self.args.allow_generate_split:
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Create/reuse training split first, or rerun with --allow-generate-split."
            )

        train_ids_file = self.data_root / "train_id.txt"
        val_ids_file = self.data_root / "val_id.txt"
        LOGGER.warning(
            "Split file missing; generating a new split for evaluation because --allow-generate-split is enabled."
        )
        if train_ids_file.exists() and val_ids_file.exists():
            id_based_test_ratio = float(self.data_cfg.get("id_based_test_ratio", 0.0))
            splits = create_splits_from_id_files(
                data_root=self.data_root,
                image_dir=self.image_dir,
                mask_dir=self.mask_dir,
                create_test_split=id_based_test_ratio > 0.0,
                test_ratio_from_train=id_based_test_ratio,
                seed=self.seed,
            )
            LOGGER.info(
                "Created split file from predefined IDs: %s, %s",
                train_ids_file.name,
                val_ids_file.name,
            )
        else:
            records = discover_records(self.data_root, image_dir=self.image_dir, mask_dir=self.mask_dir)
            splits = create_splits(
                stems=[record.stem for record in records],
                val_ratio=float(self.data_cfg.get("val_ratio", 0.1)),
                test_ratio=float(self.data_cfg.get("test_ratio", 0.1)),
                seed=self.seed,
            )
            LOGGER.info("Created generated deterministic train/val/test split for evaluation")

        save_splits(split_file, splits)
        LOGGER.info("Saved split file: %s", split_file)
        return split_file

    def _validate_split(self, split_name: str) -> None:
        """Validate requested split exists."""
        split_map = load_splits(self.split_file)
        if split_name in split_map:
            return

        raise KeyError(
            f"Split '{split_name}' is missing in {self.split_file}. "
            f"Available splits: {sorted(split_map.keys())}"
        )

    def _build_loader(self, split_name: str) -> DataLoader:
        """Build dataloader for one split."""
        dataset = PascalPartDataset(
            data_root=self.data_root,
            split=split_name,
            split_file=self.split_file,
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            image_size=self.image_size,
            train=False,
            mean=self.mean,
            std=self.std,
        )
        return make_eval_loader(
            dataset=dataset,
            batch_size=int(self.train_cfg.get("batch_size", 8)),
            num_workers=int(self.train_cfg.get("num_workers", 4)),
        )

    def _load_model(self) -> torch.nn.Module:
        """Build model and load checkpoint weights."""
        model = build_model(self.model_cfg).to(self.device)
        model.load_state_dict(self.checkpoint["model_state"])
        model.eval()
        return model

    def _evaluate(self) -> dict[str, Any]:
        """Run evaluation loop and return metrics."""
        with torch.no_grad():
            return run_epoch(
                model=self.model,
                loader=self.loader,
                device=self.device,
                optimizer=None,
                scaler=None,
                amp=self.amp,
                loss_weights=self.loss_weights,
                consistency=self.consistency,
                class_weights_l2=self.class_weights_l2,
                dice_cfg=self.dice_cfg,
                lovasz_cfg=self.lovasz_cfg,
                tversky_cfg=self.tversky_cfg,
                desc=f"eval-{self.args.split}",
            )

    def _save_metrics(self, metrics: dict[str, Any]) -> None:
        """Write metrics to JSON file when output path is provided."""
        if self.args.output_json is None:
            return

        output_path = Path(self.args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        LOGGER.info("Saved metrics to: %s", output_path)


def main() -> None:
    """Entrypoint for CLI execution."""
    configure_logging()
    app = EvaluationApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
