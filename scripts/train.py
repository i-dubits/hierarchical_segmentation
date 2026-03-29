#!/usr/bin/env python3
"""Train hierarchical semantic segmentation model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from contextlib import nullcontext
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
from hseg.utils import ensure_dir, load_yaml, parse_bool, select_device, set_seed  # noqa: E402

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


@dataclass(frozen=True)
class TrainCliArgs:
    """Command line arguments for the training script."""

    config: str
    data_root: str | None
    output_dir: str
    resume_from: str | None
    mlflow: bool
    mlflow_tracking_uri: str | None
    mlflow_experiment: str | None
    mlflow_run_name: str | None


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping configuration values."""

    enabled: bool
    patience: int
    min_delta: float
    metric: str
    mode: str


@dataclass
class EarlyStoppingState:
    """Mutable early stopping runtime state."""

    best_value: float
    epochs_no_improve: int


@dataclass(frozen=True)
class MlflowRuntimeConfig:
    """MLflow runtime configuration values."""

    enabled: bool
    tracking_uri: str | None
    experiment: str
    run_name: str
    log_artifacts: bool


def parse_args() -> TrainCliArgs:
    """Parse command line arguments.

    Returns:
        TrainCliArgs: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="Train hierarchical segmentation model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume from.",
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default=None)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    ns = parser.parse_args()

    return TrainCliArgs(
        config=ns.config,
        data_root=ns.data_root,
        output_dir=ns.output_dir,
        resume_from=ns.resume_from,
        mlflow=bool(ns.mlflow),
        mlflow_tracking_uri=ns.mlflow_tracking_uri,
        mlflow_experiment=ns.mlflow_experiment,
        mlflow_run_name=ns.mlflow_run_name,
    )


def make_loader(dataset: PascalPartDataset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    """Build a DataLoader for a dataset split.

    Args:
        dataset (PascalPartDataset): Input dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of loader workers.
        train (bool): Whether to enable shuffle.

    Returns:
        DataLoader: Configured data loader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    train_cfg: Mapping[str, Any],
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    """Build the LR scheduler from training config.

    Returns:
        tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
            Scheduler instance and human-readable scheduler name.
    """
    scheduler_cfg_raw = train_cfg.get("scheduler", {"name": "cosine"})
    if isinstance(scheduler_cfg_raw, str):
        scheduler_name = scheduler_cfg_raw.lower()
        scheduler_cfg: dict[str, object] = {}
    else:
        scheduler_cfg = dict(scheduler_cfg_raw)
        scheduler_name = str(scheduler_cfg.get("name", "cosine")).lower()

    if scheduler_name in {"none", "off"}:
        return None, "none"

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Use 'cosine' or 'none'.")

    eta_min = float(scheduler_cfg.get("eta_min", 0.0))
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if warmup_epochs >= epochs:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be smaller than epochs ({epochs})")

    if warmup_epochs == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=eta_min,
        )
        return scheduler, f"cosine(eta_min={eta_min})"

    warmup_start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.1))
    if not (0.0 < warmup_start_factor <= 1.0):
        raise ValueError(
            f"warmup_start_factor must be in (0, 1], got {warmup_start_factor}"
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler, (
        f"cosine_warmup(warmup_epochs={warmup_epochs}, "
        f"warmup_start_factor={warmup_start_factor}, eta_min={eta_min})"
    )


def _flatten_metrics(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    """Flatten nested metric dictionaries into scalar metric keys."""
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                flat[name] = number
    return flat


def _flatten_params(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested config mapping into string parameter values."""
    flat: dict[str, str] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_params(value, name))
            continue
        if isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, separators=(",", ":"))
            continue
        if isinstance(value, str) and _is_path_like_param_name(name):
            flat[name] = _sanitize_path_for_logging(value)
            continue
        flat[name] = str(value)
    return flat


def _as_mapping(payload: object, key_name: str) -> dict[str, Any]:
    """Validate and normalize a config value as dictionary."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Config key '{key_name}' must be a mapping when provided.")


def _extract_metric(payload: object, metric_name: str) -> float | None:
    """Extract a finite numeric metric from a mapping payload."""
    if not isinstance(payload, dict):
        return None
    value = payload.get(metric_name)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _infer_early_stopping_mode(metric_name: str) -> str:
    """Infer early-stopping optimization direction from metric name."""
    normalized = metric_name.strip().lower()
    if normalized.endswith("loss"):
        return "min"
    return "max"


def _configs_match(a: object, b: object) -> bool:
    """Return True when two config payloads are equal mappings."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return a == b


def _is_path_like_param_name(param_name: str) -> bool:
    """Heuristic for MLflow param keys that represent filesystem paths/URIs."""
    leaf = param_name.rsplit(".", 1)[-1].lower()
    if leaf in {
        "path",
        "root",
        "dir",
        "file",
        "uri",
        "from",
        "data_root",
        "split_file",
        "output_dir",
        "config_path",
        "resume_from",
        "checkpoint",
        "checkpoint_path",
        "tracking_uri",
        "artifact_uri",
        "artifact_location",
    }:
        return True
    return leaf.endswith(("_path", "_root", "_dir", "_file", "_uri", "_checkpoint", "_from"))


def _sanitize_path_for_logging(path_value: str | Path | None, workspace_root: Path = ROOT) -> str:
    """Return a repo-relative path string for logs/MLflow params.

    Absolute paths under the workspace root are converted to relative paths.
    Absolute paths outside the workspace are redacted to avoid leaking host paths.
    """
    if path_value is None:
        return ""

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


def setup_mlflow(runtime_cfg: MlflowRuntimeConfig) -> tuple[Any | None, Any]:
    """Initialize MLflow run context.

    Returns:
        tuple[Any | None, Any]:
            Imported MLflow module (or None) and a context manager.
    """
    if not runtime_cfg.enabled:
        return None, nullcontext()

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requested, but 'mlflow' is not installed. "
            "Install it with: python -m pip install mlflow"
        ) from exc

    if runtime_cfg.tracking_uri:
        mlflow.set_tracking_uri(runtime_cfg.tracking_uri)
    mlflow.set_experiment(runtime_cfg.experiment)
    return mlflow, mlflow.start_run(run_name=runtime_cfg.run_name)


class TrainingApp:
    """Orchestrate model training, evaluation, checkpointing, and logging."""

    def __init__(self, args: TrainCliArgs) -> None:
        """Initialize training runtime containers.

        Args:
            args (TrainCliArgs): Parsed CLI arguments.
        """
        self.args = args

        self.cfg: dict[str, Any] = {}
        self.seed = 42
        self.device = torch.device("cpu")
        self.data_cfg: dict[str, Any] = {}
        self.train_cfg: dict[str, Any] = {}
        self.model_cfg: dict[str, Any] = {}
        self.mlflow_cfg: dict[str, Any] = {}

        self.data_root = Path(".")
        self.output_dir = Path(".")
        self.split_file = Path(".")
        self.final_eval_split = "val"

        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader
        self.batch_size = 8
        self.num_workers = 4

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scheduler_name = "none"
        self.amp = True
        self.scaler: torch.cuda.amp.GradScaler | None = None

        self.class_weights_l2: torch.Tensor | None = None
        self.loss_weights: dict[str, float] = {}
        self.consistency = "mse"
        self.dice_cfg: dict[str, Any] = {}
        self.lovasz_cfg: dict[str, Any] = {}
        self.tversky_cfg: dict[str, Any] = {}
        self.grad_accum_steps = 1
        self.epochs = 40
        self.checkpoint_metric = "miou_l2"

        self.best_value = float("-inf")
        self.early_cfg = EarlyStoppingConfig(
            enabled=True,
            patience=5,
            min_delta=0.0,
            metric="loss",
            mode="min",
        )
        self.early_state = EarlyStoppingState(best_value=float("inf"), epochs_no_improve=0)
        self.early_stopped = False
        self.stopped_epoch: int | None = None

        self.start_epoch = 0
        self.best_ckpt_path = Path(".")
        self.history_path = Path(".")
        self.history: list[dict[str, Any]] = []

        self.mlflow_runtime = MlflowRuntimeConfig(
            enabled=False,
            tracking_uri=None,
            experiment="hseg",
            run_name="hseg",
            log_artifacts=True,
        )
        self.mlflow_module: Any | None = None
        self.mlflow_context: Any = nullcontext()

    def run(self) -> None:
        """Run full training workflow."""
        self._prepare()

        with self.mlflow_context:
            self._log_mlflow_start()
            self._run_epochs()
            summary = self._run_final_evaluation()
            self._log_mlflow_end(summary)

        LOGGER.info("Training complete")
        LOGGER.info("%s", json.dumps(summary, indent=2))

    def _prepare(self) -> None:
        """Prepare configuration, data, runtime state, and MLflow."""
        self._load_config()
        self._prepare_data_pipeline()
        self._prepare_training_components()
        self._prepare_runtime_state()
        self._prepare_mlflow()

    def _load_config(self) -> None:
        """Load YAML config and initialize core runtime fields."""
        self.cfg = load_yaml(self.args.config)

        self.seed = int(self.cfg.get("seed", 42))
        set_seed(self.seed)

        self.device = select_device(str(self.cfg.get("device", "cuda")))
        LOGGER.info("Using device: %s", self.device)
        self.data_cfg = _as_mapping(self.cfg.get("data"), "data")
        self.train_cfg = _as_mapping(self.cfg.get("training"), "training")
        self.model_cfg = _as_mapping(self.cfg.get("model"), "model")
        self.mlflow_cfg = _as_mapping(self.cfg.get("mlflow", {}), "mlflow")

        self.data_root = Path(self.args.data_root or self.data_cfg["root"]).resolve()
        self.output_dir = ensure_dir(self.args.output_dir)
        self.best_ckpt_path = self.output_dir / "best.pt"
        self.history_path = self.output_dir / "history.json"

    def _prepare_data_pipeline(self) -> None:
        """Prepare splits, datasets, and dataloaders."""
        self.split_file = self._prepare_split_file()

        image_dir = str(self.data_cfg.get("image_dir", "JPEGImages"))
        mask_dir = str(self.data_cfg.get("mask_dir", "gt_masks"))
        image_size = tuple(self.data_cfg.get("image_size", [384, 384]))
        mean = tuple(self.data_cfg.get("mean", [0.485, 0.456, 0.406]))
        std = tuple(self.data_cfg.get("std", [0.229, 0.224, 0.225]))
        class_aware_crop_cfg_raw = self.data_cfg.get("class_aware_crop")
        class_aware_crop_cfg: dict[str, object] | None = None
        if class_aware_crop_cfg_raw is not None:
            class_aware_crop_cfg = _as_mapping(
                class_aware_crop_cfg_raw,
                "data.class_aware_crop",
            )
        augmentation_cfg_raw = self.data_cfg.get("augmentations")
        augmentation_cfg: dict[str, object] | None = None
        if augmentation_cfg_raw is not None:
            augmentation_cfg = _as_mapping(
                augmentation_cfg_raw,
                "data.augmentations",
            )

        train_ds = PascalPartDataset(
            data_root=self.data_root,
            split="train",
            split_file=self.split_file,
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            train=True,
            mean=mean,
            std=std,
            class_aware_crop_cfg=class_aware_crop_cfg,
            augmentation_cfg=augmentation_cfg,
        )
        val_ds = PascalPartDataset(
            data_root=self.data_root,
            split="val",
            split_file=self.split_file,
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            train=False,
            mean=mean,
            std=std,
        )

        split_keys = load_splits(self.split_file).keys()
        self.final_eval_split = "test" if "test" in split_keys else "val"
        if self.final_eval_split != "test":
            LOGGER.info("Split 'test' not found. Using 'val' for final evaluation.")

        test_ds = PascalPartDataset(
            data_root=self.data_root,
            split=self.final_eval_split,
            split_file=self.split_file,
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            train=False,
            mean=mean,
            std=std,
        )

        self.batch_size = int(self.train_cfg.get("batch_size", 8))
        self.num_workers = int(self.train_cfg.get("num_workers", 4))
        self.train_loader = make_loader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers, train=True)
        self.val_loader = make_loader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers, train=False)
        self.test_loader = make_loader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, train=False)

    def _prepare_training_components(self) -> None:
        """Build model, optimizer, scheduler, AMP state, and loss settings."""
        self.model = build_model(self.model_cfg).to(self.device)

        lr = float(self.train_cfg.get("lr", 3e-4))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.epochs = int(self.train_cfg.get("epochs", 40))
        self.scheduler, self.scheduler_name = build_scheduler(
            optimizer=self.optimizer,
            epochs=self.epochs,
            train_cfg=self.train_cfg,
        )
        LOGGER.info("Scheduler: %s", self.scheduler_name)

        self.amp = parse_bool(self.train_cfg.get("amp", True), "training.amp")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.device.type == "cuda")

        class_weights = self.train_cfg.get("class_weights_l2")
        self.class_weights_l2 = None
        if class_weights is not None:
            self.class_weights_l2 = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        self.grad_accum_steps = int(self.train_cfg.get("grad_accum_steps", 1))
        if self.grad_accum_steps < 1:
            raise ValueError(f"training.grad_accum_steps must be >= 1, got {self.grad_accum_steps}")

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
        if self.dice_cfg["weight"] < 0.0:
            raise ValueError(f"training.dice.weight must be >= 0, got {self.dice_cfg['weight']}")
        if self.dice_cfg["smooth"] < 0.0:
            raise ValueError(f"training.dice.smooth must be >= 0, got {self.dice_cfg['smooth']}")
        if self.dice_cfg["eps"] <= 0.0:
            raise ValueError(f"training.dice.eps must be > 0, got {self.dice_cfg['eps']}")

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
        if self.lovasz_cfg["weight"] < 0.0:
            raise ValueError(f"training.lovasz.weight must be >= 0, got {self.lovasz_cfg['weight']}")

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
        if self.tversky_cfg["weight"] < 0.0:
            raise ValueError(f"training.tversky.weight must be >= 0, got {self.tversky_cfg['weight']}")
        if self.tversky_cfg["alpha"] < 0.0 or self.tversky_cfg["beta"] < 0.0:
            raise ValueError(
                "training.tversky.alpha and training.tversky.beta must both be >= 0 "
                f"(got alpha={self.tversky_cfg['alpha']}, beta={self.tversky_cfg['beta']})"
            )
        if (self.tversky_cfg["alpha"] + self.tversky_cfg["beta"]) <= 0.0:
            raise ValueError("training.tversky.alpha + training.tversky.beta must be > 0")
        if self.tversky_cfg["smooth"] < 0.0:
            raise ValueError(f"training.tversky.smooth must be >= 0, got {self.tversky_cfg['smooth']}")
        if self.tversky_cfg["eps"] <= 0.0:
            raise ValueError(f"training.tversky.eps must be > 0, got {self.tversky_cfg['eps']}")
        if self.tversky_cfg["focal_gamma"] <= 0.0:
            raise ValueError(
                f"training.tversky.focal_gamma must be > 0, got {self.tversky_cfg['focal_gamma']}"
            )
        if (
            self.dice_cfg["enabled"]
            and self.dice_cfg["weight"] > 0.0
            and self.tversky_cfg["enabled"]
            and self.tversky_cfg["weight"] > 0.0
        ):
            raise ValueError("Enable either training.dice or training.tversky, not both.")
        self.checkpoint_metric = str(self.train_cfg.get("checkpoint_metric", "miou_l2"))

    def _prepare_runtime_state(self) -> None:
        """Restore early-stopping and resume state from artifacts."""
        self.early_cfg = self._build_early_stopping_config()
        self.early_state = self._make_initial_early_stopping_state()
        self.history = self._load_history()
        self.early_state = self._restore_early_stopping_state(self.history)

        if self.args.resume_from:
            resume_result = self._restore_resume_state(self.args.resume_from)
            self.start_epoch = resume_result["start_epoch"]
            self.best_value = resume_result["best_value"]

            resumed_early_state = resume_result["early_state"]
            if resumed_early_state is not None:
                self.early_state = resumed_early_state
            else:
                self.early_state = self._restore_early_stopping_state(self.history)

        LOGGER.info(
            "Early stopping: enabled=%s, metric=%s, mode=%s, patience=%s, min_delta=%s",
            self.early_cfg.enabled,
            self.early_cfg.metric,
            self.early_cfg.mode,
            self.early_cfg.patience,
            self.early_cfg.min_delta,
        )

    def _prepare_mlflow(self) -> None:
        """Prepare MLflow runtime and context manager."""
        self.mlflow_runtime = self._resolve_mlflow_runtime()
        self.mlflow_module, self.mlflow_context = setup_mlflow(self.mlflow_runtime)

    def _prepare_split_file(self) -> Path:
        """Create or reuse dataset split file.

        Returns:
            Path: Split file path.
        """
        split_file = self.data_root / self.data_cfg.get("split_file", "splits.json")
        image_dir = str(self.data_cfg.get("image_dir", "JPEGImages"))
        mask_dir = str(self.data_cfg.get("mask_dir", "gt_masks"))

        if split_file.exists():
            LOGGER.info("Using existing split file: %s", split_file)
            return split_file

        train_ids_file = self.data_root / "train_id.txt"
        val_ids_file = self.data_root / "val_id.txt"
        if train_ids_file.exists() and val_ids_file.exists():
            id_based_test_ratio = float(self.data_cfg.get("id_based_test_ratio", 0.0))
            splits = create_splits_from_id_files(
                data_root=self.data_root,
                image_dir=image_dir,
                mask_dir=mask_dir,
                create_test_split=id_based_test_ratio > 0.0,
                test_ratio_from_train=id_based_test_ratio,
                seed=self.seed,
            )
            LOGGER.info(
                "Using predefined dataset split files: %s, %s",
                train_ids_file.name,
                val_ids_file.name,
            )
        else:
            records = discover_records(self.data_root, image_dir=image_dir, mask_dir=mask_dir)
            splits = create_splits(
                stems=[record.stem for record in records],
                val_ratio=float(self.data_cfg.get("val_ratio", 0.1)),
                test_ratio=float(self.data_cfg.get("test_ratio", 0.1)),
                seed=self.seed,
            )
            LOGGER.info("Using generated deterministic train/val/test split")

        save_splits(split_file, splits)
        LOGGER.info("Created split file: %s", split_file)
        return split_file

    def _build_early_stopping_config(self) -> EarlyStoppingConfig:
        """Build validated early-stopping configuration."""
        early_cfg_raw = _as_mapping(self.train_cfg.get("early_stopping", {}), "training.early_stopping")
        metric = str(early_cfg_raw.get("metric", "loss")).strip()
        if not metric:
            raise ValueError("training.early_stopping.metric cannot be empty")
        mode_raw = early_cfg_raw.get("mode")
        if mode_raw is None:
            mode = _infer_early_stopping_mode(metric)
        else:
            mode = str(mode_raw).strip().lower()
        config = EarlyStoppingConfig(
            enabled=parse_bool(early_cfg_raw.get("enabled", True), "training.early_stopping.enabled"),
            patience=int(early_cfg_raw.get("patience", 5)),
            min_delta=float(early_cfg_raw.get("min_delta", 0.0)),
            metric=metric,
            mode=mode,
        )
        if config.patience < 1:
            raise ValueError(f"training.early_stopping.patience must be >= 1, got {config.patience}")
        if config.min_delta < 0:
            raise ValueError(f"training.early_stopping.min_delta must be >= 0, got {config.min_delta}")
        if config.mode not in {"min", "max"}:
            raise ValueError(
                f"training.early_stopping.mode must be 'min' or 'max', got {config.mode!r}"
            )
        return config

    def _make_initial_early_stopping_state(self) -> EarlyStoppingState:
        """Create empty early-stopping state for the configured optimization mode."""
        best_value = float("inf") if self.early_cfg.mode == "min" else float("-inf")
        return EarlyStoppingState(best_value=best_value, epochs_no_improve=0)

    def _is_early_stopping_improvement(self, current_value: float, best_value: float) -> bool:
        """Return whether a metric value improves early-stopping state."""
        if self.early_cfg.mode == "min":
            return current_value < best_value - self.early_cfg.min_delta
        return current_value > best_value + self.early_cfg.min_delta

    def _load_history(self) -> list[dict[str, Any]]:
        """Load history file when resume mode is enabled."""
        if self.args.resume_from:
            resume_parent = Path(self.args.resume_from).resolve().parent
            if resume_parent != self.output_dir.resolve():
                LOGGER.warning(
                    "Resume checkpoint parent (%s) differs from output_dir (%s). "
                    "Ignoring local history.json for safety.",
                    resume_parent,
                    self.output_dir.resolve(),
                )
                return []
            if not self.history_path.exists():
                return []
            with self.history_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, list):
                raise TypeError(f"history.json must contain a list: {self.history_path}")
            return [entry for entry in payload if isinstance(entry, dict)]

        if self.history_path.exists():
            LOGGER.info(
                "Existing history.json found but resume is disabled; "
                "starting a fresh training history."
            )
        return []

    def _restore_early_stopping_state(self, history: list[dict[str, Any]]) -> EarlyStoppingState:
        """Recompute early-stopping state from loaded history."""
        state = self._make_initial_early_stopping_state()
        for record in history:
            current_value = _extract_metric(record.get("val"), self.early_cfg.metric)
            if current_value is None:
                continue
            if self._is_early_stopping_improvement(current_value, state.best_value):
                state.best_value = current_value
                state.epochs_no_improve = 0
            else:
                state.epochs_no_improve += 1
        return state

    def _trim_history_to_epoch(self, history: list[dict[str, Any]], max_epoch: int) -> list[dict[str, Any]]:
        """Keep only history entries up to and including the resume epoch."""
        trimmed: list[dict[str, Any]] = []
        dropped = 0
        has_resume_epoch = False
        for record in history:
            epoch_raw = record.get("epoch")
            if not isinstance(epoch_raw, int):
                dropped += 1
                continue
            if epoch_raw == max_epoch:
                has_resume_epoch = True
            if epoch_raw <= max_epoch:
                trimmed.append(record)
            else:
                dropped += 1
        if history and max_epoch > 0 and not has_resume_epoch:
            LOGGER.warning(
                "history.json does not contain resume epoch=%s; ignoring history to avoid stale state.",
                max_epoch,
            )
            return []
        if dropped > 0:
            LOGGER.warning(
                "Dropped %s stale history entries newer than resume epoch=%s.",
                dropped,
                max_epoch,
            )
        return trimmed

    def _extract_early_stopping_from_checkpoint(self, state: Mapping[str, Any]) -> EarlyStoppingState | None:
        """Extract early-stopping state from checkpoint payload."""
        early_raw = state.get("early_stopping")
        if not isinstance(early_raw, dict):
            return None

        metric_name = early_raw.get("metric")
        mode_name = early_raw.get("mode")
        if isinstance(metric_name, str) and metric_name != self.early_cfg.metric:
            return None
        if isinstance(mode_name, str) and mode_name.lower() != self.early_cfg.mode:
            return None

        best_value = _extract_metric(early_raw, "best_value")
        epochs_no_improve = early_raw.get("epochs_no_improve")
        if best_value is not None and isinstance(epochs_no_improve, int) and epochs_no_improve >= 0:
            return EarlyStoppingState(
                best_value=float(best_value),
                epochs_no_improve=int(epochs_no_improve),
            )

        legacy_best_val_loss = _extract_metric(early_raw, "best_val_loss")
        legacy_epochs_no_improve = early_raw.get("epochs_no_val_loss_improve")
        if (
            self.early_cfg.metric == "loss"
            and self.early_cfg.mode == "min"
            and legacy_best_val_loss is not None
            and isinstance(legacy_epochs_no_improve, int)
            and legacy_epochs_no_improve >= 0
        ):
            return EarlyStoppingState(
                best_value=float(legacy_best_val_loss),
                epochs_no_improve=int(legacy_epochs_no_improve),
            )
        return None

    def _best_metric_from_history(self, history: list[dict[str, Any]]) -> float | None:
        """Extract best checkpoint metric from history entries."""
        history_values = [
            value
            for value in (_extract_metric(record.get("val"), self.checkpoint_metric) for record in history)
            if value is not None
        ]
        if not history_values:
            return None
        return max(history_values)

    def _load_best_metric_from_best_checkpoint(
        self,
        start_epoch: int,
        resume_state: Mapping[str, Any],
    ) -> float | None:
        """Load best metric from best checkpoint when lineage matches resume state."""
        if not self.best_ckpt_path.exists():
            return None

        best_state = torch.load(self.best_ckpt_path, map_location=self.device)
        best_epoch = best_state.get("epoch")
        if isinstance(best_epoch, int) and best_epoch > start_epoch:
            LOGGER.warning(
                "Ignoring best checkpoint at future epoch=%s while resume epoch=%s.",
                best_epoch,
                start_epoch,
            )
            return None

        if not _configs_match(best_state.get("config"), resume_state.get("config")):
            LOGGER.warning(
                "Ignoring best checkpoint due to config mismatch with resume checkpoint."
            )
            return None

        return _extract_metric(best_state.get("val_metrics", {}), self.checkpoint_metric)

    def _restore_resume_state(self, resume_from: str) -> dict[str, Any]:
        """Restore model/optimizer/scheduler state from checkpoint.

        Args:
            resume_from (str): Checkpoint path.

        Returns:
            dict[str, Any]: Resume start epoch, best metric value, and early-stop state.
        """
        resume_path = Path(resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        LOGGER.info("Resuming from checkpoint: %s", resume_path)

        resume_state = torch.load(resume_path, map_location=self.device)
        self.model.load_state_dict(resume_state["model_state"])

        optimizer_state = resume_state.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = resume_state.get("scheduler_state")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        start_epoch = int(resume_state.get("epoch", 0))
        if start_epoch < 0:
            raise ValueError(f"Invalid checkpoint epoch in {resume_path}: {start_epoch}")

        self.history = self._trim_history_to_epoch(self.history, start_epoch)

        best_value: float | None = self._load_best_metric_from_best_checkpoint(
            start_epoch=start_epoch,
            resume_state=resume_state,
        )
        if best_value is None:
            best_value = self._best_metric_from_history(self.history)
        if best_value is None:
            best_value = _extract_metric(resume_state.get("val_metrics", {}), self.checkpoint_metric)

        resolved_best = best_value if best_value is not None else float("-inf")
        early_state = self._extract_early_stopping_from_checkpoint(resume_state)

        if early_state is not None:
            LOGGER.info(
                "Restored early stopping from checkpoint: metric=%s, best_value=%.6f, epochs_no_improve=%s",
                self.early_cfg.metric,
                early_state.best_value,
                early_state.epochs_no_improve,
            )
        else:
            LOGGER.warning(
                "Checkpoint missing valid early-stopping state; reconstructing from trimmed history."
            )

        LOGGER.info(
            "Resume state: start_epoch=%s, best_%s=%.6f",
            start_epoch,
            self.checkpoint_metric,
            resolved_best,
        )
        return {
            "start_epoch": start_epoch,
            "best_value": resolved_best,
            "early_state": early_state,
        }

    def _resolve_mlflow_runtime(self) -> MlflowRuntimeConfig:
        """Build MLflow runtime settings from CLI and config."""
        return MlflowRuntimeConfig(
            enabled=(
                bool(self.args.mlflow)
                or parse_bool(self.mlflow_cfg.get("enabled", False), "mlflow.enabled")
            ),
            tracking_uri=(
                str(self.args.mlflow_tracking_uri)
                if self.args.mlflow_tracking_uri is not None
                else (
                    str(self.mlflow_cfg.get("tracking_uri"))
                    if self.mlflow_cfg.get("tracking_uri") is not None
                    else None
                )
            ),
            experiment=str(self.args.mlflow_experiment or self.mlflow_cfg.get("experiment_name") or "hseg"),
            run_name=str(self.args.mlflow_run_name or self.mlflow_cfg.get("run_name") or self.output_dir.name),
            log_artifacts=parse_bool(self.mlflow_cfg.get("log_artifacts", True), "mlflow.log_artifacts"),
        )

    def _log_mlflow_start(self) -> None:
        """Log static run metadata to MLflow at startup."""
        if self.mlflow_module is None:
            return

        self.mlflow_module.log_params(
            {
                "seed": self.seed,
                "device": self.device.type,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "grad_accum_steps": self.grad_accum_steps,
                "effective_batch_size": self.batch_size * self.grad_accum_steps,
                "lr": float(self.train_cfg.get("lr", 3e-4)),
                "weight_decay": float(self.train_cfg.get("weight_decay", 1e-4)),
                "amp": self.amp,
                "checkpoint_metric": self.checkpoint_metric,
                "scheduler": self.scheduler_name,
                "dice_enabled": bool(self.dice_cfg.get("enabled", False)),
                "dice_weight": float(self.dice_cfg.get("weight", 0.0)),
                "dice_smooth": float(self.dice_cfg.get("smooth", 1.0)),
                "dice_eps": float(self.dice_cfg.get("eps", 1e-7)),
                "dice_exclude_background": bool(self.dice_cfg.get("exclude_background", True)),
                "lovasz_enabled": bool(self.lovasz_cfg.get("enabled", False)),
                "lovasz_weight": float(self.lovasz_cfg.get("weight", 0.0)),
                "lovasz_per_image": bool(self.lovasz_cfg.get("per_image", True)),
                "lovasz_ignore_index": (
                    -1 if self.lovasz_cfg.get("ignore_index", None) is None else int(self.lovasz_cfg["ignore_index"])
                ),
                "lovasz_exclude_background": bool(self.lovasz_cfg.get("exclude_background", True)),
                "lovasz_present_only": bool(self.lovasz_cfg.get("present_only", True)),
                "tversky_enabled": bool(self.tversky_cfg.get("enabled", False)),
                "tversky_weight": float(self.tversky_cfg.get("weight", 0.0)),
                "tversky_alpha": float(self.tversky_cfg.get("alpha", 0.5)),
                "tversky_beta": float(self.tversky_cfg.get("beta", 0.5)),
                "tversky_smooth": float(self.tversky_cfg.get("smooth", 1.0)),
                "tversky_eps": float(self.tversky_cfg.get("eps", 1e-7)),
                "tversky_focal_gamma": float(self.tversky_cfg.get("focal_gamma", 1.0)),
                "tversky_exclude_background": bool(self.tversky_cfg.get("exclude_background", True)),
                "data_root": _sanitize_path_for_logging(self.data_root),
                "split_file": _sanitize_path_for_logging(self.split_file),
                "output_dir": _sanitize_path_for_logging(self.output_dir),
                "config_path": _sanitize_path_for_logging(self.args.config),
                "resume_from": _sanitize_path_for_logging(self.args.resume_from),
                "start_epoch": self.start_epoch,
                "early_stopping_enabled": self.early_cfg.enabled,
                "early_stopping_metric": self.early_cfg.metric,
                "early_stopping_mode": self.early_cfg.mode,
                "early_stopping_patience": self.early_cfg.patience,
                "early_stopping_min_delta": self.early_cfg.min_delta,
            }
        )
        self.mlflow_module.log_params(_flatten_params(self.cfg, prefix="cfg"))
        self.mlflow_module.set_tags(
            {
                "script": "scripts/train.py",
                "model_name": str(self.model_cfg.get("name", "baseline")),
                "resumed": str(bool(self.args.resume_from)).lower(),
            }
        )
        self.mlflow_module.log_artifact(str(Path(self.args.config).resolve()), artifact_path="config")

    def _run_epochs(self) -> None:
        """Execute the epoch training/validation loop."""
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            LOGGER.info("Epoch %s/%s", epoch, self.epochs)

            train_metrics = run_epoch(
                model=self.model,
                loader=self.train_loader,
                device=self.device,
                optimizer=self.optimizer,
                scaler=self.scaler,
                amp=self.amp,
                loss_weights=self.loss_weights,
                consistency=self.consistency,
                class_weights_l2=self.class_weights_l2,
                dice_cfg=self.dice_cfg,
                lovasz_cfg=self.lovasz_cfg,
                tversky_cfg=self.tversky_cfg,
                grad_accum_steps=self.grad_accum_steps,
                desc=f"train {epoch}",
            )

            with torch.no_grad():
                val_metrics = run_epoch(
                    model=self.model,
                    loader=self.val_loader,
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
                    desc=f"val {epoch}",
                )

            if self.scheduler is not None:
                self.scheduler.step()

            is_best = self._update_best_metric(val_metrics=val_metrics, epoch=epoch)
            self._update_early_stopping(val_metrics=val_metrics, epoch=epoch)
            self._save_epoch_artifacts(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )
            self._log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )

            LOGGER.info(
                "train_loss=%.4f val_loss=%.4f val_mIoU^0=%.4f val_mIoU^1=%.4f "
                "val_mIoU^2=%.4f best_es_%s=%.4f es_wait=%s/%s",
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["miou_l0"],
                val_metrics["miou_l1"],
                val_metrics["miou_l2"],
                self.early_cfg.metric,
                self.early_state.best_value,
                self.early_state.epochs_no_improve,
                self.early_cfg.patience,
            )

            if self._should_early_stop(epoch):
                break

    def _update_best_metric(self, val_metrics: Mapping[str, Any], epoch: int) -> bool:
        """Update tracked best validation metric.

        Returns:
            bool: Whether current epoch produced a new best checkpoint.
        """
        if self.checkpoint_metric not in val_metrics:
            raise KeyError(
                f"checkpoint_metric '{self.checkpoint_metric}' is missing in validation metrics. "
                f"Available keys: {sorted(val_metrics.keys())}"
            )

        current_metric = float(val_metrics[self.checkpoint_metric])
        if not math.isfinite(current_metric):
            raise ValueError(
                f"Validation metric '{self.checkpoint_metric}' is not finite at epoch {epoch}: "
                f"{current_metric}"
            )

        is_best = current_metric > self.best_value
        if is_best:
            self.best_value = current_metric
        return is_best

    def _update_early_stopping(self, val_metrics: Mapping[str, Any], epoch: int) -> None:
        """Update early-stopping counters from current validation metrics."""
        if self.early_cfg.metric not in val_metrics:
            raise KeyError(
                f"early_stopping.metric '{self.early_cfg.metric}' is missing in validation metrics. "
                f"Available keys: {sorted(val_metrics.keys())}"
            )

        current_value = float(val_metrics[self.early_cfg.metric])
        if not math.isfinite(current_value):
            raise ValueError(
                f"Validation metric '{self.early_cfg.metric}' is not finite at epoch {epoch}: {current_value}"
            )

        if self._is_early_stopping_improvement(current_value, self.early_state.best_value):
            self.early_state.best_value = current_value
            self.early_state.epochs_no_improve = 0
            return

        self.early_state.epochs_no_improve += 1

    def _save_epoch_artifacts(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        is_best: bool,
    ) -> None:
        """Persist checkpoint and history artifacts for an epoch."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "early_stopping": {
                "metric": self.early_cfg.metric,
                "mode": self.early_cfg.mode,
                "best_value": self.early_state.best_value,
                "epochs_no_improve": self.early_state.epochs_no_improve,
                "patience": self.early_cfg.patience,
                "min_delta": self.early_cfg.min_delta,
            },
        }
        if self.early_cfg.metric == "loss":
            state["early_stopping"]["best_val_loss"] = self.early_state.best_value
            state["early_stopping"]["epochs_no_val_loss_improve"] = self.early_state.epochs_no_improve

        torch.save(state, self.output_dir / "last.pt")
        if is_best:
            torch.save(state, self.best_ckpt_path)

        self.history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
                "is_best": is_best,
            }
        )
        with self.history_path.open("w", encoding="utf-8") as fp:
            json.dump(self.history, fp, indent=2)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        is_best: bool,
    ) -> None:
        """Log per-epoch metrics to MLflow."""
        if self.mlflow_module is None:
            return

        self.mlflow_module.log_metrics(_flatten_metrics(train_metrics, "train"), step=epoch)
        self.mlflow_module.log_metrics(_flatten_metrics(val_metrics, "val"), step=epoch)
        self.mlflow_module.log_metric("train/lr", float(self.optimizer.param_groups[0]["lr"]), step=epoch)
        for group_idx, group in enumerate(self.optimizer.param_groups):
            self.mlflow_module.log_metric(
                f"train/lr_group_{group_idx}",
                float(group["lr"]),
                step=epoch,
            )
        self.mlflow_module.log_metric(f"val/best_{self.checkpoint_metric}", self.best_value, step=epoch)
        self.mlflow_module.log_metric("val/is_best", 1.0 if is_best else 0.0, step=epoch)
        self.mlflow_module.log_metric("val/early_stopping_best_value", self.early_state.best_value, step=epoch)
        self.mlflow_module.log_metric(
            "val/early_stopping_epochs_no_improve",
            float(self.early_state.epochs_no_improve),
            step=epoch,
        )
        if self.early_cfg.metric == "loss":
            self.mlflow_module.log_metric("val/best_loss", self.early_state.best_value, step=epoch)
            self.mlflow_module.log_metric(
                "val/epochs_no_loss_improve",
                float(self.early_state.epochs_no_improve),
                step=epoch,
            )

    def _should_early_stop(self, epoch: int) -> bool:
        """Check whether early stopping should terminate the run.

        Returns:
            bool: True when training should stop early.
        """
        if not self.early_cfg.enabled:
            return False

        if self.early_state.epochs_no_improve < self.early_cfg.patience:
            return False

        self.early_stopped = True
        self.stopped_epoch = epoch
        LOGGER.info(
            "Early stopping triggered: validation metric '%s' did not improve for %s epoch(s). "
            "(best_value=%.4f, mode=%s, min_delta=%s)",
            self.early_cfg.metric,
            self.early_cfg.patience,
            self.early_state.best_value,
            self.early_cfg.mode,
            self.early_cfg.min_delta,
        )
        if self.mlflow_module is not None:
            self.mlflow_module.set_tag("early_stopped", "true")
            self.mlflow_module.log_metric("early_stopping/stopped_epoch", float(epoch), step=epoch)
        return True

    def _run_final_evaluation(self) -> dict[str, Any]:
        """Evaluate best checkpoint on final split and write summary.

        Returns:
            dict[str, Any]: Final run summary payload.
        """
        if self.best_ckpt_path.exists():
            best_state = torch.load(self.best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(best_state["model_state"])

        with torch.no_grad():
            test_metrics = run_epoch(
                model=self.model,
                loader=self.test_loader,
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
                desc=f"{self.final_eval_split}-final",
            )

        summary = {
            "best_metric": self.checkpoint_metric,
            "best_value": self.best_value,
            "early_stopping_metric": self.early_cfg.metric,
            "early_stopping_mode": self.early_cfg.mode,
            "early_stopping_best_value": self.early_state.best_value,
            "early_stopped": self.early_stopped,
            "stopped_epoch": self.stopped_epoch,
            "final_eval_split": self.final_eval_split,
            "final_eval_metrics": test_metrics,
        }
        if self.early_cfg.metric == "loss":
            summary["best_val_loss"] = self.early_state.best_value
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        return summary

    def _log_mlflow_end(self, summary: dict[str, Any]) -> None:
        """Log final metrics and artifacts to MLflow."""
        if self.mlflow_module is None:
            return

        final_metrics = summary["final_eval_metrics"]
        final_prefix = f"final/{self.final_eval_split}"
        self.mlflow_module.log_metrics(_flatten_metrics(final_metrics, final_prefix), step=self.epochs + 1)
        if self.final_eval_split == "test":
            self.mlflow_module.log_metrics(_flatten_metrics(final_metrics, "test"), step=self.epochs + 1)
        self.mlflow_module.log_metric("final/best_value", self.best_value, step=self.epochs + 1)
        self.mlflow_module.log_metric(
            "final/early_stopping_best_value",
            self.early_state.best_value,
            step=self.epochs + 1,
        )
        if self.early_cfg.metric == "loss":
            self.mlflow_module.log_metric("final/best_val_loss", self.early_state.best_value, step=self.epochs + 1)
        self.mlflow_module.set_tag("final_eval_split", self.final_eval_split)
        if not self.early_stopped:
            self.mlflow_module.set_tag("early_stopped", "false")

        if not self.mlflow_runtime.log_artifacts:
            return

        self.mlflow_module.log_artifact(str(self.history_path), artifact_path="outputs")
        self.mlflow_module.log_artifact(str(self.output_dir / "summary.json"), artifact_path="outputs")
        self.mlflow_module.log_artifact(str(self.split_file), artifact_path="data")
        if self.best_ckpt_path.exists():
            self.mlflow_module.log_artifact(str(self.best_ckpt_path), artifact_path="checkpoints")


def main() -> None:
    """Entrypoint for CLI execution."""
    configure_logging()
    app = TrainingApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
