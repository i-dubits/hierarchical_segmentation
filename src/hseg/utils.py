"""Utility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_inheritance(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = " -> ".join(str(p) for p in (*stack, resolved_path))
        raise ValueError(f"Cyclic config inheritance detected: {cycle}")

    with resolved_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML config must be a mapping: {resolved_path}")

    base_entry = payload.pop("base_config", None)
    if base_entry is None:
        base_files: list[str] = []
    elif isinstance(base_entry, str):
        base_files = [base_entry]
    elif isinstance(base_entry, list) and all(isinstance(item, str) for item in base_entry):
        base_files = list(base_entry)
    else:
        raise TypeError(
            f"'base_config' must be a string or list of strings in {resolved_path}"
        )

    merged: dict[str, Any] = {}
    for base_ref in base_files:
        base_path = (resolved_path.parent / base_ref).resolve()
        base_payload = _load_yaml_with_inheritance(base_path, (*stack, resolved_path))
        merged = _deep_merge_dicts(merged, base_payload)

    return _deep_merge_dicts(merged, payload)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return _load_yaml_with_inheritance(Path(path), stack=())


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def parse_bool(value: object, key_name: str) -> bool:
    """Parse a strict boolean value from config payloads."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise TypeError(f"Config key '{key_name}' must be a boolean value, got {value!r}")


def select_device(device_name: str) -> torch.device:
    """Select and validate runtime device without silent fallback."""
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid device specification '{device_name}'") from exc

    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(f"Unsupported device type '{device.type}'. Use 'cpu' or 'cuda[:index]'.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available on this machine."
        )

    if device.index is not None:
        count = torch.cuda.device_count()
        if device.index < 0 or device.index >= count:
            raise ValueError(
                f"Requested CUDA device index {device.index} is out of range for {count} visible device(s)."
            )
    return device
