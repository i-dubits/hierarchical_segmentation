"""Image/mask transforms for segmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from .hierarchy import NUM_FINE_CLASSES


def _parse_bool(value: object, key_name: str) -> bool:
    """Parse strict boolean values from transform config payloads."""
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
    raise TypeError(f"Config key '{key_name}' must be boolean, got {value!r}")


def _parse_pair(value: object, key_name: str) -> tuple[int, int]:
    """Parse `(height, width)` integer pair from config value."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Config key '{key_name}' must be a sequence of two integers.")
    values = [int(item) for item in value]
    if len(values) != 2:
        raise ValueError(f"Config key '{key_name}' must contain two integers, got {values!r}")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"Config key '{key_name}' must be > 0, got {values!r}")
    return values[0], values[1]


def _parse_probability(value: object, key_name: str) -> float:
    """Parse probability value in the inclusive `[0, 1]` range."""
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"Config key '{key_name}' must be in [0, 1], got {probability}")
    return probability


def _as_mapping(value: object, key_name: str) -> dict[str, object]:
    """Normalize optional transform config section into a mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Config key '{key_name}' must be a mapping when provided.")


@dataclass(frozen=True)
class ClassAwareCropConfig:
    """Configuration for class-aware crop augmentation."""

    enabled: bool
    probability: float
    crop_size: tuple[int, int]
    target_class_ids: tuple[int, ...]
    min_target_pixels: int
    max_tries: int
    fallback_to_random_crop: bool


@dataclass(frozen=True)
class ScaleJitterConfig:
    """Configuration for random scale jitter augmentation."""

    enabled: bool
    min_scale: float
    max_scale: float


@dataclass(frozen=True)
class PhotometricDistortionConfig:
    """Configuration for photometric distortion augmentation."""

    enabled: bool
    probability: float
    brightness: float
    contrast: float
    saturation: float
    hue: float


@dataclass(frozen=True)
class AugmentationConfig:
    """Container for train-time augmentations beyond crop/flip."""

    scale_jitter: ScaleJitterConfig
    photometric_distortion: PhotometricDistortionConfig


class ClassAwareCropper:
    """Sample crop windows that preferentially include target classes."""

    def __init__(self, config: ClassAwareCropConfig) -> None:
        """Store immutable crop configuration."""
        self.config = config

    def maybe_apply(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """Apply class-aware crop with configured probability."""
        if not self.config.enabled or random.random() >= self.config.probability:
            return image, mask

        crop_h, crop_w = self._effective_crop_size(mask)
        if crop_h == mask.shape[0] and crop_w == mask.shape[1]:
            return image, mask

        target_pixels = np.argwhere(np.isin(mask, self.config.target_class_ids))
        if target_pixels.size > 0:
            top, left = self._sample_class_aware_window(mask, target_pixels, crop_h, crop_w)
            return self._crop(image=image, mask=mask, top=top, left=left, crop_h=crop_h, crop_w=crop_w)

        if self.config.fallback_to_random_crop:
            top, left = self._sample_random_window(mask, crop_h, crop_w)
            return self._crop(image=image, mask=mask, top=top, left=left, crop_h=crop_h, crop_w=crop_w)

        return image, mask

    def _effective_crop_size(self, mask: np.ndarray) -> tuple[int, int]:
        """Resolve crop size clipped to current mask dimensions."""
        crop_h = min(self.config.crop_size[0], int(mask.shape[0]))
        crop_w = min(self.config.crop_size[1], int(mask.shape[1]))
        return crop_h, crop_w

    def _sample_class_aware_window(
        self,
        mask: np.ndarray,
        target_pixels: np.ndarray,
        crop_h: int,
        crop_w: int,
    ) -> tuple[int, int]:
        """Find crop window that contains enough target-class pixels."""
        max_top = int(mask.shape[0] - crop_h)
        max_left = int(mask.shape[1] - crop_w)

        chosen_top = 0
        chosen_left = 0
        for attempt_idx in range(self.config.max_tries):
            candidate_idx = random.randrange(len(target_pixels))
            center_y, center_x = target_pixels[candidate_idx]

            top = max(0, min(int(center_y) - crop_h // 2, max_top))
            left = max(0, min(int(center_x) - crop_w // 2, max_left))
            chosen_top = top
            chosen_left = left

            crop = mask[top : top + crop_h, left : left + crop_w]
            target_count = int(np.isin(crop, self.config.target_class_ids).sum())
            if target_count >= self.config.min_target_pixels:
                return top, left

            if attempt_idx + 1 == self.config.max_tries:
                return chosen_top, chosen_left

        return chosen_top, chosen_left

    @staticmethod
    def _sample_random_window(mask: np.ndarray, crop_h: int, crop_w: int) -> tuple[int, int]:
        """Sample a random valid crop window for fallback behavior."""
        max_top = int(mask.shape[0] - crop_h)
        max_left = int(mask.shape[1] - crop_w)
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        return top, left

    @staticmethod
    def _crop(
        image: Image.Image,
        mask: np.ndarray,
        top: int,
        left: int,
        crop_h: int,
        crop_w: int,
    ) -> tuple[Image.Image, np.ndarray]:
        """Apply aligned crop to image and mask."""
        image_crop = image.crop((left, top, left + crop_w, top + crop_h))
        mask_crop = mask[top : top + crop_h, left : left + crop_w]
        return image_crop, mask_crop


def _resolve_class_aware_crop_config(
    class_aware_crop_cfg: Mapping[str, object] | None,
    image_size: tuple[int, int],
) -> ClassAwareCropConfig:
    """Validate and resolve class-aware crop configuration."""
    if class_aware_crop_cfg is None:
        return ClassAwareCropConfig(
            enabled=False,
            probability=0.5,
            crop_size=image_size,
            target_class_ids=tuple(range(1, NUM_FINE_CLASSES)),
            min_target_pixels=32,
            max_tries=10,
            fallback_to_random_crop=True,
        )

    cfg = dict(class_aware_crop_cfg)

    enabled = _parse_bool(cfg.get("enabled", False), "data.class_aware_crop.enabled")
    probability = _parse_probability(cfg.get("probability", 0.5), "data.class_aware_crop.probability")

    crop_size_raw = cfg.get("crop_size")
    if crop_size_raw is None:
        crop_size = image_size
    else:
        crop_size = _parse_pair(crop_size_raw, "data.class_aware_crop.crop_size")

    target_class_ids_raw = cfg.get("target_class_ids", list(range(1, NUM_FINE_CLASSES)))
    if not isinstance(target_class_ids_raw, Sequence) or isinstance(target_class_ids_raw, (str, bytes)):
        raise TypeError("Config key 'data.class_aware_crop.target_class_ids' must be a sequence of ints.")
    target_class_ids_values = [int(class_id) for class_id in target_class_ids_raw]
    if not target_class_ids_values:
        raise ValueError("Config key 'data.class_aware_crop.target_class_ids' cannot be empty.")
    normalized_class_ids: list[int] = []
    seen_ids: set[int] = set()
    for class_id in target_class_ids_values:
        if class_id < 0 or class_id >= NUM_FINE_CLASSES:
            raise ValueError(
                "Config key 'data.class_aware_crop.target_class_ids' contains invalid class id "
                f"{class_id}; expected [0..{NUM_FINE_CLASSES - 1}]"
            )
        if class_id not in seen_ids:
            seen_ids.add(class_id)
            normalized_class_ids.append(class_id)

    min_target_pixels = int(cfg.get("min_target_pixels", 32))
    if min_target_pixels < 1:
        raise ValueError(
            "Config key 'data.class_aware_crop.min_target_pixels' must be >= 1, "
            f"got {min_target_pixels}"
        )

    max_tries = int(cfg.get("max_tries", 10))
    if max_tries < 1:
        raise ValueError(
            "Config key 'data.class_aware_crop.max_tries' must be >= 1, "
            f"got {max_tries}"
        )

    fallback_to_random_crop = _parse_bool(
        cfg.get("fallback_to_random_crop", True),
        "data.class_aware_crop.fallback_to_random_crop",
    )

    return ClassAwareCropConfig(
        enabled=enabled,
        probability=probability,
        crop_size=crop_size,
        target_class_ids=tuple(normalized_class_ids),
        min_target_pixels=min_target_pixels,
        max_tries=max_tries,
        fallback_to_random_crop=fallback_to_random_crop,
    )


def _resolve_augmentation_config(augmentation_cfg: Mapping[str, object] | None) -> AugmentationConfig:
    """Validate and resolve optional train-time augmentations."""
    cfg = _as_mapping(augmentation_cfg, "data.augmentations")

    scale_cfg = _as_mapping(cfg.get("scale_jitter"), "data.augmentations.scale_jitter")
    scale_enabled = _parse_bool(
        scale_cfg.get("enabled", False),
        "data.augmentations.scale_jitter.enabled",
    )
    min_scale = float(scale_cfg.get("min_scale", 1.0))
    max_scale = float(scale_cfg.get("max_scale", 1.0))
    if min_scale <= 0.0 or max_scale <= 0.0:
        raise ValueError(
            "Config keys 'data.augmentations.scale_jitter.min_scale' and "
            "'data.augmentations.scale_jitter.max_scale' must be > 0."
        )
    if min_scale > max_scale:
        raise ValueError(
            "Config key 'data.augmentations.scale_jitter.min_scale' must be <= "
            f"'max_scale', got {min_scale} > {max_scale}"
        )

    photometric_cfg = _as_mapping(
        cfg.get("photometric_distortion"),
        "data.augmentations.photometric_distortion",
    )
    photometric_enabled = _parse_bool(
        photometric_cfg.get("enabled", False),
        "data.augmentations.photometric_distortion.enabled",
    )
    photometric_probability = _parse_probability(
        photometric_cfg.get("probability", 0.8),
        "data.augmentations.photometric_distortion.probability",
    )
    brightness = float(photometric_cfg.get("brightness", 0.0))
    contrast = float(photometric_cfg.get("contrast", 0.0))
    saturation = float(photometric_cfg.get("saturation", 0.0))
    hue = float(photometric_cfg.get("hue", 0.0))
    for key_name, value in (
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
    ):
        if value < 0.0:
            raise ValueError(
                f"Config key 'data.augmentations.photometric_distortion.{key_name}' "
                f"must be >= 0, got {value}"
            )
    if hue < 0.0 or hue > 0.5:
        raise ValueError(
            "Config key 'data.augmentations.photometric_distortion.hue' must be in [0, 0.5], "
            f"got {hue}"
        )

    return AugmentationConfig(
        scale_jitter=ScaleJitterConfig(
            enabled=scale_enabled,
            min_scale=min_scale,
            max_scale=max_scale,
        ),
        photometric_distortion=PhotometricDistortionConfig(
            enabled=photometric_enabled,
            probability=photometric_probability,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
    )


class SegmentationTransforms:
    """Applies paired transforms to image and mask."""

    def __init__(
        self,
        image_size: Sequence[int],
        train: bool,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        class_aware_crop_cfg: Mapping[str, object] | None = None,
        augmentation_cfg: Mapping[str, object] | None = None,
    ) -> None:
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must have 2 values, got {image_size}")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError(f"image_size values must be > 0, got {self.image_size}")
        self.train = train
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError(
                f"Expected 3 normalization values for mean/std, got mean={self.mean}, std={self.std}"
            )
        if any(value <= 0.0 for value in self.std):
            raise ValueError(f"Normalization std values must be > 0, got {self.std}")
        class_aware_crop_config = _resolve_class_aware_crop_config(
            class_aware_crop_cfg=class_aware_crop_cfg,
            image_size=self.image_size,
        )
        self.augmentation_config = _resolve_augmentation_config(augmentation_cfg)
        self.class_aware_cropper = ClassAwareCropper(class_aware_crop_config)
        photometric_config = self.augmentation_config.photometric_distortion
        self.color_jitter: ColorJitter | None = None
        if photometric_config.enabled:
            self.color_jitter = ColorJitter(
                brightness=photometric_config.brightness,
                contrast=photometric_config.contrast,
                saturation=photometric_config.saturation,
                hue=photometric_config.hue,
            )

    def _maybe_apply_scale_jitter(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        """Resize image and mask with a shared random scale factor."""
        cfg = self.augmentation_config.scale_jitter
        if not cfg.enabled:
            return image, mask

        scale = random.uniform(cfg.min_scale, cfg.max_scale)
        resized_height = max(1, int(round(image.height * scale)))
        resized_width = max(1, int(round(image.width * scale)))
        if resized_height == image.height and resized_width == image.width:
            return image, mask

        resized_image = F.resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        resized_mask = F.resize(
            mask_pil,
            [resized_height, resized_width],
            interpolation=InterpolationMode.NEAREST,
        )
        return resized_image, np.array(resized_mask, dtype=mask.dtype)

    def _maybe_apply_photometric_distortion(self, image: Image.Image) -> Image.Image:
        """Apply standard color jitter augmentation with configured probability."""
        cfg = self.augmentation_config.photometric_distortion
        if not cfg.enabled or self.color_jitter is None:
            return image
        if random.random() >= cfg.probability:
            return image
        return self.color_jitter(image)

    def __call__(self, image: Image.Image, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            image, mask = self._maybe_apply_scale_jitter(image, mask)
            image, mask = self.class_aware_cropper.maybe_apply(image, mask)

        if self.train and random.random() < 0.5:
            image = F.hflip(image)
            mask = np.fliplr(mask).copy()

        if self.train:
            image = self._maybe_apply_photometric_distortion(image)

        image = F.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_pil = F.resize(mask_pil, self.image_size, interpolation=InterpolationMode.NEAREST)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)
        mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        return image_tensor, mask_tensor
