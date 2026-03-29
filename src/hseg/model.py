"""Hierarchical segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    lraspp_mobilenet_v3_large,
)

from .hierarchy import NUM_FINE_CLASSES, NUM_LEVEL0_CLASSES, NUM_LEVEL1_CLASSES
from .utils import parse_bool


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = nn.functional.pad(
                x,
                [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class HierarchicalUNet(nn.Module):
    """U-Net with three segmentation heads for hierarchical targets."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)

        self.dropout = nn.Dropout2d(p=dropout)

        self.head_l0 = nn.Conv2d(c1, NUM_LEVEL0_CLASSES, kernel_size=1)
        self.head_l1 = nn.Conv2d(c1, NUM_LEVEL1_CLASSES, kernel_size=1)
        self.head_l2 = nn.Conv2d(c1, NUM_FINE_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.dropout(x)
        return {
            "logits_l0": self.head_l0(x),
            "logits_l1": self.head_l1(x),
            "logits_l2": self.head_l2(x),
        }


class HierarchicalDeepLabV3MobileNetV3(nn.Module):
    """DeepLabV3-MobileNetV3 backbone with three hierarchical output heads."""

    def __init__(
        self,
        dropout: float = 0.1,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        weights_backbone = None
        if weights is None and pretrained_backbone:
            weights_backbone = MobileNet_V3_Large_Weights.DEFAULT

        base = deeplabv3_mobilenet_v3_large(
            weights=weights,
            weights_backbone=weights_backbone,
        )

        # Keep DeepLab feature extractor and context modules, replace final classifier layer
        # with three task-specific heads.
        self.backbone = base.backbone
        self.context = nn.Sequential(*list(base.classifier.children())[:-1])  # 960 -> 256 features
        self.dropout = nn.Dropout2d(p=dropout)

        self.head_l0 = nn.Conv2d(256, NUM_LEVEL0_CLASSES, kernel_size=1)
        self.head_l1 = nn.Conv2d(256, NUM_LEVEL1_CLASSES, kernel_size=1)
        self.head_l2 = nn.Conv2d(256, NUM_FINE_CLASSES, kernel_size=1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.backbone(x)["out"]
        features = self.context(features)
        features = self.dropout(features)

        logits_l0 = self.head_l0(features)
        logits_l1 = self.head_l1(features)
        logits_l2 = self.head_l2(features)

        logits_l0 = F.interpolate(logits_l0, size=input_size, mode="bilinear", align_corners=False)
        logits_l1 = F.interpolate(logits_l1, size=input_size, mode="bilinear", align_corners=False)
        logits_l2 = F.interpolate(logits_l2, size=input_size, mode="bilinear", align_corners=False)

        return {
            "logits_l0": logits_l0,
            "logits_l1": logits_l1,
            "logits_l2": logits_l2,
        }


class HierarchicalDeepLabV3ResNet50(nn.Module):
    """DeepLabV3-ResNet50 backbone with three hierarchical output heads."""

    def __init__(
        self,
        dropout: float = 0.1,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        weights_backbone = None
        if weights is None and pretrained_backbone:
            weights_backbone = ResNet50_Weights.DEFAULT

        base = deeplabv3_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
        )

        self.backbone = base.backbone
        self.context = nn.Sequential(*list(base.classifier.children())[:-1])  # 2048 -> 256 features
        self.dropout = nn.Dropout2d(p=dropout)

        self.head_l0 = nn.Conv2d(256, NUM_LEVEL0_CLASSES, kernel_size=1)
        self.head_l1 = nn.Conv2d(256, NUM_LEVEL1_CLASSES, kernel_size=1)
        self.head_l2 = nn.Conv2d(256, NUM_FINE_CLASSES, kernel_size=1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.backbone(x)["out"]
        features = self.context(features)
        features = self.dropout(features)

        logits_l0 = self.head_l0(features)
        logits_l1 = self.head_l1(features)
        logits_l2 = self.head_l2(features)

        logits_l0 = F.interpolate(logits_l0, size=input_size, mode="bilinear", align_corners=False)
        logits_l1 = F.interpolate(logits_l1, size=input_size, mode="bilinear", align_corners=False)
        logits_l2 = F.interpolate(logits_l2, size=input_size, mode="bilinear", align_corners=False)

        return {
            "logits_l0": logits_l0,
            "logits_l1": logits_l1,
            "logits_l2": logits_l2,
        }


class HierarchicalDeepLabV3Plus(nn.Module):
    """SMP DeepLabV3+ model with one decoder and three hierarchical output heads."""

    def __init__(
        self,
        encoder_name: str = "tu-mobilenetv3_large_100",
        encoder_weights: object = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "Model 'deeplabv3plus_smp' requires optional dependencies "
                "'segmentation-models-pytorch' and 'timm'. Install with: "
                "python -m pip install segmentation-models-pytorch timm "
                "(or install -r requirements.txt)."
            ) from exc

        resolved_encoder_weights: str | None
        if encoder_weights is None:
            resolved_encoder_weights = None
        else:
            candidate = str(encoder_weights).strip()
            resolved_encoder_weights = (
                None if candidate.lower() in {"", "none", "null"} else candidate
            )

        base = smp.DeepLabV3Plus(
            encoder_name=str(encoder_name),
            encoder_weights=resolved_encoder_weights,
            encoder_output_stride=int(encoder_output_stride),
            decoder_channels=int(decoder_channels),
            classes=1,  # segmentation_head is unused; we attach 3 custom heads below.
            activation=None,
        )

        self.encoder = base.encoder
        self.decoder = base.decoder
        self.dropout = nn.Dropout2d(p=dropout)

        decoder_out_channels = self._infer_decoder_out_channels(base)
        self.head_l0 = nn.Conv2d(decoder_out_channels, NUM_LEVEL0_CLASSES, kernel_size=1)
        self.head_l1 = nn.Conv2d(decoder_out_channels, NUM_LEVEL1_CLASSES, kernel_size=1)
        self.head_l2 = nn.Conv2d(decoder_out_channels, NUM_FINE_CLASSES, kernel_size=1)

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _infer_decoder_out_channels(base_model: nn.Module) -> int:
        decoder = getattr(base_model, "decoder", None)
        out_channels = getattr(decoder, "out_channels", None)

        if isinstance(out_channels, int) and out_channels > 0:
            return int(out_channels)
        if isinstance(out_channels, (list, tuple)) and out_channels:
            last = out_channels[-1]
            if isinstance(last, int) and last > 0:
                return int(last)

        seg_head = getattr(base_model, "segmentation_head", None)
        for attr_name in ("in_channels", "_in_channels"):
            value = getattr(seg_head, attr_name, None)
            if isinstance(value, int) and value > 0:
                return int(value)

        if isinstance(seg_head, nn.Module):
            for module in seg_head.modules():
                if isinstance(module, nn.Conv2d):
                    return int(module.in_channels)

        raise RuntimeError(
            "Failed to infer decoder output channels for SMP DeepLabV3+. "
            "Please verify segmentation-models-pytorch version compatibility."
        )

    @staticmethod
    def _decode(decoder: nn.Module, features: object) -> torch.Tensor:
        if isinstance(features, (list, tuple)):
            try:
                return decoder(*features)
            except TypeError:
                return decoder(features)
        return decoder(features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        decoded = self._decode(self.decoder, features)
        decoded = self.dropout(decoded)

        logits_l0 = self.head_l0(decoded)
        logits_l1 = self.head_l1(decoded)
        logits_l2 = self.head_l2(decoded)

        logits_l0 = F.interpolate(logits_l0, size=input_size, mode="bilinear", align_corners=False)
        logits_l1 = F.interpolate(logits_l1, size=input_size, mode="bilinear", align_corners=False)
        logits_l2 = F.interpolate(logits_l2, size=input_size, mode="bilinear", align_corners=False)

        return {
            "logits_l0": logits_l0,
            "logits_l1": logits_l1,
            "logits_l2": logits_l2,
        }


class HierarchicalSegFormer(nn.Module):
    """SMP SegFormer model with one decoder and three hierarchical output heads."""

    def __init__(
        self,
        encoder_name: str = "mit_b1",
        encoder_weights: object = "imagenet",
        decoder_segmentation_channels: int = 256,
        upsampling: int = 4,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "Model 'segformer_smp' requires optional dependencies "
                "'segmentation-models-pytorch' and 'timm'. Install with: "
                "python -m pip install segmentation-models-pytorch timm "
                "(or install -r requirements.txt)."
            ) from exc

        resolved_encoder_weights: str | None
        if encoder_weights is None:
            resolved_encoder_weights = None
        else:
            candidate = str(encoder_weights).strip()
            resolved_encoder_weights = (
                None if candidate.lower() in {"", "none", "null"} else candidate
            )

        base = smp.Segformer(
            encoder_name=str(encoder_name),
            encoder_weights=resolved_encoder_weights,
            decoder_segmentation_channels=int(decoder_segmentation_channels),
            classes=1,  # segmentation_head is unused; we attach 3 custom heads below.
            activation=None,
            upsampling=int(upsampling),
        )

        self.encoder = base.encoder
        self.decoder = base.decoder
        self.dropout = nn.Dropout2d(p=dropout)

        decoder_out_channels = HierarchicalDeepLabV3Plus._infer_decoder_out_channels(base)
        self.head_l0 = nn.Conv2d(decoder_out_channels, NUM_LEVEL0_CLASSES, kernel_size=1)
        self.head_l1 = nn.Conv2d(decoder_out_channels, NUM_LEVEL1_CLASSES, kernel_size=1)
        self.head_l2 = nn.Conv2d(decoder_out_channels, NUM_FINE_CLASSES, kernel_size=1)

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _decode(decoder: nn.Module, features: object) -> torch.Tensor:
        if isinstance(features, (list, tuple)):
            try:
                return decoder(*features)
            except TypeError:
                return decoder(features)
        return decoder(features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        decoded = self._decode(self.decoder, features)
        decoded = self.dropout(decoded)

        logits_l0 = self.head_l0(decoded)
        logits_l1 = self.head_l1(decoded)
        logits_l2 = self.head_l2(decoded)

        logits_l0 = F.interpolate(logits_l0, size=input_size, mode="bilinear", align_corners=False)
        logits_l1 = F.interpolate(logits_l1, size=input_size, mode="bilinear", align_corners=False)
        logits_l2 = F.interpolate(logits_l2, size=input_size, mode="bilinear", align_corners=False)

        return {
            "logits_l0": logits_l0,
            "logits_l1": logits_l1,
            "logits_l2": logits_l2,
        }


class _LRASPPBranch(nn.Module):
    """Single LR-ASPP branch used for one hierarchy level."""

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.low_classifier = nn.Conv2d(low_channels, num_classes, kernel_size=1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, kernel_size=1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        x = self.cbr(high)
        x = self.dropout(x)
        x = x * self.scale(high)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_classifier(low) + self.high_classifier(x)


class HierarchicalLRASPPMobileNetV3(nn.Module):
    """LR-ASPP-MobileNetV3 backbone with three hierarchical output heads."""

    def __init__(
        self,
        dropout: float = 0.1,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        inter_channels: int = 128,
    ) -> None:
        super().__init__()

        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        weights_backbone = None
        if weights is None and pretrained_backbone:
            weights_backbone = MobileNet_V3_Large_Weights.DEFAULT

        base = lraspp_mobilenet_v3_large(
            weights=weights,
            weights_backbone=weights_backbone,
        )
        self.backbone = base.backbone

        low_channels = int(base.classifier.low_classifier.in_channels)
        high_channels = int(base.classifier.cbr[0].in_channels)

        self.head_l0 = _LRASPPBranch(
            low_channels=low_channels,
            high_channels=high_channels,
            num_classes=NUM_LEVEL0_CLASSES,
            inter_channels=inter_channels,
            dropout=dropout,
        )
        self.head_l1 = _LRASPPBranch(
            low_channels=low_channels,
            high_channels=high_channels,
            num_classes=NUM_LEVEL1_CLASSES,
            inter_channels=inter_channels,
            dropout=dropout,
        )
        self.head_l2 = _LRASPPBranch(
            low_channels=low_channels,
            high_channels=high_channels,
            num_classes=NUM_FINE_CLASSES,
            inter_channels=inter_channels,
            dropout=dropout,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        low = features["low"]
        high = features["high"]

        logits_l0 = self.head_l0(low, high)
        logits_l1 = self.head_l1(low, high)
        logits_l2 = self.head_l2(low, high)

        logits_l0 = F.interpolate(logits_l0, size=input_size, mode="bilinear", align_corners=False)
        logits_l1 = F.interpolate(logits_l1, size=input_size, mode="bilinear", align_corners=False)
        logits_l2 = F.interpolate(logits_l2, size=input_size, mode="bilinear", align_corners=False)

        return {
            "logits_l0": logits_l0,
            "logits_l1": logits_l1,
            "logits_l2": logits_l2,
        }


def build_model(model_cfg: dict[str, object]) -> nn.Module:
    """Builds a segmentation model from configuration."""
    model_name = str(model_cfg.get("name", "unet")).lower()

    if model_name in {"unet", "hierarchical_unet"}:
        return HierarchicalUNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    if model_name in {"deeplabv3_mobilenet_v3_large", "deeplabv3-mobilenetv3"}:
        return HierarchicalDeepLabV3MobileNetV3(
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained=parse_bool(model_cfg.get("pretrained", True), "model.pretrained"),
            pretrained_backbone=parse_bool(
                model_cfg.get("pretrained_backbone", True),
                "model.pretrained_backbone",
            ),
            freeze_backbone=parse_bool(model_cfg.get("freeze_backbone", False), "model.freeze_backbone"),
        )

    if model_name in {"deeplabv3_resnet50", "deeplabv3-resnet50"}:
        return HierarchicalDeepLabV3ResNet50(
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained=parse_bool(model_cfg.get("pretrained", True), "model.pretrained"),
            pretrained_backbone=parse_bool(
                model_cfg.get("pretrained_backbone", True),
                "model.pretrained_backbone",
            ),
            freeze_backbone=parse_bool(model_cfg.get("freeze_backbone", False), "model.freeze_backbone"),
        )

    if model_name in {"lraspp_mobilenet_v3_large", "lraspp-mobilenetv3", "lraspp_mnv3"}:
        return HierarchicalLRASPPMobileNetV3(
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained=parse_bool(model_cfg.get("pretrained", False), "model.pretrained"),
            pretrained_backbone=parse_bool(
                model_cfg.get("pretrained_backbone", True),
                "model.pretrained_backbone",
            ),
            freeze_backbone=parse_bool(model_cfg.get("freeze_backbone", False), "model.freeze_backbone"),
            inter_channels=int(model_cfg.get("inter_channels", 128)),
        )

    if model_name in {"deeplabv3plus_smp", "deeplabv3plus-smp", "deeplabv3plus_timm"}:
        return HierarchicalDeepLabV3Plus(
            encoder_name=str(model_cfg.get("encoder_name", "tu-mobilenetv3_large_100")),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            encoder_output_stride=int(model_cfg.get("encoder_output_stride", 16)),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            freeze_backbone=parse_bool(model_cfg.get("freeze_backbone", False), "model.freeze_backbone"),
        )

    if model_name in {"segformer_smp", "segformer-smp", "segformer"}:
        return HierarchicalSegFormer(
            encoder_name=str(model_cfg.get("encoder_name", "mit_b1")),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            decoder_segmentation_channels=int(model_cfg.get("decoder_segmentation_channels", 256)),
            upsampling=int(model_cfg.get("upsampling", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            freeze_backbone=parse_bool(model_cfg.get("freeze_backbone", False), "model.freeze_backbone"),
        )

    raise ValueError(f"Unknown model name '{model_name}' in config")
