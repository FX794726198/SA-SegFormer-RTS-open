"""Paper baseline model factory using the same manifest/split pipeline."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchvision.models.segmentation import fcn_resnet50

from .model import CASegFormer, SASegFormer


PAPER_BASELINE_MODELS = (
    "cnn",
    "unet",
    "unetpp",
    "deeplabv3plus",
    "resnet",
    "convnext",
    "swin",
    "segformer",
    "sa_segformer",
    "ca_segformer",
    "sa_convnext",
    "ca_convnext",
)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class TorchvisionFCN(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 1):
        super().__init__()
        self.model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=classes)
        if in_channels != 3:
            old_conv = self.model.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            self.model.backbone.conv1 = new_conv

    def forward(self, x):
        return self.model(x)["out"]


def _smp_model(cls, encoder_name: str, in_channels: int, classes: int, decoder_attention_type=None):
    kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": None,
        "in_channels": in_channels,
        "classes": classes,
    }
    if decoder_attention_type is not None:
        kwargs["decoder_attention_type"] = decoder_attention_type
    return cls(**kwargs)


def _segformer_model(model_cls, backbone: str, in_channels: int, classes: int):
    return model_cls(in_channels=in_channels, num_classes=classes, backbone=backbone)


def build_baseline_model(model_name: str, in_channels: int = 3, classes: int = 1) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name == "cnn":
        return SmallCNN(in_channels=in_channels, classes=classes)
    if name == "unet":
        return _smp_model(smp.Unet, "resnet34", in_channels, classes)
    if name in {"unetpp", "unetplusplus", "unet_plus_plus"}:
        return _smp_model(smp.UnetPlusPlus, "resnet34", in_channels, classes)
    if name in {"deeplabv3plus", "deeplabv3_plus", "deeplab"}:
        return _smp_model(smp.DeepLabV3Plus, "resnet34", in_channels, classes)
    if name == "resnet":
        return TorchvisionFCN(in_channels=in_channels, classes=classes)
    if name == "convnext":
        return _smp_model(smp.Unet, "tu-convnext_tiny", in_channels, classes)
    if name in {"ca_convnext", "attention_convnext"}:
        return _segformer_model(CASegFormer, "tu-convnext_tiny", in_channels, classes)
    if name in {"sa_convnext", "saconvnext"}:
        return _segformer_model(SASegFormer, "tu-convnext_tiny", in_channels, classes)
    if name == "swin":
        return _smp_model(smp.Unet, "tu-swin_tiny_patch4_window7_224", in_channels, classes)
    if name == "segformer":
        return SASegFormer(in_channels=in_channels, num_classes=classes, use_decoder_sa=False)
    if name == "ca_segformer":
        return CASegFormer(in_channels=in_channels, num_classes=classes)
    if name in {"sa_segformer", "sa_segformer_reference"}:
        return SASegFormer(in_channels=in_channels, num_classes=classes, use_decoder_sa=True)
    raise ValueError(
        f"Unknown baseline model '{model_name}'. Choose one of: " + ", ".join(PAPER_BASELINE_MODELS)
    )
