"""Paper model definitions for RTS semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


class MultiHeadSelfAttention2D(nn.Module):
    """Multi-head self-attention for a BCHW feature map."""

    def __init__(self, dim: int, heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=1)

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.view(batch, self.heads, channels // self.heads, height * width)
            return tensor.permute(0, 1, 3, 2)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        out = attention @ value
        out = out.permute(0, 1, 3, 2).contiguous().view(batch, channels, height, width)
        return self.proj_drop(self.proj(out))


class MultiHeadCrossAttention2D(nn.Module):
    """Cross-attention between a query feature map and a context feature map."""

    def __init__(self, dim: int, heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.query = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.key_value = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query_map: torch.Tensor, context_map: torch.Tensor) -> torch.Tensor:
        batch, channels, query_height, query_width = query_map.shape
        _batch, _channels, context_height, context_width = context_map.shape
        query = self.query(query_map)
        key, value = self.key_value(context_map).chunk(2, dim=1)

        def reshape_heads(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
            tensor = tensor.view(batch, self.heads, channels // self.heads, height * width)
            return tensor.permute(0, 1, 3, 2)

        query = reshape_heads(query, query_height, query_width)
        key = reshape_heads(key, context_height, context_width)
        value = reshape_heads(value, context_height, context_width)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        out = attention @ value
        out = out.permute(0, 1, 3, 2).contiguous().view(batch, channels, query_height, query_width)
        return self.proj_drop(self.proj(out))


class SASegFormer(nn.Module):
    """SegFormer-style encoder with optional decoder-stage self-attention."""

    def __init__(
        self,
        in_channels: int = 13,
        num_classes: int = 1,
        decoder_dim: int = 256,
        backbone: str = "mit_b0",
        pretrained: bool = True,
        use_decoder_sa: bool = True,
        use_decoder_ca: bool = False,
        sa_heads: int = 4,
        sa_downsample: int = 8,
    ):
        super().__init__()
        if use_decoder_sa and use_decoder_ca:
            raise ValueError("Choose either decoder self-attention or cross-attention, not both.")
        encoder_weights = "imagenet" if pretrained and in_channels == 3 else None
        self.encoder = get_encoder(backbone, in_channels=in_channels, depth=4, weights=encoder_weights)
        self.channels = [channels for channels in self.encoder.out_channels if channels > 0]
        self.proj_heads = nn.ModuleList([nn.Conv2d(channels, decoder_dim, 1) for channels in self.channels])

        self.use_decoder_sa = use_decoder_sa
        self.use_decoder_ca = use_decoder_ca
        self.sa_downsample = max(1, int(sa_downsample))
        if use_decoder_sa:
            self.sa = MultiHeadSelfAttention2D(dim=decoder_dim, heads=sa_heads)
        if use_decoder_ca:
            self.ca = MultiHeadCrossAttention2D(dim=decoder_dim, heads=sa_heads)
        if use_decoder_sa or use_decoder_ca:
            self.attn_norm = nn.BatchNorm2d(decoder_dim)

        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_heatmap: bool = False):
        features = [feature for feature in self.encoder(x) if feature.shape[1] > 0]
        batch, _channels, height, width = x.shape

        projected = []
        deepest_projected = None
        for idx, (feature, proj) in enumerate(zip(features, self.proj_heads)):
            projected_feature = proj(feature)
            if idx == len(features) - 1:
                deepest_projected = projected_feature
            projected.append(F.interpolate(projected_feature, size=(height, width), mode="bilinear", align_corners=False))
        fusion = torch.stack(projected, dim=0).sum(dim=0)

        if self.use_decoder_sa or self.use_decoder_ca:
            if self.sa_downsample > 1:
                sa_size = (max(1, height // self.sa_downsample), max(1, width // self.sa_downsample))
                sa_input = F.interpolate(fusion, size=sa_size, mode="bilinear", align_corners=False)
            else:
                sa_input = fusion
            if self.use_decoder_sa:
                attention_out = self.sa(sa_input)
            else:
                context = deepest_projected if deepest_projected is not None else sa_input
                if context.shape[-2:] != sa_input.shape[-2:]:
                    context = F.interpolate(context, size=sa_input.shape[-2:], mode="bilinear", align_corners=False)
                attention_out = self.ca(sa_input, context)
            sa_out = self.attn_norm(sa_input + attention_out)
            if sa_out.shape[-2:] != (height, width):
                sa_out = F.interpolate(sa_out, size=(height, width), mode="bilinear", align_corners=False)
            fusion = fusion + sa_out

        logits = self.classifier(fusion)
        if not return_heatmap:
            return logits

        deepest = features[-1]
        heatmap = F.interpolate(deepest.mean(1, keepdim=True), size=(height, width), mode="bilinear", align_corners=False)
        hm_min = heatmap.view(batch, -1).amin(dim=1)[:, None, None, None]
        hm_max = heatmap.view(batch, -1).amax(dim=1)[:, None, None, None]
        heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-6)
        return logits, heatmap.squeeze(1)


class CASegFormer(SASegFormer):
    """SegFormer-style encoder with decoder-stage cross-attention."""

    def __init__(self, *args, **kwargs):
        kwargs["use_decoder_sa"] = False
        kwargs["use_decoder_ca"] = True
        super().__init__(*args, **kwargs)


class FusionSASegFormer(SASegFormer):
    """Proposed 13-channel FusionSA-SegFormer from the manuscript."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("in_channels", 13)
        kwargs.setdefault("use_decoder_sa", True)
        kwargs["use_decoder_ca"] = False
        super().__init__(*args, **kwargs)
