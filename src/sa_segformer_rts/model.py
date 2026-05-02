"""SA-SegFormer model definition."""

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


class SASegFormer(nn.Module):
    """SegFormer encoder with decoder-stage spatial self-attention."""

    def __init__(
        self,
        in_channels: int = 13,
        num_classes: int = 1,
        decoder_dim: int = 256,
        backbone: str = "mit_b0",
        pretrained: bool = True,
        use_decoder_sa: bool = True,
        sa_heads: int = 4,
        sa_downsample: int = 8,
    ):
        super().__init__()
        encoder_weights = "imagenet" if pretrained and in_channels == 3 else None
        self.encoder = get_encoder(backbone, in_channels=in_channels, depth=4, weights=encoder_weights)
        self.channels = [channels for channels in self.encoder.out_channels if channels > 0]
        self.proj_heads = nn.ModuleList([nn.Conv2d(channels, decoder_dim, 1) for channels in self.channels])

        self.use_decoder_sa = use_decoder_sa
        self.sa_downsample = max(1, int(sa_downsample))
        if use_decoder_sa:
            self.sa = MultiHeadSelfAttention2D(dim=decoder_dim, heads=sa_heads)
            self.sa_norm = nn.BatchNorm2d(decoder_dim)

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
        for feature, proj in zip(features, self.proj_heads):
            projected.append(F.interpolate(proj(feature), size=(height, width), mode="bilinear", align_corners=False))
        fusion = torch.stack(projected, dim=0).sum(dim=0)

        if self.use_decoder_sa:
            if self.sa_downsample > 1:
                sa_size = (max(1, height // self.sa_downsample), max(1, width // self.sa_downsample))
                sa_input = F.interpolate(fusion, size=sa_size, mode="bilinear", align_corners=False)
            else:
                sa_input = fusion
            sa_out = self.sa_norm(sa_input + self.sa(sa_input))
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
