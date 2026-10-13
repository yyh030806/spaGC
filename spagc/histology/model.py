"""Joint CTransPath/RGB segmentation and checkpoint restoration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BatchNorm + Activate"""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size: int = 3,
        stride: int = 1,
        padding=None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RGBEncoder(nn.Module):
    """
    input:  [B, 3, 224, 224],
    output: [B, out_channel, 8, 8]
    """

    def __init__(self, out_dim: int = 32, region_size: int = 28, region_grid: int = 8):
        super().__init__()
        self.region_grid = region_grid
        self.pool = nn.AvgPool2d(kernel_size=region_size, stride=region_size)
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb):
        if rgb.shape[-2:] == (self.region_grid, self.region_grid):
            x = rgb
        else:
            x = self.pool(rgb)
        x = self.proj(x)
        return x


class CTransPathRefiner(nn.Module):
    """Apply lightweight self-attention refinement to the 7-by-7 CTransPath token grid."""

    def __init__(
        self, dim: int = 768, depth: int = 2, heads: int = 12, dropout: float = 0.0
    ):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 49, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens):
        return self.encoder(tokens + self.pos)


class CTransPathRGBFusionSeg(nn.Module):
    """Dual-head network for joint coarse-domain and fine-niche segmentation."""

    def __init__(
        self,
        num_domain_classes: int = 6,
        num_niche_classes: int = 23,
        ctrans_in_dim: int = 768,
        ctrans_dim: int = 256,
        rgb_dim: int = 32,
        refine_depth: int = 2,
        region_grid: int = 8,
        region_size: int = 28,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.region_grid = region_grid
        self.refiner = (
            CTransPathRefiner(dim=ctrans_in_dim, depth=refine_depth, dropout=dropout)
            if refine_depth > 0
            else None
        )
        self.ctrans_proj = nn.Sequential(
            nn.Conv2d(ctrans_in_dim, ctrans_dim, 1, bias=False),
            nn.BatchNorm2d(ctrans_dim),
            nn.SiLU(inplace=True),
        )
        self.rgb_encoder = RGBEncoder(
            out_dim=rgb_dim, region_size=region_size, region_grid=region_grid
        )
        self.shared_head = nn.Sequential(
            ConvBNAct(ctrans_dim + rgb_dim, 256, kernel_size=3),
            ConvBNAct(256, 128, kernel_size=3),
        )
        self.head_drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.domain_head = nn.Conv2d(128, num_domain_classes, kernel_size=1)
        self.niche_head = nn.Conv2d(128, num_niche_classes, kernel_size=1)

    def forward(self, ctrans_tokens, rgb):
        (b, n, c) = ctrans_tokens.shape
        if self.refiner is not None:
            ctrans_tokens = self.refiner(ctrans_tokens)
        ctrans_feat = ctrans_tokens.view(b, 7, 7, c)
        ctrans_feat = ctrans_feat.permute(0, 3, 1, 2).contiguous()
        ctrans_feat = self.ctrans_proj(ctrans_feat)
        ctrans_feat = F.interpolate(
            ctrans_feat,
            size=(self.region_grid, self.region_grid),
            mode="bilinear",
            align_corners=False,
        )
        rgb_feat = self.rgb_encoder(rgb)
        feat = torch.cat([ctrans_feat, rgb_feat], dim=1)
        shared = self.shared_head(feat)
        shared = self.head_drop(shared)
        domain_logits = self.domain_head(shared)
        niche_logits = self.niche_head(shared)
        return (domain_logits, niche_logits)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def infer_model_config_from_state_dict(state_dict, fallback=None):
    """Infer the segmentation architecture saved by legacy checkpoints."""
    fallback = dict(fallback or {})

    def get_shape(name):
        value = state_dict.get(name)
        return tuple(value.shape) if value is not None else None

    domain_shape = get_shape("domain_head.weight")
    niche_shape = get_shape("niche_head.weight")
    ctrans_proj_shape = get_shape("ctrans_proj.0.weight")
    rgb_proj_shape = get_shape("rgb_encoder.proj.0.weight")
    pos_shape = get_shape("refiner.pos")
    if domain_shape is not None:
        fallback["num_domain_classes"] = domain_shape[0]
    if niche_shape is not None:
        fallback["num_niche_classes"] = niche_shape[0]
    if ctrans_proj_shape is not None:
        fallback["ctrans_dim"] = ctrans_proj_shape[0]
        fallback["ctrans_in_dim"] = ctrans_proj_shape[1]
    if rgb_proj_shape is not None:
        fallback["rgb_dim"] = rgb_proj_shape[0]
    if pos_shape is not None:
        fallback["refine_depth"] = sum(
            1
            for key in state_dict
            if key.startswith("refiner.encoder.layers.")
            and key.endswith(".self_attn.in_proj_weight")
        )
    else:
        fallback["refine_depth"] = 0
    return fallback


def load_segmentation_model(path, model_config=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    architecture = None
    if isinstance(checkpoint, dict):
        architecture = checkpoint.get("model_config")
    if architecture is None:
        architecture = infer_model_config_from_state_dict(state_dict, model_config)
    model = CTransPathRGBFusionSeg(**architecture)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if isinstance(checkpoint, dict):
        checkpoint.setdefault("model_config", architecture)
    return (model, checkpoint)
