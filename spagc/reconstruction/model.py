"""Joint spatial, molecular and cell-state velocity transformer."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    """Modulates the input tensor x with learned shift and scale parameters."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """The final layer of the model, projecting features back to the latent space."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (shift, scale) = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
        embed_dim, np.arange(grid_size, dtype=np.float32)
    )
    return pos_embed


class PatchEmbedder(nn.Module):
    """
    Embeds 1D vectors (e.g., gene expressions) into patch tokens via an MLP.

    Parameters
    ----------
    input_size : int
        The total number of features in the input 1D vector (e.g., `gene_dim`).
    patch_size : int
        The number of features to group into a single patch token.
    hidden_size : int
        The embedding dimension for the output patch tokens.
    """

    def __init__(self, input_size, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size + patch_size - 1) // patch_size
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x):
        (B, L) = x.shape
        pad_size = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size), "constant", 0)
        x = x.reshape(B, -1, self.patch_size)
        x = self.mlp(x)
        return x


class GiTBlock(nn.Module):
    """
    Transformer block with adaptive layer normalization (adaLN-Zero) conditioning.

    Parameters
    ----------
    hidden_size : int
        The dimensionality of the input and output token embeddings.
    num_heads : int
        The number of attention heads in the multi-head self-attention mechanism.
    mlp_ratio : float, optional
        The expansion ratio for the hidden dimension inside the MLP, by default 4.0.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class GiT(nn.Module):
    """
    GiT: Generative Transformer with separated streams for spatial transcriptomics.

    Parameters
    ----------
    gene_dim : int
        The number of genes (features) in the expression matrix.
    patch_size : int
        The size of the patches used to tokenize the gene expression vector.
    hidden_size : int
        The embedding dimension of the transformer backbone.
    depth : int
        The number of GiTBlocks (transformer layers).
    num_heads : int
        The number of attention heads in each block.
    num_classes : int
        The number of unique cell types for categorical conditioning.
    mlp_ratio : float, optional
        The expansion ratio for the MLP inside transformer blocks, by default 4.0.
    """

    def __init__(
        self,
        gene_dim,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        num_classes,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_x = 1
        self.num_patches_g = math.ceil(gene_dim / patch_size)
        self.num_patches = self.num_patches_x + self.num_patches_g
        self.x_embedder = nn.Linear(2, hidden_size)
        self.g_embedder = PatchEmbedder(gene_dim, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = nn.Linear(num_classes, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                GiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.x_head = FinalLayer(hidden_size, 2, 1)
        self.g_head = FinalLayer(hidden_size, patch_size, 1)
        self.c_head = nn.Linear(hidden_size, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights using GiT/DiT standards (adaLN-Zero)."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.x_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.x_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.x_head.linear.weight, 0)
        nn.init.constant_(self.x_head.linear.bias, 0)
        nn.init.constant_(self.g_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.g_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.g_head.linear.weight, 0)
        nn.init.constant_(self.g_head.linear.bias, 0)

    def forward(self, xt, gt, t, zt, delta_z, ct):
        """
        Forward pass computing the vector fields (velocity) for the given state.

        Parameters
        ----------
        xt : torch.Tensor
            Current spatial coordinates, shape `(Batch, 2)`.
        gt : torch.Tensor
            Current gene expression, shape `(Batch, Gene_Dim)`.
        t : torch.Tensor
            Current integration timestep, shape `(Batch, 1)`.
        zt : torch.Tensor
            Physical Z-depth coordinates, shape `(Batch, 1)`.
        delta_z : torch.Tensor
            The physical distance gap between the source and target slices, shape `(Batch, 1)`.
        ct : torch.Tensor
            One-hot encoded cell types, shape `(Batch, Num_Classes)`.
        """
        gene_dim = gt.shape[1]
        x_feat = self.x_embedder(xt).unsqueeze(1)
        g_feat = self.g_embedder(gt)
        h = torch.cat([x_feat, g_feat], dim=1) + self.pos_embed
        cond = (
            self.t_embedder(t.view(-1))
            + self.z_embedder(zt.view(-1))
            + self.z_embedder(delta_z.view(-1))
            + self.c_embedder(ct)
        )
        for block in self.blocks:
            h = block(h, cond)
        hx = h[:, :1, :]
        x = self.x_head(hx, cond).squeeze(1)
        g = self.g_head(h[:, 1:, :], cond).reshape(xt.shape[0], -1)[:, :gene_dim]
        c = self.c_head(h.mean(dim=1))
        return (x, g, c)
