"""
core/fusion.py
TERRA·LENS — Adaptive Multi-GFM Fusion Module

Drop this file into the `core/` folder of your forked repo.
It adds a cross-attention fusion layer that learns which GFM
to trust more per spatial location, then feeds into the
existing decoder_residual backbone.

Usage in train.py:
    from core.fusion import TerraLens, fuse_embeddings
    model = TerraLens(n_classes=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel projection — maps each GFM to a common dimension
# ---------------------------------------------------------------------------

class GFMProjector(nn.Module):
    """
    Projects one GFM embedding (arbitrary channels) → hidden_dim.
    Works for both pixel-wise (B, C, H, W) and token (B, C, H, W) embeddings.
    """
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Spatial cross-attention fusion
# ---------------------------------------------------------------------------

class AdaptiveFusionModule(nn.Module):
    """
    Cross-attention fusion across N GFM embeddings.

    Each spatial location (pixel) independently learns how much
    to weight each GFM via a softmax attention score.

    Args:
        hidden_dim : common projection dimension for all GFMs
        num_gfms   : number of GFMs to fuse (default 4)
        num_heads  : attention heads (default 4)
    """
    def __init__(self, hidden_dim: int = 256, num_gfms: int = 4, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gfms = num_gfms
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # Query: learned spatial context
        self.query = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        # Keys: one per GFM
        self.keys = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
            for _ in range(num_gfms)
        ])
        # Values: one per GFM
        self.values = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
            for _ in range(num_gfms)
        ])

        self.scale = head_dim ** -0.5
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, hidden_dim)

    def forward(self, projected: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            projected: list of N tensors, each (B, hidden_dim, H, W)
        Returns:
            fused: (B, hidden_dim, H, W)
        """
        # Use mean of all projections as the spatial query context
        context = torch.stack(projected, dim=0).mean(dim=0)  # (B, D, H, W)
        q = self.query(context)  # (B, D, H, W)

        B, D, H, W = q.shape

        # Compute attention score per GFM per spatial location
        attn_scores = []
        values_list = []
        for i, emb in enumerate(projected):
            k = self.keys[i](emb)   # (B, D, H, W)
            v = self.values[i](emb) # (B, D, H, W)
            # Dot-product score: sum over channel dim → (B, H, W)
            score = (q * k).sum(dim=1, keepdim=True) * self.scale  # (B, 1, H, W)
            attn_scores.append(score)
            values_list.append(v)

        # Softmax over GFM dimension → adaptive weights
        attn = torch.cat(attn_scores, dim=1)          # (B, N, H, W)
        attn = F.softmax(attn, dim=1)                 # (B, N, H, W) — sums to 1 per pixel

        # Weighted sum of values
        fused = torch.zeros_like(values_list[0])
        for i, v in enumerate(values_list):
            w = attn[:, i:i+1, :, :]                  # (B, 1, H, W)
            fused = fused + w * v

        fused = self.out_proj(fused)
        fused = self.norm(fused)
        return fused


# ---------------------------------------------------------------------------
# TERRA·LENS — full model
# ---------------------------------------------------------------------------

class TerraLens(nn.Module):
    """
    TERRA·LENS: Adaptive Multi-GFM Fusion + Dual-Head Decoder

    Fuses 4 GFM embeddings via cross-attention, then decodes to:
      - 3 segmentation channels : % Building, % Vegetation, % Water
      - 1 height channel        : nDSM in meters (normalised)

    Args:
        gfm_channels : list of input channels per GFM
                       Default matches AlphaEarth=64, TESSERA=128,
                       TerraMind=256, THOR=256
        hidden_dim   : internal fusion dimension
        n_classes    : total output channels (default 4)
    """
    def __init__(
        self,
        gfm_channels: list[int] = [64, 128, 256, 256],
        hidden_dim: int = 256,
        n_classes: int = 4,
    ):
        super().__init__()
        self.num_gfms = len(gfm_channels)

        # Per-GFM channel projectors
        self.projectors = nn.ModuleList([
            GFMProjector(c, hidden_dim) for c in gfm_channels
        ])

        # Adaptive cross-attention fusion
        self.fusion = AdaptiveFusionModule(hidden_dim=hidden_dim, num_gfms=self.num_gfms)

        # FPN-style decoder backbone
        self.decoder = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim // 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(hidden_dim // 2, hidden_dim // 4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(hidden_dim // 4, hidden_dim // 4),
        )

        # Dual output heads
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),   # Building, Vegetation, Water
            nn.Sigmoid(),                        # outputs in [0, 1]
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),    # nDSM height
            nn.ReLU(),                           # height >= 0
        )

    def forward(self, gfm_inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gfm_inputs: list of 4 tensors [(B, C_i, H, W), ...]
                        H, W must be identical across all GFMs
                        (interpolate upstream if needed)
        Returns:
            out: (B, 4, H, W)
                 channels: [% Building, % Vegetation, % Water, Height]
        """
        # 1. Project each GFM to hidden_dim
        projected = [proj(x) for proj, x in zip(self.projectors, gfm_inputs)]

        # 2. Adaptive fusion
        fused = self.fusion(projected)  # (B, hidden_dim, H, W)

        # 3. Decode
        decoded = self.decoder(fused)   # (B, hidden_dim//4, H*4, W*4)

        # Resize back to input spatial size if needed
        h_in, w_in = gfm_inputs[0].shape[2], gfm_inputs[0].shape[3]
        if decoded.shape[2] != h_in:
            decoded = F.interpolate(decoded, size=(h_in, w_in),
                                    mode='bilinear', align_corners=False)

        # 4. Dual heads
        seg = self.seg_head(decoded)         # (B, 3, H, W)
        height = self.height_head(decoded)   # (B, 1, H, W)

        return torch.cat([seg, height], dim=1)  # (B, 4, H, W)


# ---------------------------------------------------------------------------
# Residual block (used inside decoder)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) \
                    if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Helper: align all GFM embeddings to the same spatial size
# ---------------------------------------------------------------------------

def align_embeddings(embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Resize all embeddings to the spatial size of the first one.
    Call this before passing to TerraLens.forward().
    """
    target_h, target_w = embeddings[0].shape[2], embeddings[0].shape[3]
    aligned = []
    for emb in embeddings:
        if emb.shape[2] != target_h or emb.shape[3] != target_w:
            emb = F.interpolate(emb, size=(target_h, target_w),
                                mode='bilinear', align_corners=False)
        aligned.append(emb)
    return aligned


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, H, W = 2, 256, 256

    # Simulate 4 GFM embeddings with different channel counts
    alpha_earth  = torch.randn(B, 64,  H, W)   # AlphaEarth
    tessera      = torch.randn(B, 128, H, W)   # TESSERA
    terramind    = torch.randn(B, 256, H, W)   # TerraMind
    thor         = torch.randn(B, 256, H, W)   # THOR

    inputs = align_embeddings([alpha_earth, tessera, terramind, thor])

    model = TerraLens(gfm_channels=[64, 128, 256, 256], hidden_dim=256)
    out = model(inputs)

    print(f"Input shapes : {[list(x.shape) for x in inputs]}")
    print(f"Output shape : {list(out.shape)}")   # expect [2, 4, 256, 256]
    print(f"Seg range    : {out[:, :3].min():.3f} – {out[:, :3].max():.3f}")
    print(f"Height range : {out[:, 3].min():.3f} – {out[:, 3].max():.3f}")
    print("TERRA·LENS sanity check passed.")
