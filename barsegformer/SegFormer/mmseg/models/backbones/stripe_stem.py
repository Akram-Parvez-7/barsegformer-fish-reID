# ---------------------------------------------------------------
# Stripe-Aware Anisotropic Input Stem for BarSegFormer
# Replaces OverlapPatchEmbed in stage 1 of MiT backbone
# Parallel conv branches: 3x3, 7x1 (vertical), 1x7 (horizontal)
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from mmcv.cnn import build_norm_layer


class StripeAwareStem(nn.Module):
    """
    Stripe-aware anisotropic input stem.
    Replaces the first OverlapPatchEmbed in MiT.
    Uses parallel conv branches to capture vertical stripe patterns
    from the very first layer.

    Branches:
        - 3x3 depthwise conv: local texture details
        - 7x1 conv: long vertical stripe patterns
        - 1x7 conv: limited horizontal patterns
    All branches concatenated and fused with 1x1 conv.
    """

    def __init__(self, in_chans=3, embed_dim=64, stride=4, norm_cfg=dict(type='LN', eps=1e-6)):
        super().__init__()

        # Each branch outputs embed_dim // 3 channels, we use 3 branches
        # We make sure the total adds up to embed_dim after fusion
        branch_dim = embed_dim // 3
        remainder = embed_dim - branch_dim * 3  # handle non-divisible dims

        self.branch_dim = branch_dim
        self.embed_dim = embed_dim

        # Branch 1: 3x3 depthwise separable conv — local texture
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_chans, branch_dim, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU()
        )

        # Branch 2: 7x1 conv — vertical stripe patterns
        self.branch_7x1 = nn.Sequential(
            nn.Conv2d(in_chans, branch_dim, kernel_size=(7, 1), stride=stride,
                      padding=(3, 0), bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU()
        )

        # Branch 3: 1x7 conv — horizontal patterns
        # branch_dim + remainder to make total = embed_dim
        self.branch_1x7 = nn.Sequential(
            nn.Conv2d(in_chans, branch_dim + remainder, kernel_size=(1, 7), stride=stride,
                      padding=(0, 3), bias=False),
            nn.BatchNorm2d(branch_dim + remainder),
            nn.GELU()
        )

        # 1x1 fusion conv — combine all branches into embed_dim
        self.fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)

        # LayerNorm at the end (same as original OverlapPatchEmbed)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Run all three branches in parallel
        x1 = self.branch_3x3(x)   # (B, branch_dim, H/4, W/4)
        x2 = self.branch_7x1(x)   # (B, branch_dim, H/4, W/4)
        x3 = self.branch_1x7(x)   # (B, branch_dim+remainder, H/4, W/4)

        # Concatenate along channel dimension
        x = torch.cat([x1, x2, x3], dim=1)  # (B, embed_dim, H/4, W/4)

        # Fuse with 1x1 conv
        x = self.fuse(x)  # (B, embed_dim, H/4, W/4)

        # Get spatial dims for transformer
        _, _, H, W = x.shape

        # Flatten and normalise (same interface as OverlapPatchEmbed)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm(x)

        return x, H, W
