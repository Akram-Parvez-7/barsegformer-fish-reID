# ---------------------------------------------------------------
# BarSegFormer Decode Head
# U-shaped decoder with:
#   - Skip connections from all 4 encoder stages
#   - Progressive upsampling (4 stages)
#   - 3x3 conv refinement at each step
#   - Edge prediction head (training only)
#   - Combined segmentation + edge loss
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy


class ConvBNReLU(nn.Module):
    """3x3 Conv + BN + ReLU refinement block used throughout the decoder."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Single upsampling step in the U-shaped decoder.
    - Bilinear upsample x2
    - Concatenate with skip connection from encoder
    - 3x3 conv refinement
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.refine = ConvBNReLU(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # Upsample to match skip connection spatial size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)
        # Refine
        x = self.refine(x)
        return x


@HEADS.register_module()
class BarSegFormerHead(BaseDecodeHead):
    """
    BarSegFormer decode head.
    U-shaped decoder with skip connections and edge supervision.

    Args:
        feature_strides (list): strides of encoder feature maps [4, 8, 16, 32]
        embed_dim (int): internal decoder channel dimension
        edge_loss_weight (float): weight for edge loss in combined loss
    """

    def __init__(self, feature_strides, embed_dim=256, edge_loss_weight=0.4, **kwargs):
        super(BarSegFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert len(feature_strides) == len(self.in_channels)
        self.feature_strides = feature_strides
        self.embed_dim = embed_dim
        self.edge_loss_weight = edge_loss_weight

        c1_ch, c2_ch, c3_ch, c4_ch = self.in_channels

        # Project all encoder features to embed_dim
        self.proj_c4 = ConvBNReLU(c4_ch, embed_dim)
        self.proj_c3 = ConvBNReLU(c3_ch, embed_dim)
        self.proj_c2 = ConvBNReLU(c2_ch, embed_dim)
        self.proj_c1 = ConvBNReLU(c1_ch, embed_dim)

        # U-shaped decoder: progressively upsample with skip connections
        # Stage 1: c4 (1/32) -> fuse with c3 (1/16)
        self.up1 = UpBlock(embed_dim, embed_dim, embed_dim)
        # Stage 2: (1/16) -> fuse with c2 (1/8)
        self.up2 = UpBlock(embed_dim, embed_dim, embed_dim)
        # Stage 3: (1/8) -> fuse with c1 (1/4)
        self.up3 = UpBlock(embed_dim, embed_dim, embed_dim)
        # Stage 4: final refinement at 1/4 scale
        self.final_refine = ConvBNReLU(embed_dim, embed_dim)

        # Segmentation prediction head
        self.seg_head = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

        # Edge prediction head (used during training only)
        self.edge_head = nn.Sequential(
            ConvBNReLU(embed_dim, 64),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Edge loss
        self.edge_loss_fn = nn.BCEWithLogitsLoss()

    def _generate_edge_map(self, seg_label):
        """
        Auto-generate edge maps from segmentation masks using Sobel filtering.
        Args:
            seg_label: (B, H, W) integer mask with values 0/1
        Returns:
            edge_map: (B, 1, H, W) float binary edge map
        """
        # Convert to float and add channel dim: (B, 1, H, W)
        x = seg_label.float().unsqueeze(1)

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=torch.float32,
                                 device=seg_label.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                  [0,  0,  0],
                                  [1,  2,  1]], dtype=torch.float32,
                                 device=seg_label.device).view(1, 1, 3, 3)

        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        # Binarise: any non-zero gradient = edge
        edge = (edge > 0).float()
        return edge

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x

        # Project encoder features to embed_dim
        p4 = self.proj_c4(c4)   # (B, embed_dim, H/32, W/32)
        p3 = self.proj_c3(c3)   # (B, embed_dim, H/16, W/16)
        p2 = self.proj_c2(c2)   # (B, embed_dim, H/8,  W/8)
        p1 = self.proj_c1(c1)   # (B, embed_dim, H/4,  W/4)

        # U-shaped progressive upsampling with skip connections
        d3 = self.up1(p4, p3)   # (B, embed_dim, H/16, W/16)
        d2 = self.up2(d3, p2)   # (B, embed_dim, H/8,  W/8)
        d1 = self.up3(d2, p1)   # (B, embed_dim, H/4,  W/4)
        d1 = self.final_refine(d1)

        # Segmentation output
        seg_out = self.seg_head(d1)

        if self.training:
            # Edge output (training only)
            edge_out = self.edge_head(d1)
            return seg_out, edge_out
        else:
            return seg_out

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Override forward_train to add edge loss."""
        seg_out, edge_out = self.forward(inputs)

        gt_seg = gt_semantic_seg.squeeze(1).long()  # (B, H, W)

        # Resize seg output to match gt size
        seg_out = resize(seg_out, size=gt_seg.shape[1:],
                        mode='bilinear', align_corners=self.align_corners)

        # Resize edge output to match gt size
        edge_out = resize(edge_out, size=gt_seg.shape[1:],
                        mode='bilinear', align_corners=self.align_corners)

        # Generate edge maps from gt masks
        edge_gt = self._generate_edge_map(gt_seg)  # (B, 1, H, W)

        # Segmentation loss directly
        seg_loss = F.cross_entropy(seg_out, gt_seg, ignore_index=self.ignore_index)

        # Edge loss
        edge_loss = self.edge_loss_fn(edge_out, edge_gt)

        losses = dict()
        losses['loss_seg'] = seg_loss
        losses['loss_edge'] = edge_loss * self.edge_loss_weight
        losses['decode.loss_seg'] = seg_loss
        losses['decode.acc_seg'] = accuracy(seg_out, gt_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Inference — edge head not used."""
        return self.forward(inputs)
