import torch
import torch.nn as nn
import timm

class CichlidIDModel(nn.Module):
    def __init__(self, num_classes=25, pretrained=True, drop_rate=0.3, in_channels=4):
        super().__init__()
        # Load DINOv2 ViT-S/14 via timm
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,       # remove default head
            global_pool="token", # use CLS token
            img_size=224,
        )

        # Modify patch embedding only if input channels != 3
        if in_channels != 3:
            old_proj = self.backbone.patch_embed.proj  # Conv2d(3, embed_dim, ...)
            new_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None,
            )
            with torch.no_grad():
                new_proj.weight[:, :3, :, :] = old_proj.weight
                new_proj.weight[:, 3:, :, :] = old_proj.weight.mean(dim=1, keepdim=True)
                if old_proj.bias is not None:
                    new_proj.bias = old_proj.bias
            self.backbone.patch_embed.proj = new_proj

        # Freeze early blocks, fine-tune last 4 + head
        blocks = list(self.backbone.blocks)
        for block in blocks[:-4]:
            for param in block.parameters():
                param.requires_grad = False

        # Classification head
        embed_dim = self.backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, embed_dim]
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CichlidIDModel(num_classes=25, in_channels=3).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")  # Should be [2, 25]