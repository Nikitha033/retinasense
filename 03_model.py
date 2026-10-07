"""
03_model.py
-----------
RetinaSense High-Accuracy Medical Deep Learning Architecture:
  - EfficientNet-B0 backbone (ImageNet-pretrained)
  - Dual Global Context Pooling (AdaptiveAvgPool2d + AdaptiveMaxPool2d -> 2560 channels)
  - Dual-head design:
      * disease_head : 6-way multi-label prediction (D, G, C, A, H, M)
      * severity_head: 5-way clinical DR severity grading (0=Early ... 4=Proliferative)
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_DISEASES = 6
NUM_SEVERITY_CLASSES = 5


class RetinaSenseModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone_layers=False):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = backbone.features
        feat_dim = 1280
        self.cam_layer = self.backbone[-1]

        if freeze_backbone_layers:
            for i, child in enumerate(self.backbone.children()):
                if i < 4:
                    for p in child.parameters():
                        p.requires_grad = False

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        pooled_dim = feat_dim * 2  # 2560 channels (AvgPool + MaxPool)

        self.disease_head = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_DISEASES),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, NUM_SEVERITY_CLASSES),
        )

    def get_cam_layer(self):
        return self.cam_layer

    def forward(self, x):
        feats = self.backbone(x)
        avg_f = self.avg_pool(feats).flatten(1)
        max_f = self.max_pool(feats).flatten(1)
        combined = torch.cat([avg_f, max_f], dim=1)

        disease_logits = self.disease_head(combined)
        severity_logits = self.severity_head(combined)
        return disease_logits, severity_logits


if __name__ == "__main__":
    m = RetinaSenseModel(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    d, s = m(dummy)
    print("RetinaSenseModel initialized successfully.")
    print("disease logits shape:", d.shape)
    print("severity logits shape:", s.shape)