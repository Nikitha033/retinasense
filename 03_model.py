"""
03_model.py
-----------
RetinaSense model: an EfficientNet-B0 backbone (ImageNet-pretrained)
shared by two heads:
  - disease_head : 7-way sigmoid (multi-label disease presence: N, D, G, C, A, H, M)
  - severity_head: 5-way softmax (DR severity: 0=None/unspecified ... 4=Proliferative)

The severity head is only meaningfully supervised on DR-positive samples
(see 04_train.py for how its loss is masked).
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_DISEASES = 7
NUM_SEVERITY_CLASSES = 5


class RetinaSenseModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone_layers=True):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280  # EfficientNet-B0 final feature channels

        if freeze_backbone_layers:
            # Freeze the early layers, fine-tune the later blocks — standard
            # transfer-learning trick when the target dataset (ODIR, ~7000
            # images) is much smaller than ImageNet.
            for i, child in enumerate(self.backbone.children()):
                if i < 5:
                    for p in child.parameters():
                        p.requires_grad = False

        self.disease_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_DISEASES),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_SEVERITY_CLASSES),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        disease_logits = self.disease_head(feats)
        severity_logits = self.severity_head(feats)
        return disease_logits, severity_logits


if __name__ == "__main__":
    m = RetinaSenseModel()
    dummy = torch.randn(2, 3, 224, 224)
    d_out, s_out = m(dummy)
    print("disease logits:", d_out.shape)   # [2, 7]
    print("severity logits:", s_out.shape)  # [2, 5]