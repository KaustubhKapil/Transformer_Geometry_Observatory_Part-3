from __future__ import annotations

import timm
import torch.nn as nn


class ViTProbe(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, drop_rate: float = 0.0, drop_path_rate: float = 0.1):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.model(x)

    @property
    def vit(self):
        return self.model
