import torch.nn as nn
from torchvision import models

class CovidResNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet18(weights="DEFAULT")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x): return self.backbone(x)

