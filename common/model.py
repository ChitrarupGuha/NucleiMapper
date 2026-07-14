"""
model.py - builds the transfer-learning CNN backbone and loads checkpoints.
Shared by train.py, evaluate.py and the FastAPI inference endpoint.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def load_checkpoint(checkpoint_path: str, num_classes: int, arch: str = "resnet18", device="cpu") -> nn.Module:
    model = build_model(num_classes, arch=arch, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
