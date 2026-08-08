import torch
import torch.nn as nn
from torchvision import models

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def build_model(num_classes: int, pretrained: bool = True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace the final classifier layer to match our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model