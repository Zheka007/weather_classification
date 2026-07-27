import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=3, type_model='resnet50', freeze_backbone=True, unfreeze_last_block=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type_model == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif type_model == 'resnet18':
        model = models.resnet18(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_last_block:
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model.to(device)