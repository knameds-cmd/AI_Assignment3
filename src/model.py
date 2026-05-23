import torch.nn as nn
import torchvision.models as tvm


SUPPORTED = {"resnet18", "resnet34"}


def build_resnet(name: str, pretrained: bool, num_classes: int = 100) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported model '{name}'. Choose from {sorted(SUPPORTED)}.")

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
    else:
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet34(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
