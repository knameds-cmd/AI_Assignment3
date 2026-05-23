import os
import random
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_optimizer(name: str, params: Iterable, lr: float, weight_decay: float) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'. Choose from sgd, adam, adamw.")


def get_scheduler(name: str, optimizer: Optimizer, epochs: int):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "multistep":
        milestones = [int(epochs * 0.5), int(epochs * 0.75)]
        return MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    raise ValueError(f"Unsupported scheduler '{name}'. Choose from none, cosine, step, multistep.")


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, ...] = (1, 5)) -> Tuple[float, ...]:
    maxk = max(ks)
    batch = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        results.append(correct_k / batch)
    return tuple(results)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
