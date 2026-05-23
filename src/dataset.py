from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMG_SIZE = 224
NUM_CLASSES = 100


def _build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.25),
    ])


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_size: int = 5000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = _build_train_transform()
    eval_tf = _build_eval_transform()

    full_train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_tf)

    n_total = len(full_train)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader
