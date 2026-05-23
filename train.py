import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import NUM_CLASSES, build_dataloaders
from src.model import build_resnet
from src.utils import AverageMeter, count_params, get_optimizer, get_scheduler, set_seed, topk_accuracy


RESULTS_FIELDS = [
    "run_name", "model", "init", "batch_size", "lr", "epochs",
    "optimizer", "scheduler", "weight_decay",
    "best_val_acc", "best_val_epoch", "test_acc", "test_top5_acc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet on CIFAR-100")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"], help="model architecture")
    p.add_argument("--batch_size", type=int, default=128, help="mini-batch size")
    p.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    p.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="optimizer")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "multistep"], help="learning-rate scheduler")
    p.add_argument("--pretrained", type=int, default=0, choices=[0, 1], help="1 for ImageNet-pretrained init, 0 for from scratch")
    p.add_argument("--save_csv", type=str, default="results/results.csv", help="aggregate results CSV path")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--save_dir", type=str, default="checkpoints", help="directory to save best checkpoints")
    p.add_argument("--data_dir", type=str, default="./data", help="CIFAR-100 root directory")
    p.add_argument("--run_name", type=str, default=None, help="unique experiment name (auto-generated if omitted)")
    p.add_argument("--val_size", type=int, default=5000, help="number of images held out for validation")
    p.add_argument("--log_dir", type=str, default="results/logs", help="per-epoch log directory")
    p.add_argument("--plot_dir", type=str, default="results/plots", help="learning-curve plot directory")
    return p.parse_args()


def auto_run_name(args: argparse.Namespace) -> str:
    init = "pretrained" if args.pretrained else "scratch"
    return f"{args.model}_{init}_{args.optimizer}_{args.scheduler}_lr{args.lr}_e{args.epochs}"


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: nn.Module, device: torch.device,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    start = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.size(0))
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"    step {i + 1}/{len(loader)}  loss {loss_meter.avg:.4f}  elapsed {elapsed:.1f}s", flush=True)
    return loss_meter.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        top1, top5 = topk_accuracy(logits, targets, ks=(1, 5))
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1, n)
        top5_meter.update(top5, n)
    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def evaluate_with_class_acc(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    device: torch.device, num_classes: int,
) -> Tuple[float, float, float, List[float]]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        top1, top5 = topk_accuracy(logits, targets, ks=(1, 5))
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1, n)
        top5_meter.update(top5, n)
        preds = logits.argmax(dim=1)
        for cls in range(num_classes):
            mask = targets == cls
            total_per_class[cls] += mask.sum().item()
            correct_per_class[cls] += (preds[mask] == cls).sum().item()
    per_class_acc = (correct_per_class.float() / total_per_class.clamp(min=1).float()).tolist()
    return loss_meter.avg, top1_meter.avg, top5_meter.avg, per_class_acc


def append_results_row(csv_path: str, row: Dict[str, object]) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_FIELDS})


def write_epoch_log(log_path: str, history: List[Dict[str, float]]) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in fieldnames})


def maybe_plot_curves(log_path: str, plot_path: str) -> None:
    try:
        from plot_curves import plot_from_csv
    except Exception as exc:
        print(f"[warn] could not import plot_curves: {exc}", flush=True)
        return
    try:
        plot_from_csv(log_path, plot_path)
    except Exception as exc:
        print(f"[warn] plotting failed: {exc}", flush=True)


def write_test_report(path: str, top1: float, top5: float, per_class_acc: List[float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"top1_acc: {top1:.4f}\n")
        f.write(f"top5_acc: {top5:.4f}\n")
        f.write("per_class_acc:\n")
        for cls, acc in enumerate(per_class_acc):
            f.write(f"  {cls:3d}: {acc:.4f}\n")
        order = sorted(range(len(per_class_acc)), key=lambda i: per_class_acc[i], reverse=True)
        f.write("\ntop_5_easiest_classes:\n")
        for cls in order[:5]:
            f.write(f"  {cls:3d}: {per_class_acc[cls]:.4f}\n")
        f.write("\ntop_5_hardest_classes:\n")
        for cls in order[-5:]:
            f.write(f"  {cls:3d}: {per_class_acc[cls]:.4f}\n")


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = auto_run_name(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}", flush=True)
    print(f"[info] run_name: {args.run_name}", flush=True)
    print(f"[info] args: {json.dumps(vars(args), indent=2)}", flush=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"[info] train batches: {len(train_loader)}  val batches: {len(val_loader)}  test batches: {len(test_loader)}", flush=True)

    model = build_resnet(args.model, pretrained=bool(args.pretrained), num_classes=NUM_CLASSES)
    model.to(device)
    print(f"[info] trainable params: {count_params(model):,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.run_name}_best.pth")
    log_path = os.path.join(args.log_dir, f"{args.run_name}.csv")
    plot_path = os.path.join(args.plot_dir, f"{args.run_name}_curves.png")
    test_report_path = os.path.join(args.log_dir, f"{args.run_name}_test_report.txt")

    best_val_acc = -1.0
    best_val_epoch = -1
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n[epoch {epoch}/{args.epochs}]", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        print(
            f"    train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_acc {val_top1*100:.2f}%  val_top5 {val_top5*100:.2f}%  "
            f"lr {current_lr:.5f}  time {elapsed:.1f}s",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_top1,
            "lr": current_lr,
        })

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            best_val_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_top1,
                "args": vars(args),
            }, ckpt_path)
            print(f"    [+] new best val_acc {best_val_acc*100:.2f}% -> saved {ckpt_path}", flush=True)

    write_epoch_log(log_path, history)
    print(f"[info] wrote epoch log: {log_path}", flush=True)

    print(f"\n[info] loading best checkpoint for test evaluation: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_top1, test_top5, per_class_acc = evaluate_with_class_acc(
        model, test_loader, criterion, device, NUM_CLASSES,
    )
    print(
        f"[test] loss {test_loss:.4f}  top1 {test_top1*100:.2f}%  top5 {test_top5*100:.2f}%",
        flush=True,
    )
    write_test_report(test_report_path, test_top1, test_top5, per_class_acc)
    print(f"[info] wrote test report: {test_report_path}", flush=True)

    row = {
        "run_name": args.run_name,
        "model": args.model,
        "init": "pretrained" if args.pretrained else "scratch",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "best_val_acc": round(best_val_acc, 6),
        "best_val_epoch": best_val_epoch,
        "test_acc": round(test_top1, 6),
        "test_top5_acc": round(test_top5, 6),
    }
    append_results_row(args.save_csv, row)
    print(f"[info] appended row to {args.save_csv}", flush=True)

    maybe_plot_curves(log_path, plot_path)
    print(f"[info] wrote plot: {plot_path}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
