import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    epochs: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_loss, val_loss, val_acc


def plot_from_csv(log_path: str, out_path: str, title: str = "") -> None:
    epochs, train_loss, val_loss, val_acc = read_log(log_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, train_loss, marker="o", label="train loss")
    ax_loss.plot(epochs, val_loss, marker="s", label="val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    val_acc_pct = [a * 100 for a in val_acc]
    ax_acc.plot(epochs, val_acc_pct, marker="^", color="tab:green", label="val accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot learning curves from a per-epoch CSV log.")
    p.add_argument("--log", type=str, required=True, help="per-epoch CSV log path")
    p.add_argument("--out", type=str, required=True, help="output PNG path")
    p.add_argument("--title", type=str, default="", help="optional figure title")
    args = p.parse_args()
    plot_from_csv(args.log, args.out, args.title)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
