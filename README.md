# AI_Assignment3 — ResNet on CIFAR-100

Programming Assignment 3 for *Introduction to AI Programming*, 2026 Spring.

Train a ResNet-based image classifier on CIFAR-100 with pretrained vs scratch
comparison, learning-rate scheduling, data augmentation, and learning-curve
visualization.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate            # Windows PowerShell
pip install -r requirements.txt
```

CUDA 11.8+ / PyTorch 2.x with a GPU is recommended.

## Quick smoke test (CPU OK)

```bash
python train.py \
    --model resnet18 --pretrained 0 --epochs 1 \
    --batch_size 32 --lr 0.01 \
    --optimizer sgd --scheduler none \
    --num_workers 0 --run_name smoke \
    --save_csv results/results.csv
```

CIFAR-100 (~170 MB) will be downloaded to `./data` on the first run.

## Full experiments (GPU)

```bash
bash run_resnet_experiments.sh
```

This runs the four required configurations described in the assignment and
writes one row per run to `results/results.csv` plus per-epoch logs and
learning-curve plots under `results/`.

## Outputs

- `results/results.csv` — one row per experiment with val/test accuracy and config
- `results/logs/<run_name>.csv` — per-epoch train loss, val loss, val accuracy
- `results/plots/<run_name>_curves.png` — learning curves
- `checkpoints/<run_name>_best.pth` — best validation-accuracy checkpoint (ignored by git)

## Report

```bash
cd report
bash build.sh        # → report.pdf and report.docx
```

Requires `pdflatex` and `pandoc` on PATH.

## Files

```
train.py                       main training script
plot_curves.py                 standalone learning-curve plotter
run_resnet_experiments.sh      batch experiment runner
src/dataset.py                 CIFAR-100 loaders, augmentation, train/val split
src/model.py                   torchvision ResNet wrapper, final fc -> 100
src/utils.py                   seed, optimizer, scheduler, top-k accuracy
report/report.tex              English LaTeX report template
report/build.sh                pdflatex + pandoc build
docs/assignment.pdf            assignment specification
```
