#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_CSV="${SAVE_CSV:-results/results.csv}"
DATA_DIR="${DATA_DIR:-./data}"

mkdir -p results/logs results/plots checkpoints

run () {
    local name=$1
    local model=$2
    local pretrained=$3
    local optimizer=$4
    local scheduler=$5
    local lr=$6
    echo
    echo "=================================================================="
    echo "  RUN: ${name}"
    echo "  model=${model} pretrained=${pretrained} optimizer=${optimizer} scheduler=${scheduler} lr=${lr} epochs=${EPOCHS}"
    echo "=================================================================="
    ${PYTHON} train.py \
        --model "${model}" \
        --pretrained "${pretrained}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${lr}" \
        --epochs "${EPOCHS}" \
        --optimizer "${optimizer}" \
        --scheduler "${scheduler}" \
        --num_workers "${NUM_WORKERS}" \
        --data_dir "${DATA_DIR}" \
        --save_csv "${SAVE_CSV}" \
        --run_name "${name}"
}

run r18_scratch_sgd_step       resnet18 0 sgd  step       0.1
run r18_pretrained_sgd_step    resnet18 1 sgd  step       0.01
run r18_scratch_adam_cosine    resnet18 0 adam cosine     0.001
run r18_pretrained_adam_cosine resnet18 1 adam cosine     0.001

echo
echo "All runs complete. Aggregate results: ${SAVE_CSV}"
