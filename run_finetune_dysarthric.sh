#!/bin/bash
# 在构音障碍训练集上微调 CTC 模型
# 基于 exp/finetune_aishell/epoch_20.pt

set -euo pipefail

CONDA_PATH="/cmy/cmy/miniconda3"
ENV_NAME="qwen3-asr"
PROJECT_DIR="/cmy/cmy/real-asr"

source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT_DIR}"

mkdir -p exp/finetune_dysarthric
mkdir -p logs

echo "Starting dysarthric finetune at $(date)"
echo "Resuming from: exp/finetune_aishell/epoch_20.pt"
echo "Data: data/manifests/dysarthric/train (78259 utterances)"

torchrun --nproc_per_node=1 \
    train/train.py \
    --config configs/train_finetune_dysarthric.yaml \
    2>&1 | tee logs/finetune_dysarthric.log

echo "Training finished at $(date)"
