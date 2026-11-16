#!/bin/bash
# Finetune CTC model on AISHELL-1 train set
# 基于 exp/ctc_L/epoch_10.pt (WenetSpeech L baseline, CER 29.78%)

set -euo pipefail

CONDA_PATH="/cmy/cmy/miniconda3"
ENV_NAME="qwen3-asr"
PROJECT_DIR="/cmy/cmy/real-asr"

source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_SHM_DIR=/tmp

cd "${PROJECT_DIR}"

mkdir -p exp/finetune_aishell
mkdir -p logs

echo "Starting AISHELL finetune at $(date)"
echo "Resuming from: exp/ctc_L/epoch_10.pt"
echo "Data: data/manifests/aishell/train (120098 utterances)"

torchrun --nproc_per_node=1 \
    train/train.py \
    --config configs/train_finetune_aishell.yaml \
    2>&1 | tee logs/finetune_aishell.log

echo "Training finished at $(date)"
