#!/bin/bash
# RNN-T AISHELL 微调（基于 rnnt_joint_L/epoch_39.pt）
set -eo pipefail

cd /cmy/cmy/real-asr

export CUDA_VISIBLE_DEVICES=0
export TORCH_SHM_DIR=/tmp
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== RNN-T AISHELL finetune ==="
echo "Config: train_rnnt/config_aishell.yaml"
echo "Resume from: exp/rnnt_joint_L/epoch_40.pt"
echo "Exp dir: exp/rnnt_joint_aishell"
echo "$(date)"

mkdir -p exp/rnnt_joint_aishell logs

PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PYTHONPATH="/cmy/cmy/real-asr:${PYTHONPATH:-}" \
nohup $PYTHON train_rnnt/train.py \
    --config train_rnnt/config_aishell.yaml \
    > logs/train_rnnt_aishell.log 2>&1 &

echo $! > .rnnt_train_pid
echo "Training started, PID=$(cat .rnnt_train_pid)"
echo "Log: logs/train_rnnt_aishell.log"
echo "Monitor: tail -f logs/train_rnnt_aishell.log"
