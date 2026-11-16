#!/bin/bash
# RNN-T 构音障碍微调（模仿 CTC finetune_dysarthric）
# 起点: exp/rnnt_joint_aishell/avg_46_50.pt
set -eo pipefail

cd /cmy/cmy/real-asr

export CUDA_VISIBLE_DEVICES=0
export TORCH_SHM_DIR=/tmp
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== RNN-T dysarthric finetune ==="
echo "Config: train_rnnt/config_dysarthric.yaml"
echo "Resume from: exp/rnnt_joint_aishell/avg_46_50.pt"
echo "Data: data/manifests/dysarthric/train (78259 utts)"
echo "Exp dir: exp/rnnt_joint_dysarthric"
echo "$(date)"

mkdir -p exp/rnnt_joint_dysarthric logs

PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PYTHONPATH="/cmy/cmy/real-asr:${PYTHONPATH:-}" \
nohup $PYTHON -u train_rnnt/train.py \
    --config train_rnnt/config_dysarthric.yaml \
    >| logs/train_rnnt_dysarthric.log 2>&1 &

echo $! >| .rnnt_train_pid
echo "Training started, PID=$(cat .rnnt_train_pid)"
echo "Log: logs/train_rnnt_dysarthric.log"
echo "Monitor: tail -f logs/train_rnnt_dysarthric.log"
