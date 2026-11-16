#!/bin/bash
# RNN-T 从头训练启动脚本
# 联合 CTC + RNN-T，不依赖预训练 checkpoint
set -eo pipefail

cd /cmy/cmy/real-asr

export CUDA_VISIBLE_DEVICES=0
export TORCH_SHM_DIR=/tmp
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting RNN-T joint training from scratch ==="
echo "Config: train_rnnt/config.yaml"
echo "Exp dir: exp/rnnt_joint"
echo "$(date)"

mkdir -p exp/rnnt_joint

PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PYTHONPATH="/cmy/cmy/real-asr:$PYTHONPATH" \
nohup $PYTHON train_rnnt/train.py \
    --config train_rnnt/config.yaml \
    > logs/train_rnnt_joint.log 2>&1 &

echo $! > .rnnt_train_pid
echo "Training started, PID=$(cat .rnnt_train_pid)"
echo "Log: logs/train_rnnt_joint.log"
echo "Monitor: tail -f logs/train_rnnt_joint.log"
