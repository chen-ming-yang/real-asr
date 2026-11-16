#!/bin/bash
# RNN-T 在 WenetSpeech-L 上继续训练（在线特征提取）
set -eo pipefail

cd /cmy/cmy/real-asr

export CUDA_VISIBLE_DEVICES=0
export TORCH_SHM_DIR=/tmp
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== RNN-T training on WenetSpeech-L (online fbank) ==="
echo "Config: train_rnnt/config_L.yaml"
echo "Resume from: exp/rnnt_joint/epoch_30.pt"
echo "Exp dir: exp/rnnt_joint_L"
echo "$(date)"

mkdir -p exp/rnnt_joint_L logs

PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PYTHONPATH="/cmy/cmy/real-asr:${PYTHONPATH:-}" \
nohup $PYTHON train_rnnt/train.py \
    --config train_rnnt/config_L.yaml \
    > logs/train_rnnt_joint_L.log 2>&1 &

echo $! > .rnnt_train_pid
echo "Training started, PID=$(cat .rnnt_train_pid)"
echo "Log: logs/train_rnnt_joint_L.log"
echo "Monitor: tail -f logs/train_rnnt_joint_L.log"
