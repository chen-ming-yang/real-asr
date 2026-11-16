#!/bin/bash
# Launch RNN-T Stage A training: frozen encoder, train predictor + joiner
set -e

CONDA_ENV="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PROJECT_DIR="/cmy/cmy/real-asr"
CONFIG="configs/train_rnnt_stageA.yaml"
LOG_FILE="logs/rnnt_stageA.log"

cd "$PROJECT_DIR"
mkdir -p logs exp/rnnt_stageA

echo "Starting RNN-T Stage A training (frozen encoder)..."
echo "Config: $CONFIG"
echo "Log: $LOG_FILE"
echo "Loss weights: rnnt=3.0, ctc=0.1"
echo "Encoder source: exp/stage1_icefall/epoch_50.pt"
echo ""

PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/model:$PYTHONPATH" \
nohup "$CONDA_ENV" train/train.py --config "$CONFIG" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Monitoring: tail -f $LOG_FILE"
