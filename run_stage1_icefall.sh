#!/bin/bash
# Launch icefall Zipformer2 training for comparison
# Run this AFTER the current v10 training finishes or is stopped.

set -e

CONDA_ENV="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PROJECT_DIR="/cmy/cmy/real-asr"
CONFIG="configs/train_stage1_icefall.yaml"
LOG_FILE="logs/stage1_icefall.log"

cd "$PROJECT_DIR"
mkdir -p logs exp/stage1_icefall

echo "Starting icefall Zipformer2 training..."
echo "Config: $CONFIG"
echo "Log: $LOG_FILE"
echo "Model: icefall Zipformer2 (~79.6M params)"
echo ""

PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/model:$PYTHONPATH" \
nohup "$CONDA_ENV" train/train.py --config "$CONFIG" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Monitoring: tail -f $LOG_FILE"
