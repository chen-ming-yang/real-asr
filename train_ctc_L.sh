#!/bin/bash
# =============================================================================
# CTC Loss 大规模训练脚本 (WenetSpeech L 集 10000h)
# 用途：纯 CTC loss 训练 Encoder + CTC Head，为后续 RNN-T 提供更强的 encoder
# 用法：
#   ./train_ctc_L.sh                    # 默认参数启动
#   ./train_ctc_L.sh --resume           # 从最新 checkpoint 恢复
#   ./train_ctc_L.sh --batch-dur 1600   # 调整 batch duration
#   ./train_ctc_L.sh --fg               # 前台运行
# =============================================================================
set -e

# === 默认参数 ===
PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PROJECT_DIR="/cmy/cmy/real-asr"
CONFIG="configs/train_ctc_L.yaml"
EXP_DIR="exp/ctc_L"
LOG_FILE="logs/ctc_L.log"

# 训练超参数
BATCH_DUR=1200        # max_duration_per_batch
NUM_WORKERS=4         # dataloader workers
GRAD_ACCUM=4          # gradient accumulation (大数据集用大 effective batch)
LR=0.0001             # learning rate (大数据集用较小 lr)
RESUME=false
FOREGROUND=false
CHECKPOINT=""

# === 解析命令行参数 ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-dur)    BATCH_DUR="$2"; shift 2 ;;
        --workers)      NUM_WORKERS="$2"; shift 2 ;;
        --grad-accum)   GRAD_ACCUM="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --resume)       RESUME=true; shift ;;
        --checkpoint)   CHECKPOINT="$2"; RESUME=true; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --exp-dir)      EXP_DIR="$2"; shift 2 ;;
        --log)          LOG_FILE="$2"; shift 2 ;;
        --fg)           FOREGROUND=true; shift ;;
        --help|-h)
            echo "用法: $0 [OPTIONS]"
            echo ""
            echo "CTC 大规模训练 (WenetSpeech L 集 10000h)"
            echo ""
            echo "Options:"
            echo "  --batch-dur N     max_duration_per_batch (default: 1200)"
            echo "  --workers N       dataloader num_workers (default: 4)"
            echo "  --grad-accum N    gradient accumulation steps (default: 4)"
            echo "  --lr FLOAT        learning rate (default: 0.0001)"
            echo "  --resume          从最新 checkpoint 恢复训练"
            echo "  --checkpoint PATH 从指定 checkpoint 恢复"
            echo "  --config PATH     配置文件路径"
            echo "  --exp-dir PATH    实验输出目录"
            echo "  --log PATH        日志文件路径"
            echo "  --fg              前台运行"
            echo "  -h, --help        显示帮助"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"
mkdir -p logs "$EXP_DIR"

# === 构建额外参数 ===
EXTRA_ARGS=""
EXTRA_ARGS="$EXTRA_ARGS --override data.num_workers=$NUM_WORKERS"
EXTRA_ARGS="$EXTRA_ARGS --override data.bucketing_sampler.max_duration_per_batch=$BATCH_DUR"
EXTRA_ARGS="$EXTRA_ARGS --override training.optimizer.grad_accum_steps=$GRAD_ACCUM"
EXTRA_ARGS="$EXTRA_ARGS --override training.optimizer.lr=$LR"

# === 恢复训练 ===
if [ "$RESUME" = true ]; then
    if [ -n "$CHECKPOINT" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --resume $CHECKPOINT"
    else
        LATEST=$(ls -t "$EXP_DIR"/epoch_*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --resume $LATEST"
            echo "Resuming from: $LATEST"
        else
            echo "No checkpoint found in $EXP_DIR, starting from scratch"
        fi
    fi
fi

# === 打印配置 ===
echo "============================================================"
echo "CTC Large-Scale Training (WenetSpeech L, 10000h)"
echo "============================================================"
echo "Config:           $CONFIG"
echo "Exp dir:          $EXP_DIR"
echo "Log:              $LOG_FILE"
echo "Batch duration:   $BATCH_DUR s"
echo "Num workers:      $NUM_WORKERS"
echo "Grad accum:       $GRAD_ACCUM"
echo "Learning rate:    $LR"
echo "Resume:           $RESUME"
echo "Foreground:       $FOREGROUND"
echo ""
echo "Data:             data/manifests/wenetspeech/train (L set)"
echo "Supervisions:     14,625,227 utterances"
echo "Duration:         ~10005 hours"
echo "Loss:             CTC only (rnnt_weight=0, ctc_weight=1.0)"
echo "Encoder init:     exp/stage1_icefall/epoch_50.pt"
echo "============================================================"
echo ""

# === 环境变量 ===
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/model:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4

# === 启动训练 ===
CMD="$PYTHON train/train.py --config $CONFIG $EXTRA_ARGS"

if [ "$FOREGROUND" = true ]; then
    echo "Running in foreground..."
    echo "Command: $CMD"
    echo ""
    $CMD 2>&1 | tee "$LOG_FILE"
else
    echo "Running in background..."
    echo "Command: $CMD"
    echo ""
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "PID: $PID"
    echo "$PID" > "$EXP_DIR/train.pid"
    echo ""
    echo "Monitor:  tail -f $LOG_FILE"
    echo "Stop:     kill $PID"
    echo "GPU:      nvidia-smi"
fi
