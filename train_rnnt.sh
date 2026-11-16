#!/bin/bash
# =============================================================================
# RNN-T Stage A 训练脚本
# 用途：冻结 Encoder，训练 Predictor + Joiner
# 用法：
#   ./train_rnnt.sh                    # 默认参数启动
#   ./train_rnnt.sh --resume           # 从最新 checkpoint 恢复
#   ./train_rnnt.sh --batch-dur 1600   # 调整 batch duration
#   ./train_rnnt.sh --workers 4        # 调整 dataloader workers
#   ./train_rnnt.sh --fg               # 前台运行（方便调试）
# =============================================================================
set -e

# === 默认参数 ===
PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
PROJECT_DIR="/cmy/cmy/real-asr"
CONFIG="configs/train_rnnt_stageA.yaml"
EXP_DIR="exp/rnnt_stageA_v2"
LOG_FILE="logs/rnnt_stageA_v2.log"

# 可通过命令行覆盖的训练参数
BATCH_DUR=1600        # max_duration_per_batch (原800，显存够可以加大)
NUM_WORKERS=16        # dataloader workers (原0，加速数据加载)
GRAD_ACCUM=2          # gradient accumulation steps (原4，减少可加速)
LR=0.0005             # learning rate
RESUME=false          # 是否从 checkpoint 恢复
FOREGROUND=true       # 是否前台运行
CHECKPOINT=""         # 指定恢复的 checkpoint 路径

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
            echo "Options:"
            echo "  --batch-dur N     max_duration_per_batch (default: 1600)"
            echo "  --workers N       dataloader num_workers (default: 4)"
            echo "  --grad-accum N    gradient accumulation steps (default: 2)"
            echo "  --lr FLOAT        learning rate (default: 0.0005)"
            echo "  --resume          从最新 checkpoint 恢复训练"
            echo "  --checkpoint PATH 从指定 checkpoint 恢复"
            echo "  --config PATH     配置文件路径"
            echo "  --exp-dir PATH    实验输出目录"
            echo "  --log PATH        日志文件路径"
            echo "  --fg              前台运行（不用 nohup）"
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
        # 自动找最新 checkpoint
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
echo "RNN-T Stage A Training"
echo "============================================================"
echo "Config:           $CONFIG"
echo "Exp dir:          $EXP_DIR"
echo "Log:              $LOG_FILE"
echo "Batch duration:   $BATCH_DUR"
echo "Num workers:      $NUM_WORKERS"
echo "Grad accum:       $GRAD_ACCUM"
echo "Learning rate:    $LR"
echo "Resume:           $RESUME"
echo "Foreground:       $FOREGROUND"
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
