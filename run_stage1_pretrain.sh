#!/bin/bash
# Real-ASR Stage 1 预训练启动脚本
# 在任务机（有 GPU）上直接运行即可

set -e

# ===== 配置区 =====
CONDA_PATH="/cmy/cmy/miniconda3"
ENV_NAME="qwen3-asr"
PROJECT_DIR="/cmy/cmy/real-asr"
CONFIG="configs/train_stage1.yaml"
NUM_GPUS=4          # 根据任务机 GPU 数量调整
LOG_DIR="${PROJECT_DIR}/logs"
EXP_DIR="${PROJECT_DIR}/exp/stage1_pretrain"

# ===== 防重复启动 =====
LOCK_FILE="${PROJECT_DIR}/.train_stage1.lock"

cleanup() {
    rm -f "${LOCK_FILE}"
}
trap cleanup EXIT INT TERM

if [ -f "${LOCK_FILE}" ]; then
    EXISTING_PID=$(cat "${LOCK_FILE}")
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "ERROR: Training already running (PID: ${EXISTING_PID})"
        echo "If you want to restart, run: kill ${EXISTING_PID}"
        exit 1
    else
        echo "WARNING: Stale lock file found (PID ${EXISTING_PID} not running). Removing."
        rm -f "${LOCK_FILE}"
    fi
fi

echo $$ > "${LOCK_FILE}"

# ===== 环境初始化 =====
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${EXP_DIR}"

# ===== 检查环境 =====
echo "============================================"
echo "  Real-ASR Stage 1 Pretrain"
echo "============================================"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "k2: $(python -c 'import k2; print(k2.__dev_version__)')"
echo "Config: ${CONFIG}"
echo "Num GPUs: ${NUM_GPUS}"
echo "============================================"

# 检查 GPU 是否可用
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "${GPU_COUNT}" -eq 0 ]; then
    echo "ERROR: No GPU available! Please run on a GPU node."
    exit 1
fi

# 如果实际 GPU 数量少于配置，自动调整
if [ "${GPU_COUNT}" -lt "${NUM_GPUS}" ]; then
    echo "WARNING: Requested ${NUM_GPUS} GPUs but only ${GPU_COUNT} available. Using ${GPU_COUNT}."
    NUM_GPUS=${GPU_COUNT}
fi

# ===== 数据检查 =====
MANIFEST_DIR="${PROJECT_DIR}/data/manifests/wenetspeech/train"
if [ ! -f "${MANIFEST_DIR}/recordings.jsonl" ]; then
    echo "ERROR: Training data not found at ${MANIFEST_DIR}"
    echo "Please run: python data/prepare_wenetspeech.py --subsets L"
    exit 1
fi

SUP_COUNT=$(wc -l < "${MANIFEST_DIR}/supervisions.jsonl")
echo "Training data: ${SUP_COUNT} supervisions"

# ===== 启动训练 =====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/stage1_pretrain_${TIMESTAMP}.log"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Starting training..."
echo "Log file: ${LOG_FILE}"
echo ""

if [ "${NUM_GPUS}" -gt 1 ]; then
    # 多 GPU 分布式训练
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=12345 \
        train/train.py \
        --config "${CONFIG}" \
        2>&1 | tee "${LOG_FILE}"
else
    # 单 GPU 训练
    python train/train.py \
        --config "${CONFIG}" \
        2>&1 | tee "${LOG_FILE}"
fi

echo ""
echo "Training finished at $(date)"
echo "Checkpoints saved to: ${EXP_DIR}"
