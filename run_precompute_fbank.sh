#!/bin/bash
# =============================================================================
# Fbank 特征预计算脚本
# 用途：从 WenetSpeech opus 文件预提取 80-dim log-mel fbank + tokenize
# 策略：按录音文件分组，每个 worker 整文件读入内存后切片，避免 opus seek 开销
#
# 用法：
#   ./run_precompute_fbank.sh                       # 默认参数
#   ./run_precompute_fbank.sh --num-workers 64      # 增加并行度
#   ./run_precompute_fbank.sh --shard-size 2000     # 调整 shard 大小
# =============================================================================
set -e

# === Conda 环境 ===
CONDA_BASE="/cmy/cmy/miniconda3"
CONDA_ENV="qwen3-asr"
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"

# === 项目路径 ===
PROJECT_DIR="/cmy/cmy/real-asr"
SCRIPT="/cmy/cmy/real-asr/scripts/precompute_fbank_mp.py"

# === 默认参数 ===
MANIFEST_DIR="/cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms"
OUTPUT_DIR="/cmy/cmy/real-asr/data/fbank_shards/wenetspeech_ms"
TOKENIZER="/cmy/cmy/real-asr/data/tokenizer/sp_6000.model"
NUM_WORKERS=32
SHARD_SIZE=1000
MIN_DURATION=0.5
MAX_DURATION=20.0
LOG_FILE="/cmy/cmy/real-asr/logs/precompute_fbank.log"

# === 解析命令行参数 ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --manifest-dir)  MANIFEST_DIR="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --tokenizer)     TOKENIZER="$2"; shift 2 ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --shard-size)    SHARD_SIZE="$2"; shift 2 ;;
        --min-duration)  MIN_DURATION="$2"; shift 2 ;;
        --max-duration)  MAX_DURATION="$2"; shift 2 ;;
        --log)           LOG_FILE="$2"; shift 2 ;;
        --conda-env)     CONDA_ENV="$2"; PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"; shift 2 ;;
        --help|-h)
            echo "用法: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --manifest-dir PATH   Lhotse manifest 目录 (default: data/manifests/wenetspeech/train_ms)"
            echo "  --output-dir PATH     输出 shard 目录 (default: data/fbank_shards/wenetspeech_ms)"
            echo "  --tokenizer PATH      SentencePiece model 路径 (default: data/tokenizer/sp_6000.model)"
            echo "  --num-workers N       并行 worker 数 (default: 32)"
            echo "  --shard-size N        每个 shard 的 utterance 数 (default: 1000)"
            echo "  --min-duration FLOAT  最短时长过滤 (default: 0.5)"
            echo "  --max-duration FLOAT  最长时长过滤 (default: 20.0)"
            echo "  --log PATH            日志文件 (default: logs/precompute_fbank.log)"
            echo "  --conda-env NAME      Conda 环境名 (default: qwen3-asr)"
            echo "  -h, --help            显示帮助"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# === 初始化环境 ===
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

# 激活 conda（兼容不同 shell 初始化方式）
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# === 环境变量 ===
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# === 检查依赖 ===
echo "============================================================"
echo "Fbank Feature Extraction"
echo "============================================================"
echo "Python:         $PYTHON"
echo "Conda env:      $CONDA_ENV"
echo "Manifest dir:   $MANIFEST_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "Tokenizer:      $TOKENIZER"
echo "Num workers:    $NUM_WORKERS"
echo "Shard size:     $SHARD_SIZE"
echo "Duration:       [${MIN_DURATION}s, ${MAX_DURATION}s]"
echo "Log:            $LOG_FILE"
echo "============================================================"
echo ""

$PYTHON -c "import torch, torchaudio, sentencepiece, soundfile, tqdm; print('Dependencies OK')" || {
    echo "ERROR: Missing dependencies. Install: pip install torch torchaudio sentencepiece soundfile tqdm"
    exit 1
}

# === 启动预计算 ===
echo "Starting fbank extraction..."
echo ""

$PYTHON "$SCRIPT" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --tokenizer "$TOKENIZER" \
    --num-workers "$NUM_WORKERS" \
    --shard-size "$SHARD_SIZE" \
    --min-duration "$MIN_DURATION" \
    --max-duration "$MAX_DURATION" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Fbank extraction complete!"
echo "Output: $OUTPUT_DIR"
echo "Shards: $(ls "$OUTPUT_DIR"/*.pt 2>/dev/null | wc -l)"
echo "============================================================"
