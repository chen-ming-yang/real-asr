#!/bin/bash
# =============================================================================
# 预先建立 fbank shard index 缓存
# 只需要跑一次，之后训练启动时直接加载缓存（秒级）
#
# 用法：
#   bash /cmy/cmy/real-asr/run_build_shard_index.sh
# =============================================================================
set -e

PYTHON="/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python"
SHARD_DIR="/cmy/cmy/real-asr/data/fbank_shards/wenetspeech_ms"

export PYTHONPATH="/cmy/cmy/real-asr:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "Building shard index for: $SHARD_DIR"
echo "Cache will be saved to: $SHARD_DIR/.shard_index.npz"
echo ""

$PYTHON /cmy/cmy/real-asr/scripts/build_shard_index.py --shard-dir "$SHARD_DIR"

echo ""
echo "Training will now start instantly on next run."
