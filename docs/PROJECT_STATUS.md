# Real-ASR 项目状态快照

> 更新时间: 2026-08-25  
> 用途: 跨会话恢复上下文；CTC 相关代码/模型**不可修改**。

---

## 一、总体目标

构建面向构音障碍语音的流式 ASR。当前两条线：

1. **CTC 线（冻结）**：已完成 WenetSpeech → AISHELL → 构音障碍微调；AISHELL CER 最优约 5.28%（beam+LM）
2. **RNN-T 线（独立新代码）**：从头联合训练，解决旧两阶段训练 blank dominance（CER 100%）问题

---

## 二、环境

| 项 | 值 |
|----|-----|
| Conda 环境 | `qwen3-asr` (`/cmy/cmy/miniconda3/envs/qwen3-asr`) |
| Python / Torch | 3.12 / 2.9.1+cu128 |
| k2 | 项目内 wheel（与 torch/cuda 匹配） |
| GPU | NVIDIA H800 80GB |
| Tokenizer | `data/tokenizer/sp_6000.model` (SP unigram, V=6000) |

常用启动：

```bash
cd /cmy/cmy/real-asr
export PYTHONPATH=/cmy/cmy/real-asr:$PYTHONPATH
/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python ...
```

---

## 三、CTC 线（不可修改）

详见 `docs/CTC_MODEL_REGISTRY.md`。

| Checkpoint | 说明 | AISHELL CER |
|------------|------|:-----------:|
| `exp/ctc_L/epoch_10.pt` | WenetSpeech CTC 基线 | — |
| `exp/stage1_icefall/epoch_50.pt` | Icefall 预训练 | 17.89% / 15.96% (+LM) |
| `exp/finetune_aishell/epoch_20.pt` | AISHELL 微调（最佳通识） | **5.87% / 5.28% (+LM)** |
| `exp/finetune_dysarthric/epoch_30.pt` | 构音障碍微调 | 构音障碍 test 33.10% / 31.44% |
| `exp/transformer_lm/best.pt` | Transformer LM | rescore 用 |

相关代码目录：`model/`、`train/`、`configs/train_*.yaml`、`eval_ctc.py`、`eval_dysarthric.py` — **不要改**。

---

## 四、RNN-T 线（新代码 `train_rnnt/`）

### 4.1 为什么重做

旧方案：冻结 CTC encoder → 只训 predictor/joiner → **blank dominance**，推理 CER≈100%。  
新方案：encoder + predictor + joiner **联合训练**，k2 pruned RNN-T + CTC 辅助 loss，不依赖旧 RNN-T checkpoint。

### 4.2 架构

- Encoder: Icefall Zipformer2（约 61.9M）
- Predictor: ConvPredictor / context_size=2（约 3.3M）
- Joiner: Additive + tanh（约 9.8M）
- CTC proj: 3.1M  
- **合计约 78M**

### 4.3 训练历程

| 阶段 | 配置 | 数据 | Checkpoint | AISHELL CER (greedy) |
|------|------|------|------------|:--------------------:|
| 1. 从头 M/S | `train_rnnt/config.yaml` | WenetSpeech M/S shards（151万） | `exp/rnnt_joint/epoch_30.pt` | 19.63% |
| 2. 继续 L | `train_rnnt/config_L.yaml` | WenetSpeech L 在线 fbank（1462万） | `exp/rnnt_joint_L/epoch_40.pt` | 11.20%（用 epoch_39 测） |
| 3. AISHELL 微调 | `train_rnnt/config_aishell.yaml` | AISHELL train（12万） | `exp/rnnt_joint_aishell/epoch_50.pt` / `avg_46_50.pt` | 5.85% / 5.55% (beam) |
| 4. 构音障碍微调 | `train_rnnt/config_dysarthric.yaml` | 构音障碍 train（78259） | `exp/rnnt_joint_dysarthric/avg_26_30.pt` | 构音障碍 test **29.23% / 27.47% (beam)** |

AISHELL 微调：从 `epoch_40.pt` resume，epoch 41→50，lr=5e-5。

### 4.4 关键修复（训练速度）

- 问题：旧 `ShardBucketingSampler` 跨 shard 组 batch → 频繁 `torch.load` 42MB → GPU util 0%
- 解决：`train_rnnt/shard_sampler.py`（shard-local）→ **0.1 → 4.0 batch/s**
- `/dev/shm` 仅 64MB：`num_workers=0`（shard）或 `4`（在线 opus）+ 避免 file_system 共享内存炸

### 4.5 解码优化（已完成）

1. **Checkpoint average**：`epoch_46`–`50` → `exp/rnnt_joint_aishell/avg_46_50.pt`
2. **Beam search**：`train_rnnt/eval.py` 支持 greedy + beam（beam=8）
3. 修正 RNN-T greedy（按 blank 推进帧）

平均 checkpoint：

```bash
python train_rnnt/average_checkpoints.py \
  --exp-dir exp/rnnt_joint_aishell \
  --epochs 46 47 48 49 50 \
  --output exp/rnnt_joint_aishell/avg_46_50.pt
```

评估命令：

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/cmy/cmy/real-asr:$PYTHONPATH \
/cmy/cmy/miniconda3/envs/qwen3-asr/bin/python -u train_rnnt/eval.py \
  --checkpoint exp/rnnt_joint_aishell/avg_46_50.pt \
  --config train_rnnt/config_aishell.yaml \
  --beam-size 8 --decode both
```

**最终结果（7176 条，2026-08-24）：**

| Checkpoint | CER greedy | CER beam(8) |
|------------|:----------:|:-----------:|
| `avg_46_50.pt` | **5.85%** | **5.55%** |

相对 `epoch_50` 旧 greedy 6.41%：checkpoint 平均 + 修正 greedy ≈ −0.56%；再加 beam ≈ 再 −0.30%。  
日志：`logs/eval_rnnt_aishell_avg_beam.log`

### 4.6 关键代码文件

| 路径 | 作用 |
|------|------|
| `train_rnnt/model.py` | RNNTModel |
| `train_rnnt/train.py` | 联合训练（支持 shard / online manifest） |
| `train_rnnt/shard_sampler.py` | Shard-local sampler |
| `train_rnnt/eval.py` | greedy + beam 评估 |
| `train_rnnt/average_checkpoints.py` | checkpoint 平均 |
| `train_rnnt/config.yaml` | M/S 从头训练 |
| `train_rnnt/config_L.yaml` | L 继续训练 |
| `train_rnnt/config_aishell.yaml` | AISHELL 微调 |
| `train_rnnt/config_dysarthric.yaml` | 构音障碍微调 |
| `run_train_rnnt_joint.sh` | 启动 M/S |
| `run_train_rnnt_L.sh` | 启动 L |
| `run_train_rnnt_aishell.sh` | 启动 AISHELL 微调 |
| `run_train_rnnt_dysarthric.sh` | 启动构音障碍微调 |

---

## 五、数据

| 数据 | 路径 | 备注 |
|------|------|------|
| WenetSpeech 原始音频 | `/cmy/cmy/WenetSpeech/` (~2.1T) | L 在线读 opus |
| M/S fbank shards | `data/fbank_shards/wenetspeech_ms/` (55G, 151万) | 预计算 |
| WenetSpeech L manifests | `data/manifests/wenetspeech/train/` | 1462万 supervisions |
| AISHELL | `data/manifests/aishell/{train,test}/` | train 12万 / test 7176 |
| 构音障碍 | `data/manifests/dysarthric/` | 来自 Qwen3-ASR `train.jsonl` |

磁盘：完整 L 预计算 fbank 约需 530G，当前可用不足，故 L 用在线特征。

---

## 六、结果对比（AISHELL test）

| 模型 | CER |
|------|:---:|
| CTC baseline | 17.89% |
| CTC + AISHELL + LM | **5.28%** |
| 旧 RNN-T（冻结 encoder） | ~100%（失败） |
| RNN-T M/S | 19.63% |
| RNN-T + L | 11.20% |
| RNN-T + L + AISHELL（epoch_50, 旧 greedy） | 6.41% |
| **RNN-T avg(46–50) greedy** | **5.85%** |
| **RNN-T avg(46–50) beam=8** | **5.55%** |

---

## 七、下一步候选

1. ~~等 `avg_46_50` 评估跑完，记录最终 CER~~ ✅ greedy 5.85% / beam 5.55%
2. ~~构音障碍微调 RNN-T~~ ✅ avg_26_30：greedy **29.23%** / beam **27.47%**（优于 CTC 33.10% / 31.44%）
3. RNN-T + **LM rescore**（复用 `exp/transformer_lm/best.pt`）
4. SpecAugment / 数据增强进一步压低构音障碍 CER

---

## 八、常用监控

```bash
# 训练
tail -f logs/train_rnnt_aishell.log
tail -f logs/train_rnnt_joint_L.log

# 当前评估
tail -f logs/eval_rnnt_aishell_avg_beam.log

# GPU
nvidia-smi
```
