# ASR 模型优化方案：RNN-T 重训 + 外置 LM

## 当前问题诊断

### 问题现象

icefall Zipformer2 模型训练 50 epoch 完成后：
- CTC greedy 解码 CER: **17.8%**（正常）
- RNN-T greedy 解码 CER: **100%**（全部输出为空）

### 根因分析

通过检查 `joiner.output_proj` 权重发现：
- blank token (id=0) 的 bias 极大（~15.0），远超其他 token（~0.0）
- `encoder_proj` 和 `predictor_proj` 的权重值极小（接近零）
- 这意味着 predictor 和 joiner 在训练过程中几乎没有被有效更新

### 原因

训练配置 `rnnt_weight: 1.0, ctc_weight: 0.3`，看似 RNN-T 权重更大，但实际上：

1. **CTC 梯度天然更强** — CTC loss 直接作用于 encoder 输出，梯度路径短且密集
2. **RNN-T 梯度被稀释** — pruned RNN-T loss 经过 joiner 的 T×U lattice 再反传，梯度分散
3. **Blank 是安全策略** — 当 encoder 被 CTC 快速优化后，joiner 发现输出 blank 总能得到低 loss（因为 RNN-T 的 blank 路径和 CTC 对齐一致），于是 predictor/joiner 收敛到 trivial solution
4. **结果** — encoder 学到了声学表征（CTC 好），但 predictor + joiner 退化为"永远输出 blank"

### 现有最佳结果

| 解码方式 | CER | 测试集 | 状态 |
|---------|-----|--------|------|
| CTC greedy | 29.78% | WenetSpeech test_net (24,774条) | 正式 baseline |
| CTC greedy | 15.1% | 14条训练集内样本 | 非正式参考 |
| CTC beam+LM | 10.7% | 14条训练集内样本 | 非正式参考 |
| RNN-T greedy | 100% | - | blank dominance + tanh饱和 |

---

## 解决方案

### 问题二：Joiner tanh 饱和（Stage A 训练发现）

**现象：** Stage A 训练 18 epoch 后 loss 从 50 降至 6.7，但 RNN-T greedy decode 仍然 100% CER（只输出 1-2 个随机 token）。

**诊断过程：**

1. 检查 joiner 权重：blank bias 从原始的 15 降到 1.06，看似改善
2. 逐帧分析 joiner 输出：blank 的 log-prob 仍然是 -0.00（≈100%），其他 token 是 -14 到 -18
3. 深入检查 joiner 内部计算：
   - `encoder_proj` 输出 norm=558，std=24.7
   - `predictor_proj` 输出 norm=502，std=22.2
   - 两者相加后 pre-tanh 值 std=36.15
   - **tanh 饱和率 95.1%**（几乎所有维度都被压到 ±1）
4. 训练过程中权重膨胀：`encoder_proj` norm 从 14.76 增长到 61.01（4.1x），`predictor_proj` 从 38.33 增长到 94.20（2.5x）

**根因：**

Joiner 使用 `tanh(encoder_proj(x) + predictor_proj(y))` 作为激活函数。训练过程中 proj 权重不断增大，导致：
- pre-tanh 值远超 tanh 线性区（|x|<1），进入完全饱和区
- tanh 输出几乎全是 ±1，梯度接近 0
- 改变 predictor 输入几乎不改变 tanh 输出（被 encoder 的大值淹没）
- output_proj 只能看到 ±1 的 sign pattern，无法区分不同 token

而 loss 仍能下降是因为 k2 pruned loss 中的 `simple_loss`（使用 `simple_am_proj/simple_lm_proj`，不经过 tanh）可以正常优化。

**修复方案：**

将 Joiner 的激活从 `tanh` 改为 `LayerNorm + ReLU`：
- LayerNorm 在激活前将值标准化到 mean=0, std=1，防止饱和
- ReLU 不会像 tanh 那样双侧压缩，梯度更稳定
- 这是 icefall 较新版本 joiner 的做法

修改文件：`model/transducer.py` 中的 `Joiner` 类

### 方案 A：外置 Transformer LM（快速提升 CTC 效果）

在不重训 ASR 模型的前提下，训练一个外置语言模型做 CTC beam search + LM rescore，利用语言先验纠正同音字错误。

**架构：**
- 6 层 Transformer decoder, d_model=512, 8 heads, ffn=2048
- 参数量：~45M
- 训练数据：WenetSpeech M+S transcript 文本

**预期收益：** CER 下降 3-5 个百分点（17.8% → 13-15%）

**资源：**
- 训练：3-4 GB 显存，4-6 小时
- 推理：额外 ~200 MB 显存 (fp16)
- 磁盘：~100 MB (fp16)

### 方案 B：RNN-T 重训（两阶段冻结策略）

通过调整 loss 权重（rnnt_weight: 3.0, ctc_weight: 0.1）重新训练，使 predictor + joiner 有效学习。

**Stage A — 冻结 encoder：**
- 加载 epoch_50.pt 的 encoder 权重（冻结）
- 仅训练 predictor + joiner（11.2M 参数）
- LR: 0.001, 15-20 epoch
- 资源：4-8 GB 显存，10-12h

**Stage B — 解冻 encoder，全模型微调：**
- 从 Stage A 最优 checkpoint 加载
- 小 LR (3e-5) 微调全模型
- 5-10 epoch
- 资源：16-25 GB 显存，15-20h

**预期收益：** RNN-T 解码 CER 达到或优于 CTC（预期 15-18%），加 LM 后可进一步提升

---

## 执行计划

| 步骤 | 内容 | 预计时间 |
|------|------|---------|
| 1 | 提取 LM 训练文本 | 10 min |
| 2 | 训练 Transformer LM | 4-6h |
| 3 | CTC beam search + LM 评估 | 30 min |
| 4 | RNN-T Stage A（冻结 encoder） | 10-12h |
| 5 | RNN-T Stage B（解冻微调） | 15-20h |
| 6 | RNN-T 解码评估 | 30 min |
| 7 | 最终对比 | - |

---

## 最终对比结果

### 外置 LM 结果

- LM 架构：6-layer Transformer, d_model=512, 8 heads, 28.3M params
- 训练数据：WenetSpeech M+S transcript 1.35M 句（去重后）
- 训练时间：~11 分钟（10 epochs, 440 steps/epoch）
- 最终 PPL：train=22.5, val=26.5

### 解码对比

#### WenetSpeech test_net（24,774 samples，正式评估）

| 解码方式 | CER | 备注 |
|---------|-----|------|
| CTC greedy (baseline) | **29.78%** | epoch 50，全量 test_net |
| CTC beam (size=10) + LM rescore (w=0.3) | - | 待跑全量 |
| RNN-T greedy | - | Stage A/B 训练中 |
| RNN-T beam + LM shallow fusion | - | 待评估 |

#### 非正式测试（14 samples from train set）

| 解码方式 | CER | 备注 |
|---------|-----|------|
| CTC greedy | 15.1% | 训练集内样本，参考价值有限 |
| CTC beam (size=10) + LM rescore (w=0.3) | **10.7%** | 提升 4.4% 绝对值 |

### 关键改善案例

| 样本 | REF | CTC Greedy | CTC+LM |
|------|-----|-----------|--------|
| 1 | 我就搬出去住 | 我就搬出去起助 (33.3%) | 我就搬出去住 (0%) |
| 3 | 怎么能叫混呢我这叫来激发来感悟 | ...这想来吉发... (20%) | ...这叫来吉发... (13.3%) |
| 10 | 这个每个人都做出了 | ...都做 (22.2%) | ...都做错了 (11.1%) |
| 12 | 那美妙的歌声常常在草原上飘得很远很远 | ...飘得得很远... (11.1%) | ...飘得很远... (5.6%) |
