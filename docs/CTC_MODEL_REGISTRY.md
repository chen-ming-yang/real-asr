# CTC 模型注册表（不可修改）

本文件记录当前已训练完成的 CTC 系列模型和相关代码，**这些内容不得修改**。

## 已训练模型


| 模型              | 路径                                    | 说明                         |
| --------------- | ------------------------------------- | -------------------------- |
| CTC-L baseline  | `exp/ctc_L/epoch_10.pt`               | WenetSpeech-L 上预训练的 CTC 基线 |
| CTC-L + AISHELL | `exp/finetune_aishell/epoch_20.pt`    | AISHELL-1 微调               |
| CTC-L + 构音障碍    | `exp/finetune_dysarthric/epoch_30.pt` | 最终构音障碍语音微调 (最佳 CTC 模型)     |
| Transformer LM  | `exp/transformer_lm/`                 | CTC + LM rescore 使用的语言模型   |


## 评估结果

### AISHELL Test (7176 条)


| 模型                      | Checkpoint                            | CER (greedy)     | CER (beam+LM)            |
| ----------------------- | ------------------------------------- | ---------------- | ------------------------ |
| CTC-L baseline (stage1) | `exp/stage1_icefall/epoch_50.pt`      | 17.89%           | 15.96%                   |
| CTC-L + AISHELL 微调      | `exp/finetune_aishell/epoch_20.pt`    | **5.87%**        | **5.28%**                |
| RNN-T 联合训练 (M/S)        | `exp/rnnt_joint/epoch_30.pt`          | 19.63% (greedy)  | —                        |
| RNN-T + WenetSpeech-L   | `exp/rnnt_joint_L/epoch_39.pt`        | 11.20% (greedy)  | —                        |
| RNN-T + L + AISHELL 微调  | `exp/rnnt_joint_aishell/epoch_50.pt`  | 6.41% (旧 greedy) | —                        |
| RNN-T avg(46–50)        | `exp/rnnt_joint_aishell/avg_46_50.pt` | **5.85%**        | **5.55%** (beam=8, 无 LM) |




### 构音障碍 Test


| 模型                 | Checkpoint                               | CER (greedy) | CER (beam / +LM)    |
| ------------------ | ---------------------------------------- | ------------ | ------------------- |
| CTC-L + 构音障碍微调     | `exp/finetune_dysarthric/epoch_30.pt`    | 33.10%       | 31.44% (+LM)        |
| **RNN-T + 构音障碍微调** | `exp/rnnt_joint_dysarthric/avg_26_30.pt` | **29.23%**   | **27.47%** (beam=8) |


RNN-T 按说话人（avg_26_30, beam=8）：


| 说话人 | 样本数 | CER greedy | CER beam |
| --- | --- | ---------- | -------- |
| 01  | 638 | 14.46%     | 13.48%   |
| 04  | 567 | 37.18%     | 35.38%   |
| 06  | 516 | 50.85%     | 48.64%   |
| 08  | 724 | 34.68%     | 32.53%   |
| 09  | 725 | 35.18%     | 33.81%   |
| 12  | 779 | 24.52%     | 22.35%   |
| 20  | 405 | 19.34%     | 17.44%   |


CTC 按说话人（beam+LM）：


| 说话人 | 样本数 | CER greedy | CER beam+LM |
| --- | --- | ---------- | ----------- |
| 01  | 638 | 16.12%     | 15.48%      |
| 04  | 567 | 34.28%     | 32.41%      |
| 06  | 516 | 60.91%     | 59.09%      |
| 08  | 724 | 43.84%     | 41.16%      |
| 09  | 725 | 39.67%     | 37.96%      |
| 12  | 779 | 26.90%     | 24.95%      |
| 20  | 405 | 22.54%     | 22.10%      |




## RNN-T 模型（新，独立代码）


| 项目                 | 内容                                                                 |
| ------------------ | ------------------------------------------------------------------ |
| M/S Checkpoint     | `exp/rnnt_joint/epoch_30.pt` (CER 19.63%)                          |
| L Checkpoint       | `exp/rnnt_joint_L/epoch_40.pt` (epoch_39 CER 11.20%)               |
| AISHELL Checkpoint | `exp/rnnt_joint_aishell/epoch_50.pt` / `avg_46_50.pt`              |
| 完整状态文档             | `docs/PROJECT_STATUS.md`                                           |
| 代码目录               | `train_rnnt/`                                                      |
| 架构                 | Zipformer2 + ConvPredictor + Additive Joiner                       |
| 参数量                | 78M (encoder 61.9M + predictor 3.3M + joiner 9.8M + ctc_proj 3.1M) |
| 训练策略               | 从头联合训练 (CTC + pruned RNN-T)，不冻结 encoder                            |
| 训练数据               | WenetSpeech M/S → 继续 L (1462万条, 在线 fbank)                          |
| 关键改进               | Shard-Local Sampler 提速 26x (0.1→4.0 batch/s)                       |




## 相关代码（不可修改）


| 文件                                       | 用途                                   |
| ---------------------------------------- | ------------------------------------ |
| `model/ctc_head.py`                      | CTC 投影头                              |
| `model/transducer.py`                    | Transducer 结构定义 (Predictor + Joiner) |
| `model/icefall_zipformer.py`             | Icefall Zipformer 编码器实现              |
| `model/icefall_encoder_wrapper.py`       | 编码器包装器                               |
| `model/zipformer.py`                     | 原始 Zipformer 编码器                     |
| `model/transformer_lm.py`                | Transformer 语言模型                     |
| `train/train.py`                         | 原始训练脚本                               |
| `train/train_lm.py`                      | LM 训练脚本                              |
| `eval_ctc.py`                            | CTC 评估脚本                             |
| `eval_dysarthric.py`                     | 构音障碍评估脚本                             |
| `configs/train_ctc_L.yaml`               | CTC-L 训练配置                           |
| `configs/train_finetune_aishell.yaml`    | AISHELL 微调配置                         |
| `configs/train_finetune_dysarthric.yaml` | 构音障碍微调配置                             |
| `configs/train_lm.yaml`                  | LM 训练配置                              |
| `configs/model.yaml`                     | 模型结构定义                               |




## Tokenizer

- 路径: `data/tokenizer/sp_6000.model`
- 类型: SentencePiece unigram, vocab_size=6000



## 数据

- WenetSpeech fbank shards: `data/fbank_shards/wenetspeech_ms/`
- AISHELL manifests: `data/manifests/aishell/`
- 构音障碍 manifests: `data/manifests/dysarthric/`

