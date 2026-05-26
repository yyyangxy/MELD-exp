# MELD STL 当前实验设置、代码修改与结果总结

生成时间：2026-05-21 10:04 CST

本文档总结 2026-05-19 至 2026-05-21 期间围绕 MELD STL 实验做过的主要代码修改、当前实验设置、已完成实验结果，以及这些结果应该如何解释。重点包括：

- S3：Dialogue-level Task-Incremental MELD-STL，端到端文本输入。
- S4：Dialogue-level Modality-Incremental MELD-STL，端到端 text/audio 输入。
- MELD 文本读取问题的修正，以及为什么部分旧结果需要重跑。

## 1. 当前任务背景

当前实验主要围绕 MELD 上的连续学习设置：

1. Task-incremental STL：任务按顺序到来，不同阶段学习不同标签任务。
   - Stage 1：sentiment
   - Stage 2：emotion
   - Stage 3：shift

2. Modality-incremental STL：任务标签保持为 emotion，输入模态按顺序增加。
   - Stage 1：text
   - Stage 2：text + audio
   - 当前端到端 S4 runner 暂时只实现 text/audio，visual 还没有端到端接入。

3. 主要目标：
   - 验证当前方法是否优于传统 sequential fine-tuning / KD baseline。
   - 比较回放与不回放。
   - 分析 replay KD、old head freeze、view-specific heads 等组件是否真正有效。

## 2. 关键代码修改

### 2.1 MELD 文本读取修正

修改文件：

- `src/data/meld_csv.py`

问题：

- `/data2/yangxy/dataset/MELD/MELD.Raw` 中部分文本存在 cp1252 smart quote 被错误读成控制字符的情况。
- 例如 smart quote / dash / ellipsis 会变成 `\x91`, `\x92`, `\x93`, `\x94`, `\x96`, `\x97`, `\x85`。
- 这不会改变标签和 split，但会改变 XLM-R tokenizer 的输入，因此会影响端到端 text 实验。

当前修正：

```python
_MOJIBAKE_TRANSLATION = str.maketrans(
    {
        "\x91": "'",
        "\x92": "'",
        "\x93": '"',
        "\x94": '"',
        "\x96": "-",
        "\x97": "-",
        "\x85": "...",
    }
)

def normalize_meld_text(text: str) -> str:
    return text.translate(_MOJIBAKE_TRANSLATION).strip()
```

并在读取 CSV 时使用：

```python
utterance=normalize_meld_text(row["Utterance"])
```

影响判断：

- 旧的端到端 text 结果不能作为最终结果。
- 标签、Dialogue_ID、Utterance_ID、split 本身没有发现错位。
- fixed-feature 结果是否受影响，取决于当初 feature extraction 是否使用了同一份错误文本；当前新端到端实验已经使用修正后的文本读取。

### 2.2 S3 dialogue task runner 改为端到端文本

主要文件：

- `scripts/run_dialogue_task_stl.py`
- `src/train/dialogue_text_task_runner.py`

当前行为：

- 输入为 dialogue 中多条 utterance。
- tokenizer 输出形状为 `(batch, max_dialogue_len, token_len)`。
- 送入 XLM-R 前展平为 `(batch * max_dialogue_len, token_len)`。
- 再 reshape 回 dialogue 序列。
- 对 context-free 之外的方法，继续经过 BiLSTM。

当前 S3 支持的方法包括：

- `dlg_seq_ft`
- `dlg_seq_kd`
- `dlg_random_replay`
- `dlg_sa_cmd_no_replay`
- `text_task_sa_cmd`
- `text_task_sa_cmd_replay_kd`
- `text_task_sa_cmd_freeze_old_heads`
- `text_task_sa_cmd_replay_kd_freeze_old_heads`

新增或规范化的 CLI 参数：

- `--gpu-id`
- `--epochs`
- `--batch-size`
- `--grad-accum-steps`
- `--memory-per-class`
- `--replay-strategy`
- `--fp16`
- `--replay-batch-kd`
- `--freeze-old-heads`

### 2.3 S3 replay batch KD 与 old head freeze

当前 replay 逻辑：

- 训练新任务时，可以从旧任务 memory 中取 replay batch。
- replay batch 可以只做监督，也可以额外做 teacher KD / relation loss。
- `--replay-batch-kd` 控制 replay batch 上是否应用旧 teacher 的蒸馏约束。

当前 freeze 逻辑：

- `--freeze-old-heads` 会在旧任务学完后冻结对应分类头。
- 目的：减少旧任务 head 在新任务训练阶段被漂移。
- 但从修正文本后的结果看，freeze old heads 并没有稳定带来提升。

### 2.4 S4 fixed-feature modality runner 的 view-specific heads

主要文件：

- `src/models/dialogue_model.py`
- `src/train/dialogue_modality_runner.py`
- `scripts/run_dialogue_modality_stl.py`

修改点：

- `DialogueMultimodalSTLModel.forward()` 新增 `head_name` 参数。
- 固定特征版 S4 可以使用不同 view head，例如：
  - `emotion_text`
  - `emotion_text_audio`
  - `emotion_full`

新增方法：

- `dlg_modality_sa_cmd_view_heads`
- `dlg_modality_sa_cmd_view_heads_freeze`

说明：

- 这是 fixed-feature S4 的改动。
- 但当前更关注的是新的端到端 text/audio S4。

### 2.5 新增 S4 end-to-end text/audio runner

主要文件：

- `src/train/dialogue_modality_e2e_runner.py`
- `scripts/run_dialogue_modality_e2e_stl.py`

当前模型结构：

- Text encoder：XLM-R large，本地路径 `/data2/yangxy/models/xlm-roberta-large`
- Audio encoder：raw waveform Conv1d encoder
- Speaker embedding
- Fusion MLP
- Dialogue BiLSTM
- Emotion classification head 或 view-specific heads

当前支持方法：

- `dlg_e2e_mod_seq_ft`
- `dlg_e2e_mod_seq_kd`
- `dlg_e2e_modality_sa_cmd`
- `dlg_e2e_modality_sa_cmd_view_heads`
- `dlg_e2e_modality_sa_cmd_view_heads_freeze`

当前支持模态阶段：

- `text`
- `text_audio`

Audio 读取方式：

- 从 MELD `.mp4` 文件中通过 ffmpeg 解码 16k mono PCM waveform。
- 缓存路径默认可设为 `outputs/audio_waveforms_16k_s4_e2e`。
- 缓存的是 raw waveform tensor，不是模型特征，因此仍然属于端到端训练 audio encoder。

### 2.6 S4 e2e 速度和显存相关修正

修改文件：

- `src/train/dialogue_modality_e2e_runner.py`

修正 1：text-only 阶段不再运行 audio encoder。

修正前：

- 即使 `active_modalities=["text"]`，代码仍先跑 raw audio CNN，再把 audio embedding 清零。
- 这会造成 text stage 的无效计算。

修正后：

```python
if "audio" in active_modality_set:
    audio = batch["audio"].reshape(batch_size * max_len, -1)
    audio_emb = self.audio_encoder(audio).reshape(batch_size, max_len, -1)
    audio_emb = audio_emb * batch["audio_enabled"].view(batch_size, 1, 1)
else:
    audio_emb = text_emb.new_zeros(batch_size, max_len, self.audio_dim)
```

修正 2：text-only 阶段 Dataset 不再构造大尺寸零 audio tensor。

修正前：

- text-only 阶段仍然给每个 utterance 构造长度为 `max_audio_seconds * sample_rate` 的零音频。

修正后：

- text-only 阶段只构造长度为 1 的占位 tensor。
- 这不改变结果，因为模型在 text-only 阶段不会读取 `batch["audio"]`。

## 3. 当前实验设置

### 3.1 S3 Dialogue Task-STL

配置：

- Config：`configs/dialogue_task_stl_v2.yaml`
- 输入：end-to-end text
- Text encoder：XLM-R large
- Dialogue encoder：BiLSTM
- Epochs：30 per stage
- Batch size：2
- Grad accumulation：4
- Effective batch size：8
- Max length：128
- Optimizer lr：`2e-5`
- Weight decay：`0.01`
- FP16：enabled
- Replay strategy：random

重要说明：

- 每个任务阶段训练 30 epoch，因此完整三阶段训练是 90 个 epoch。
- 但每个阶段的数据不同，不应理解为同一个数据集训练 90 epoch。

### 3.2 S4 Dialogue Modality-STL E2E

配置：

- Config：`configs/dialogue_modality_stl_v2.yaml`
- 输入：end-to-end text/audio
- Text encoder：XLM-R large
- Audio encoder：raw waveform Conv1d
- Dialogue encoder：BiLSTM
- Epochs：30 per modality stage
- Batch size：1
- Grad accumulation：8
- Effective batch size：8
- Max text length：64
- Max audio seconds：4
- FP16：enabled
- Audio cache：`outputs/audio_waveforms_16k_s4_e2e`

当前风险点：

- S4 当前统一 lr 为 `0.001`，对 XLM-R large 端到端训练可能过大。
- 已完成的 S4 baseline 结果偏低，后续应考虑 encoder lr 和新增模块 lr 分组。

## 4. 已完成实验结果

### 4.1 S3：修正文本后的重跑结果

这些结果使用修正后的 MELD 文本读取逻辑。

结果文件：

- `outputs/runs/dialogue_text_task_stl/20260520_183819_s3_dialogue_e2e_core_rerun_fixed_text_gpu8_20260520_dlg_seq_kd/results/dialogue_text_task_stl_results.csv`
- `outputs/runs/dialogue_text_task_stl/20260520_195752_s3_dialogue_e2e_core_rerun_fixed_text_gpu8_20260520_text_task_sa_cmd_replay_kd_freeze_old_heads/results/dialogue_text_task_stl_results.csv`
- `outputs/runs/dialogue_text_task_stl/20260520_183612_bilstm_full_kd1.0_mem100/results/dialogue_text_task_stl_results.csv`
- `outputs/runs/dialogue_text_task_stl/20260521_111001_s3_baseline_er_seed13_gpu9_20260521/results/dialogue_text_task_stl_results.csv`
- `outputs/runs/dialogue_text_task_stl/20260521_124819_s3_baseline_icarl_seed13_gpu3_20260521_seed13/results/dialogue_text_task_stl_results.csv`

| Method | Sent acc | Emo acc | Shift acc | Avg acc | Avg weighted-F1 | 主要观察 |
|---|---:|---:|---:|---:|---:|---|
| `dlg_seq_kd` | 0.6717 | 0.5713 | 0.5799 | 0.6076 | 0.6002 | baseline |
| `dlg_er` | 0.6717 | 0.5307 | 0.5799 | 0.5941 | 0.5943 | random ER，emotion 明显偏低 |
| `dlg_icarl` | 0.6662 | 0.5844 | 0.6096 | **0.6201** | **0.6093** | iCaRL-style，目前单 seed 略高 |
| `text_task_sa_cmd` | 0.6538 | 0.5768 | 0.6068 | 0.6125 | 0.6039 | shift 比 baseline 高 |
| `text_task_sa_cmd_replay_kd` | 0.6717 | 0.5768 | 0.6011 | 0.6165 | **0.6084** | 与 iCaRL-style 非常接近 |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 0.6648 | 0.5757 | 0.6040 | 0.6148 | 0.6042 | freeze 没有稳定提升 |

更详细指标：

| Method | Task | Acc | Weighted-F1 | Macro-F1 | Shift positive F1 | Forgetting | Retention |
|---|---|---:|---:|---:|---:|---:|---:|
| `dlg_seq_kd` | sentiment | 0.6717 | 0.6679 | 0.6310 | - | 0.0029 | 0.9956 |
| `dlg_seq_kd` | emotion | 0.5713 | 0.5565 | 0.3353 | - | 0.0106 | 0.9812 |
| `dlg_seq_kd` | shift | 0.5799 | 0.5761 | 0.5739 | 0.6245 | 0.0358 | 0.9415 |
| `dlg_er` | sentiment | 0.6717 | 0.6702 | 0.6361 | - | 0.0073 | 0.9892 |
| `dlg_er` | emotion | 0.5307 | 0.5352 | 0.3338 | - | 0.0272 | 0.9517 |
| `dlg_er` | shift | 0.5799 | 0.5776 | 0.5758 | 0.6178 | 0.0064 | 0.9891 |
| `dlg_icarl` | sentiment | 0.6662 | 0.6611 | 0.6247 | - | 0.0016 | 0.9975 |
| `dlg_icarl` | emotion | 0.5844 | 0.5600 | 0.3283 | - | 0.0003 | 0.9995 |
| `dlg_icarl` | shift | 0.6096 | 0.6068 | 0.6049 | 0.6480 | 0.0000 | 1.0000 |
| `text_task_sa_cmd` | sentiment | 0.6538 | 0.6467 | 0.6051 | - | 0.0089 | 0.9864 |
| `text_task_sa_cmd` | emotion | 0.5768 | 0.5587 | 0.3328 | - | 0.0000 | 1.0000 |
| `text_task_sa_cmd` | shift | 0.6068 | 0.6062 | 0.6050 | 0.6313 | 0.0143 | 0.9770 |
| `text_task_sa_cmd_replay_kd` | sentiment | 0.6717 | 0.6671 | 0.6323 | - | 0.0016 | 0.9976 |
| `text_task_sa_cmd_replay_kd` | emotion | 0.5768 | 0.5597 | 0.3319 | - | 0.0000 | 1.0000 |
| `text_task_sa_cmd_replay_kd` | shift | 0.6011 | 0.5985 | 0.5966 | 0.6394 | 0.0059 | 0.9902 |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | sentiment | 0.6648 | 0.6591 | 0.6211 | - | 0.0007 | 0.9989 |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | emotion | 0.5757 | 0.5525 | 0.3190 | - | 0.0020 | 0.9964 |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | shift | 0.6040 | 0.6011 | 0.5992 | 0.6429 | 0.0054 | 0.9911 |

当前结论：

- 修正文本后，当前单 seed 最高的是 `dlg_icarl`，Avg acc = 0.6201，Avg weighted-F1 = 0.6093；`text_task_sa_cmd_replay_kd` 非常接近，Avg acc = 0.6165，Avg weighted-F1 = 0.6084。
- 相比 `dlg_seq_kd`，`dlg_icarl` 提升约 +0.0125 avg acc / +0.0092 avg weighted-F1，`text_task_sa_cmd_replay_kd` 提升约 +0.0089 avg acc / +0.0083 avg weighted-F1。
- replay KD / exemplar replay 的贡献主要体现在更低 forgetting 和更好的 shift 保持。
- `dlg_er` 的 Avg acc = 0.5941，Avg weighted-F1 = 0.5943，低于 `dlg_seq_kd`，主要短板是 emotion acc 只有 0.5307；单纯随机 ER 没有带来收益。
- `dlg_icarl` 使用 `prototype_nearest` exemplar selection + replay batch KD，在 emotion、shift 上都高于 `dlg_seq_kd`，但它和 `text_task_sa_cmd_replay_kd` 只差 0.0009，需要多 seed 才能判断谁更稳。
- freeze old heads 当前没有稳定贡献；它保护旧任务很强，但可能限制后续任务适配。

### 4.2 S3：修正文本前的历史结果

以下结果来自文本修正之前，只能作为探索参考，不建议作为最终论文结果。

代表性旧结果：

| Method | Sent acc | Emo acc | Shift acc | Avg acc |
|---|---:|---:|---:|---:|
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 0.6772 | 0.5976 | 0.6110 | 0.6246 |

旧结果解释：

- 该结果曾是 S3 dialogue 最好结果。
- 但由于当时 text 输入存在 mojibake 问题，不能与修正文本后的结果直接混合比较。
- 如果论文或正式汇报使用 S3，需要以修正文本后的重跑为准。

### 4.3 S4：当前已完成和仍在运行的结果

截至 2026-05-21 10:04，S4 还有两个进程仍在运行：

- GPU 3：`dlg_e2e_mod_seq_ft` / `dlg_e2e_mod_seq_kd`
- GPU 9：`dlg_e2e_modality_sa_cmd_view_heads` / `dlg_e2e_modality_sa_cmd_view_heads_freeze`

已写出完整 CSV 的结果：

结果文件：

- `outputs/runs/dialogue_modality_e2e_stl/20260520_183219_bilstm_ln_kd1.0_cmd1.0_mem100/results/dialogue_modality_e2e_results.csv`

| Method | Stage | Train modalities | Eval modalities | Acc | Weighted-F1 | Macro-F1 | Final avg |
|---|---|---|---|---:|---:|---:|---:|
| `dlg_e2e_mod_seq_ft` | text | text | text | 0.3533 | 0.3253 | 0.1463 | - |
| `dlg_e2e_mod_seq_ft` | text_audio | text+audio | text | 0.3471 | 0.3231 | 0.1523 | 0.3240 |
| `dlg_e2e_mod_seq_ft` | text_audio | text+audio | text+audio | 0.3494 | 0.3249 | 0.1537 | 0.3240 |

当前判断：

- 这个 S4 baseline 明显偏低。
- 当前不能据此说明 audio 有帮助。
- 更可能说明 S4 e2e runner 的超参还不合适，尤其是统一 `lr=0.001` 对 XLM-R large 过大。
- 需要等 `dlg_e2e_mod_seq_kd`、`view_heads`、`view_heads_freeze` 完整跑完，再做最终判断。

## 5. 当前主要结论

### 5.1 S3 结论

修正文本后，当前最稳的 S3 dialogue 方法是：

```text
text_task_sa_cmd_replay_kd
```

它相对 `dlg_seq_kd` 的优势：

- Avg acc：0.6084 vs 0.6002
- Shift acc：0.6011 vs 0.5799
- Forgetting 更低，尤其 shift 从 0.0358 降到 0.0059。

但优势幅度不算很大，需要后续多 seed 验证。

### 5.2 freeze old heads 的当前判断

修正文本后，`text_task_sa_cmd_replay_kd_freeze_old_heads` 没有复现旧的 0.6246。

当前结果：

- `text_task_sa_cmd_replay_kd`：0.6084
- `text_task_sa_cmd_replay_kd_freeze_old_heads`：0.6042

因此目前不能声称 freeze old heads 是稳定有效组件。更合理的表述是：

- freeze old heads 可以减少旧 head 漂移。
- 但它可能限制新任务阶段共享表示和 head 的联合适配。
- 当前实验里 replay KD 的收益比 freeze old heads 更明确。

### 5.3 S4 当前判断

S4 e2e 还不能下最终结论。

已完成的 `dlg_e2e_mod_seq_ft` 很低，说明：

- 统一 lr 可能不合适。
- raw audio encoder 和 XLM-R large 一起端到端训练需要更谨慎的参数分组。
- 当前 S4 的训练速度和显存压力都明显高于 S3。

## 6. 建议的下一步

### 6.1 先等当前 S4 跑完

不要在当前 S4 未完成时下结论。等以下方法都出 CSV：

- `dlg_e2e_mod_seq_kd`
- `dlg_e2e_modality_sa_cmd_view_heads`
- `dlg_e2e_modality_sa_cmd_view_heads_freeze`

再统一比较。

### 6.2 S4 下一轮建议改 optimizer 参数

建议把 S4 e2e 从单一 lr 改成 parameter groups：

- XLM-R encoder：`2e-5`
- audio encoder / fusion / BiLSTM / heads：`1e-4` 或 `5e-4`

原因：

- XLM-R large 对 `1e-3` 非常敏感。
- 新增模块可以用更大学习率，但不应让 pretrained encoder 同步用这么大的 lr。

### 6.3 S3 后续建议

正式结果建议至少补：

- `dlg_seq_kd`
- `dlg_er`
- `dlg_icarl`
- `text_task_sa_cmd`
- `text_task_sa_cmd_replay_kd`
- `text_task_sa_cmd_replay_kd_freeze_old_heads`

每个方法做多个 seed，否则当前 0.6 左右的小幅差距不够稳。尤其是 `dlg_icarl` 和 `text_task_sa_cmd_replay_kd` 当前只差 0.0009，单 seed 不能说明显著优劣。

## 7. 可复现实验命令记录

### 7.1 S3 core rerun

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 8 --config configs/dialogue_task_stl_v2.yaml --methods dlg_seq_kd text_task_sa_cmd_replay_kd_freeze_old_heads --epochs 30 --batch-size 2 --grad-accum-steps 4 --replay-strategy random --replay-batch-kd --freeze-old-heads --fp16 --run-name s3_dialogue_e2e_core_rerun_fixed_text_gpu8_20260520
```

### 7.2 S3 extra rerun

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 7 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --replay-strategy random --replay-batch-kd --fp16 --run-name s3_dialogue_e2e_extra_rerun_fixed_text_gpu7_20260520
```

### 7.3 S4 e2e baselines

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_modality_e2e_stl.py --gpu-id 3 --config configs/dialogue_modality_stl_v2.yaml --methods dlg_e2e_mod_seq_ft dlg_e2e_mod_seq_kd --epochs 30 --batch-size 1 --grad-accum-steps 8 --max-length 64 --max-audio-seconds 4 --audio-cache-root outputs/audio_waveforms_16k_s4_e2e --fp16 --run-name s4_e2e_text_audio_baselines_gpu3_20260520_fixed_text
```

### 7.4 S4 e2e current methods

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_modality_e2e_stl.py --gpu-id 9 --config configs/dialogue_modality_stl_v2.yaml --methods dlg_e2e_modality_sa_cmd_view_heads dlg_e2e_modality_sa_cmd_view_heads_freeze --epochs 30 --batch-size 1 --grad-accum-steps 8 --max-length 64 --max-audio-seconds 4 --audio-cache-root outputs/audio_waveforms_16k_s4_e2e --fp16 --run-name s4_e2e_text_audio_ours_gpu9_20260520_fixed_text
```

## 8. 注意事项

1. 旧结果和修正文本后的结果不能混合当作同一实验条件。
2. 当前 S4 e2e 还在跑，本文档中的 S4 结论只是阶段性观察。
3. S3 的当前优势主要来自 replay KD，而不是 freeze old heads。
4. S4 的下一步重点不是继续盲跑更多方法，而是先修 optimizer 参数组和学习率。
5. 如果后续要写论文结果，建议至少做 3 seeds，并统一使用修正文本后的数据读取逻辑。
