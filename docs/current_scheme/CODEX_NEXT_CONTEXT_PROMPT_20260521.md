# MELD-STL 项目上下文 Prompt

生成时间：2026-05-21

这份文档用于喂给下一个对话框，让新对话能快速了解当前 MELD-STL 项目的背景、实现路径、实验结果、代码细节、输出格式和当前困难。

## 1. 项目背景

当前项目在 MELD 数据集上研究连续学习下的单任务学习扩展，重点不是传统一次性训练，而是让模型按阶段学习不同任务或不同模态。

数据集路径：

- 项目目录：`/data2/yangxy/MELD/MELD-exp`
- MELD Raw 数据：`/data2/yangxy/dataset/MELD/MELD.Raw`
- XLM-R large 本地模型：`/data2/yangxy/models/xlm-roberta-large`

主要实验设置：

- S3：Dialogue-level Task-Incremental MELD-STL
  - Stage 1：sentiment
  - Stage 2：emotion
  - Stage 3：shift
  - 当前主线是端到端 text 输入，不再调用固定提取的 text feature。

- S4：Dialogue-level Modality-Incremental MELD-STL
  - Stage 1：text
  - Stage 2：text + audio
  - 原计划还包括 full/text+audio+visual，但端到端 visual 尚未实现。
  - 当前 S4 端到端显存、速度和效果都有明显问题，暂时不作为主线推进。

- S5：Dialogue-level Text+Audio Task-Incremental MELD-STL
  - 目标是在多模态输入固定为 text+audio 的情况下做任务增量。
  - 任务顺序仍是 sentiment -> emotion -> shift。
  - 这是用户当前想替代 S4 继续推进的新方向。

## 2. 数据与任务划分

当前 S3/S5 使用固定 STL task split：

```text
/data2/yangxy/dataset/MELD/MELD.Raw/stl_task_splits
```

已确认日志中的 split 规模：

| Split | Sentiment dialogues | Emotion dialogues | Shift dialogues |
|---|---:|---:|---:|
| train | 346 | 346 | 346 |
| dev | 39 | 38 | 37 |
| test | 94 | 93 | 93 |

重要细节：

- 测试数据也按任务分成三个 task split。
- 评估时会带 task id，即 sentiment 样本用 sentiment head，emotion 样本用 emotion head，shift 样本用 shift head。
- 这是 task-incremental learning，不是 task-free/class-incremental learning。

## 3. 已发现并修正的重要问题

### 3.1 MELD 文本 mojibake 修正

文件：

```text
src/data/meld_csv.py
```

问题：

- MELD CSV 里部分 cp1252 smart quote/dash/ellipsis 被错误读成控制字符。
- 例如 `\x91`, `\x92`, `\x93`, `\x94`, `\x96`, `\x97`, `\x85`。
- 这会影响 XLM-R tokenizer 输入，因此影响所有端到端 text 实验。

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

影响：

- 2026-05-20 之前的端到端 text 结果只能作为探索参考。
- 后续正式 S3/S5 结果应使用修正文本后的重跑结果。

### 3.2 之前容易被忽略的实验细节

需要在后续讨论中主动检查这些点：

- 是否三个任务其实用了同一份数据。现在 S3/S5 已使用 fixed STL task split。
- 是否旧分类头在新任务训练时冻结。当前只有显式传 `--freeze-old-heads` 的方法会冻结旧 head。
- 是否训练和测试都带 task id。当前 task-incremental 设置会带 task id。
- 是否 replay batch 真正参与训练。S3 的 `dlg_er`、`dlg_icarl`、`text_task_sa_cmd_replay_kd` 等会用 memory/replay；纯 KD 不一定有 replay。
- iCaRL 当前已经实现 NME/prototype-style 分类器版本，结果目录名含 `icarl_nme`。
- S4 之前 text-only 阶段曾错误地跑了 audio encoder，已修正。
- S4 text-only dataset 曾构造大零 audio tensor，已修正为短占位 tensor。

## 4. 当前核心代码

### 4.1 S3 端到端 dialogue task runner

入口：

```text
scripts/run_dialogue_task_stl.py
```

核心实现：

```text
src/train/dialogue_text_task_runner.py
```

主要方法：

- `dlg_seq_ft`
- `dlg_seq_kd`
- `dlg_er`
- `dlg_icarl`
- `dlg_ewc`
- `dlg_mas`
- `dlg_si`
- `text_task_sa_cmd`
- `text_task_sa_cmd_replay_kd`
- `text_task_sa_cmd_freeze_old_heads`
- `text_task_sa_cmd_replay_kd_freeze_old_heads`

当前模型流程：

1. 输入一个 dialogue 的多条 utterance。
2. tokenized shape 为 `(batch, max_dialogue_len, token_len)`。
3. 展平成 `(batch * max_dialogue_len, token_len)` 输入 XLM-R large。
4. reshape 回 dialogue 序列。
5. context-free 之外的方法会接 BiLSTM。
6. 不同 task 使用不同分类头。

常用参数：

- `--gpu-id`
- `--seed`
- `--epochs`
- `--batch-size`
- `--grad-accum-steps`
- `--memory-per-class`
- `--replay-strategy random|prototype_nearest|diverse|hybrid`
- `--replay-batch-kd`
- `--freeze-old-heads`
- `--importance-max-batches`
- `--regularizer-scope all|non_encoder`
- `--si-xi`
- `--fp16`

### 4.2 S4 端到端 modality runner

入口：

```text
scripts/run_dialogue_modality_e2e_stl.py
```

核心实现：

```text
src/train/dialogue_modality_e2e_runner.py
```

当前支持：

- `dlg_e2e_mod_seq_ft`
- `dlg_e2e_mod_seq_kd`
- `dlg_e2e_modality_sa_cmd`
- `dlg_e2e_modality_sa_cmd_view_heads`
- `dlg_e2e_modality_sa_cmd_view_heads_freeze`

当前阶段：

- `text`
- `text_audio`

主要问题：

- 很慢。
- 显存占用高。
- 当前结果偏低。
- 端到端 audio 只是 raw waveform Conv1d encoder，还不是强 audio pretrained encoder。
- 统一学习率对 XLM-R large 可能过大。

### 4.3 S5 text+audio task-incremental runner

入口：

```text
scripts/run_dialogue_text_audio_task_e2e_stl.py
```

核心实现：

```text
src/train/dialogue_text_audio_task_e2e_runner.py
```

当前支持：

- `s5_e2e_ta_seq_ft`
- `s5_e2e_ta_seq_kd`
- `s5_e2e_ta_sa_cmd`

模型结构：

- Text encoder：XLM-R large
- Audio encoder：`RawAudioEncoder`
- Speaker embedding
- Fusion MLP
- Dialogue BiLSTM
- Task-specific heads：sentiment/emotion/shift

S5 当前没有 replay memory，实现的是：

- seq-ft：只训练当前任务 CE。
- seq-kd：新任务阶段对旧任务 head 做 KD。
- sa-cmd：在 KD 基础上加入 teacher confidence weighting 和 sample relation loss。

S5 输出文件：

```text
outputs/runs/dialogue_text_audio_task_e2e_stl/<timestamp>_<run_name>_<method>/results/dialogue_text_audio_task_e2e_results.csv
```

## 5. 方法说明

### 5.1 text_task_sa_cmd 是什么

`text_task_sa_cmd` 是当前 S3 主方法的核心版本，用于 dialogue-level task-incremental learning。

它的思想：

- 当前任务用 supervised CE 学习。
- 保存旧模型作为 teacher。
- 学新任务时，约束 student 在旧任务上的输出不要偏离 teacher。
- 蒸馏不是简单平均，而是用 teacher confidence 给样本加权。
- 同时加入 sample relation loss，约束 student/teacher embedding 之间的样本关系结构。

简化损失：

```text
L = L_CE(new task)
  + lambda_kd * L_KD(old tasks, confidence weights)
  + lambda_rel * L_relation(old tasks, confidence weights)
```

带 replay 的版本：

```text
text_task_sa_cmd_replay_kd
```

会从旧任务 memory 采 replay batch，并在 replay batch 上施加旧任务监督/KD 约束。

### 5.2 ConfidenceKD_old 的作用

Confidence KD 使用 teacher 对旧任务样本的预测置信度作为权重。

直觉：

- teacher 高置信的旧知识更可靠，应强约束。
- teacher 低置信样本可能本身困难或噪声大，应降低 KD 权重。

数学上：

```text
w_i = max_c softmax(z_teacher_i / T)_c
L_KD = mean_i w_i * KL(p_teacher_i || p_student_i)
```

这样可以减少低质量 teacher target 对 student 的负面影响。

### 5.3 iCaRL 当前实现

最开始的 iCaRL 近似于 replay + KD + prototype selection；之后已进一步加入 NME/prototype-style classifier。

当前结果中应优先看：

```text
outputs/runs/dialogue_text_task_stl/20260521_153224_s3_baseline_icarl_nme_seed13_gpu3_20260521_seed13/results/dialogue_text_task_stl_results.csv
```

而不是早期非 NME 版本。

## 6. 当前实验结果

### 6.1 S3 修正文本后的主要结果

| Method | Seed | Final avg weighted-F1 | Final avg acc | 结果路径 |
|---|---:|---:|---:|---|
| `dlg_seq_kd` | 13 | 0.6002 | 未单独写入旧格式 | `outputs/runs/dialogue_text_task_stl/20260520_183819_s3_dialogue_e2e_core_rerun_fixed_text_gpu8_20260520_dlg_seq_kd/results/dialogue_text_task_stl_results.csv` |
| `text_task_sa_cmd` | 13 | 0.6039 | 未单独写入旧格式 | `outputs/runs/dialogue_text_task_stl/20260520_183612_bilstm_full_kd1.0_mem100/results/dialogue_text_task_stl_results.csv` |
| `text_task_sa_cmd_replay_kd` random | 13 | 0.6084 | 未单独写入旧格式 | `outputs/runs/dialogue_text_task_stl/20260520_183612_bilstm_full_kd1.0_mem100/results/dialogue_text_task_stl_results.csv` |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 13 | 0.6042 | 未单独写入旧格式 | `outputs/runs/dialogue_text_task_stl/20260520_195752_s3_dialogue_e2e_core_rerun_fixed_text_gpu8_20260520_text_task_sa_cmd_replay_kd_freeze_old_heads/results/dialogue_text_task_stl_results.csv` |
| `dlg_er` | 13 | 0.5943 | 未单独写入旧格式 | `outputs/runs/dialogue_text_task_stl/20260521_111001_s3_baseline_er_seed13_gpu9_20260521/results/dialogue_text_task_stl_results.csv` |
| `dlg_icarl` early | 13 | 0.6093 | 未单独写入旧格式 | `outputs/runs/dialogue_text_task_stl/20260521_124819_s3_baseline_icarl_seed13_gpu3_20260521_seed13/results/dialogue_text_task_stl_results.csv` |
| `dlg_icarl` NME | 13 | 0.6147 | 0.6209 | `outputs/runs/dialogue_text_task_stl/20260521_153224_s3_baseline_icarl_nme_seed13_gpu3_20260521_seed13/results/dialogue_text_task_stl_results.csv` |
| `text_task_sa_cmd_replay_kd` prototype_nearest | 13 | 0.6079 | 0.6185 | `outputs/runs/dialogue_text_task_stl/20260521_154751_s3_ours_replay_kd_prototype_nearest_seed13_gpu2_20260521_seed13/results/dialogue_text_task_stl_results.csv` |

多 seed 结果：

| Method | Seed | Final avg weighted-F1 |
|---|---:|---:|
| `dlg_seq_kd` | 21 | 0.6186 |
| `text_task_sa_cmd_replay_kd` | 21 | 0.6221 |
| `dlg_seq_kd` | 42 | 0.6096 |
| `text_task_sa_cmd_replay_kd` | 42 | 0.6100 |

阶段性判断：

- 方法间差距很小，很多差异可能与随机种子同量级。
- `text_task_sa_cmd_replay_kd` 相比 `dlg_seq_kd` 有一定提升，但需要至少 3 seeds 报均值和方差。
- `dlg_icarl` NME 当前 seed13 表现最好之一，说明 prototype/NME 机制值得认真对比。
- `prototype_nearest` memory 对我们的方法没有明显超过 random replay。

### 6.2 S3 下界和上界

结果路径：

- 下界 seq-ft：`outputs/runs/dialogue_text_task_stl/20260521_153907_s3_task_bounds_gpu8_20260521_lower_seq_ft_seed13/results/dialogue_text_task_stl_results.csv`
- 上界 joint/multi-task：`outputs/runs/dialogue_text_task_stl/20260521_164507_s3_task_bounds_gpu8_20260521_upper_joint_bilstm_seed13/results/dialogue_text_task_stl_results.csv`

当前结果：

| Bound | Method | Seed | Final avg weighted-F1 | Final avg acc |
|---|---|---:|---:|---:|
| Lower | `dlg_seq_ft` | 13 | 0.4716 | 0.4805 |
| Upper | `hier_bilstm` joint | 13 | 0.5964 | 0.5987 |

注意：

- 这个 upper 是三任务 joint/multi-task 训练，多分类头，并不一定是真正理论上界。
- 它目前低于部分 CL 方法，说明 upper 实现/超参/训练方式需要重新检查，不能直接作为论文上界。

### 6.3 S4 当前结果

结果文件：

```text
outputs/runs/dialogue_modality_e2e_stl/20260520_183219_bilstm_ln_kd1.0_cmd1.0_mem100/results/dialogue_modality_e2e_results.csv
```

已完成 KD 结果：

| Method | Stage | Eval modalities | Acc | Weighted-F1 | Final avg |
|---|---|---|---:|---:|---:|
| `dlg_e2e_mod_seq_kd` | text | text | 0.3533 | 0.3253 | - |
| `dlg_e2e_mod_seq_kd` | text_audio | text | 0.3406 | 0.3150 | 0.3152 |
| `dlg_e2e_mod_seq_kd` | text_audio | text+audio | 0.3490 | 0.3153 | 0.3152 |

判断：

- S4 结果明显偏低，不正常。
- 可能原因包括：统一 lr 不适合 XLM-R large、raw audio encoder 太弱、S4 训练太慢导致调参成本高、modality incremental 本身比 S3/S5 更难。
- 当前建议暂时把 S4 放后面，先推进 S5 text+audio task-incremental。

## 7. 当前正在跑的程序

最近检查到两个相关进程：

```text
GPU 2:
python scripts/run_dialogue_task_stl.py --gpu-id 2 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-batch-kd --replay-strategy diverse --fp16 --run-name s3_ours_replay_kd_diverse_seed13_gpu2_20260521

GPU 3:
python scripts/run_dialogue_text_audio_task_e2e_stl.py --gpu-id 3 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods s5_e2e_ta_seq_ft s5_e2e_ta_seq_kd s5_e2e_ta_sa_cmd --epochs 30 --batch-size 1 --grad-accum-steps 8 --max-length 64 --max-audio-seconds 4 --audio-cache-root outputs/audio_waveforms_16k_s5_task_e2e --fp16 --run-name s5_e2e_text_audio_task_core_gpu3_20260521
```

S5 当前已经产生部分文件：

```text
outputs/runs/dialogue_text_audio_task_e2e_stl/20260521_172208_s5_e2e_text_audio_task_core_gpu3_20260521_s5_e2e_ta_seq_ft/
```

## 8. 结果存储格式

### 8.1 S3

路径格式：

```text
outputs/runs/dialogue_text_task_stl/<timestamp>_<run_name>/results/dialogue_text_task_stl_results.csv
```

常见字段：

- `method`
- `stage`
- `task`
- `accuracy`
- `weighted_f1`
- `macro_f1`
- `positive_f1_for_shift`
- `final_avg`
- `final_avg_weighted_f1`
- `final_avg_accuracy`
- `forgetting`
- `retention`

有些早期 CSV 只有 `final_avg`，后来整理目录后新增了 `final_avg_weighted_f1` 和 `final_avg_accuracy`。

### 8.2 S4

路径格式：

```text
outputs/runs/dialogue_modality_e2e_stl/<timestamp>_<run_name>/results/dialogue_modality_e2e_results.csv
```

字段：

- `method`
- `stage`
- `train_modalities`
- `eval_modalities`
- `accuracy`
- `weighted_f1`
- `macro_f1`
- `final_avg`
- `modality_forgetting`
- `modality_retention`
- `num_eval_dialogues`
- `num_eval_utterances`

### 8.3 S5

路径格式：

```text
outputs/runs/dialogue_text_audio_task_e2e_stl/<timestamp>_<run_name>_<method>/results/dialogue_text_audio_task_e2e_results.csv
```

字段：

- `method`
- `stage`
- `task`
- `accuracy`
- `weighted_f1`
- `macro_f1`
- `positive_f1_for_shift`
- `final_avg`
- `final_avg_weighted_f1`
- `final_avg_accuracy`
- `forgetting`
- `retention`

每个 run 还会记录：

```text
logs/run_parameters.json
config.yaml
config_latest.yaml
checkpoints/*.pt
```

## 9. 重要命令

### 9.1 S3 seed sweep

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 3 --seed 21 --config configs/dialogue_task_stl_v2.yaml --methods dlg_seq_kd text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-batch-kd --replay-strategy random --fp16 --run-name s3_dialogue_seed_sweep_gpu3_20260521_seed21
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 3 --seed 42 --config configs/dialogue_task_stl_v2.yaml --methods dlg_seq_kd text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-batch-kd --replay-strategy random --fp16 --run-name s3_dialogue_seed_sweep_gpu3_20260521_seed42
```

### 9.2 S3 baseline

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 9 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods dlg_er --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-strategy random --fp16 --run-name s3_baseline_er_seed13_gpu9_20260521
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 3 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods dlg_icarl --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-strategy prototype_nearest --replay-batch-kd --fp16 --run-name s3_baseline_icarl_nme_seed13_gpu3_20260521
```

### 9.3 S3 我们的方法不同 replay selection

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 2 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-batch-kd --replay-strategy diverse --fp16 --run-name s3_ours_replay_kd_diverse_seed13_gpu2_20260521
```

### 9.4 S5 text+audio task-incremental

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_text_audio_task_e2e_stl.py --gpu-id 3 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods s5_e2e_ta_seq_ft s5_e2e_ta_seq_kd s5_e2e_ta_sa_cmd --epochs 30 --batch-size 1 --grad-accum-steps 8 --max-length 64 --max-audio-seconds 4 --audio-cache-root outputs/audio_waveforms_16k_s5_task_e2e --fp16 --run-name s5_e2e_text_audio_task_core_gpu3_20260521
```

## 10. 当前困难和下一步

### 10.1 当前困难

1. S3 方法间差距太小。
   - `dlg_seq_kd` 和 `text_task_sa_cmd_replay_kd` 的差距经常只有 0.001 到 0.01。
   - 必须多 seed 才能说明方法有效。

2. iCaRL/NME 表现很强。
   - 这说明 prototype classifier 或 exemplar representation 对当前 task split 很有帮助。
   - 我们的方法需要明确和 iCaRL 的差异，不能只说 replay。

3. S4 端到端太慢且结果低。
   - text+audio stage 每轮耗时长。
   - 当前 weighted-F1 只有约 0.315，不适合作为主线结果。

4. EWC/MAS/SI 这类 regularization baseline 训练额外耗显存。
   - `regularizer_scope=all` 会把 encoder 也纳入重要性估计，更公平但更耗显存。
   - 曾出现 OOM。
   - 曾出现 `cudnn RNN backward can only be called in training mode`，原因是 importance 估计时模型处于 eval/RNN backward 状态不匹配，需要确认代码已修。

5. 上界实现需要复查。
   - 当前 joint upper weighted-F1 约 0.596，低于部分 CL 方法，不合理。
   - 可能是 upper 的数据、训练、sampler、评估或超参不一致。

### 10.2 下一步建议

优先级从高到低：

1. 等 S5 text+audio task-incremental 跑完，先看 text+audio 是否比纯 text S3 有收益。
2. 检查 S5 的速度和显存，确认 batch size 1 + grad accumulation 8 是否可长期跑。
3. 汇总 S3 seed13/21/42 的 `dlg_seq_kd` 与 `text_task_sa_cmd_replay_kd` 均值和标准差。
4. 复查 joint upper bound 的实现，使其成为可信上界。
5. 补齐 iCaRL NME、ER、EWC、MAS、SI 的同 seed 对比。
6. 如果继续 S4，需要先改 optimizer parameter groups：
   - XLM-R encoder lr：`2e-5`
   - audio/fusion/BiLSTM/head lr：`1e-4` 或 `5e-4`

## 11. 给下一个对话框的注意事项

如果下一个对话要继续修改代码，请先读这些文件：

```text
docs/current_scheme/CODEX_NEXT_CONTEXT_PROMPT_20260521.md
docs/current_scheme/MELD_STL_experiment_summary_20260521.md
src/train/dialogue_text_task_runner.py
src/train/dialogue_text_audio_task_e2e_runner.py
scripts/run_dialogue_task_stl.py
scripts/run_dialogue_text_audio_task_e2e_stl.py
configs/dialogue_task_stl_v2.yaml
```

回答用户时要主动说明：

- 当前 task-incremental 设置是带 task id 的。
- 当前 S3/S5 的测试集也是按任务 split 的。
- 旧结果是否是修正文本前的结果。
- 当前结果差距很小，需要多 seed。
- S4 当前低结果不能直接说明多模态无效。

