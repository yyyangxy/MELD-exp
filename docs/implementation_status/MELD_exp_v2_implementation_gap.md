# MELD-exp v2 实现状态与缺口清单

> 对齐文档：`/data2/yangxy/MELD/document/plan/MELD_exp_code_plan_v2.md`  
> 代码目录：`/data2/yangxy/MELD/MELD-exp`  
> 更新时间：2026-05-15  
> 用途：记录当前代码相对 v2 code plan 的已实现部分、部分实现部分、未实现部分和后续补齐顺序。

---

## 0. 当前总体结论

当前代码已经具备两条 utterance-level 主线的基础能力：

```text
Protocol 1: Utterance-level Task-Sequential MELD-STL
Protocol 2: Utterance-level Modality-Sequential MELD-STL
```

其中 Task-STL 的数据读取、三任务构造、sequential runner、replay、prototype replay、KD 已基本可运行；Modality-STL 的 feature-level dataset、multimodal model、modality runner、modality KD、confidence-aware CMD 已有基础实现。

当前尚未实现或未完全对齐 v2 plan 的重点是：

```text
1. Dialogue-level 两条 protocol 尚未实现
2. 正式 v2 特征方案尚未完成：XLM-R 1024 / openSMILE 6373 / ResNet-50 2048
3. feature audit 脚本尚未独立实现
4. CIL 仍是占位入口，未实现训练流程
5. full_joint / text_static / mod_dropout 等 Modality-STL baseline 未完全实现
6. plotting / ablation runner / 论文表格生成尚未实现
7. README 与当前代码已有不一致，需要后续更新
```

---

## 1. 状态标记说明

| 标记 | 含义 |
|---|---|
| Done | 已实现并通过基础检查 |
| Partial | 有基础实现，但未完全符合 v2 plan 或缺少验证 |
| Missing | 尚未实现 |
| Blocked | 依赖外部资源、模型、特征缓存或环境 |

---

## 2. Protocol 实现状态

### 2.1 Protocol 1: Utterance-level Task-Sequential MELD-STL

状态：`Partial -> 接近 Done`

已实现：

| 项目 | 状态 | 当前文件 |
|---|---|---|
| MELD CSV 读取 | Done | `src/data/meld_csv.py` |
| train/dev/test split 保持 | Done | `src/data/meld_csv.py` |
| sentiment task 构造 | Done | `src/data/task_builder.py` |
| emotion task 构造 | Done | `src/data/task_builder.py` |
| same-speaker shift task 构造 | Done | `src/data/task_builder.py` |
| text-only token dataset | Done | `src/data/datasets.py` |
| text collate | Done | `src/data/collate.py` |
| Multi-task text model | Done | `src/models/stl_model.py` |
| sequential FT | Done | `src/train/sequential_runner.py` |
| LwF | Done | `src/train/trainer.py` |
| random replay | Done | `src/continual/replay_buffer.py` |
| prototype replay | Done | `src/continual/prototype_memory.py` |
| prototype replay + KD | Done | `src/train/trainer.py` |
| WeightedRandomSampler replay/current 平衡 | Done | `src/train/trainer.py`, `configs/main_stl.yaml` |
| stage-wise evaluation matrix | Done | `src/train/sequential_runner.py` |
| main result CSV | Done | `outputs/results/main_stl_results.csv` |

仍需补充：

| 缺口 | 状态 | 建议补充 |
|---|---|---|
| result metadata 记录 label mapping / seed / config hash | Missing | 在结果 CSV 或旁路 JSON 中写入实验元信息 |
| per-class F1 输出 | Partial | 当前 metric 主要输出 accuracy/WF1/Macro-F1，需补 per-class |
| 自动论文主表汇总 | Missing | 新增 `src/eval` 或 `scripts/summarize_results.py` |
| forgetting curve 绘图 | Missing | 新增 plotting 脚本 |
| 多 seed 汇总 | Missing | 新增 seed loop 或结果聚合脚本 |

---

### 2.2 Protocol 2: Utterance-level Modality-Sequential MELD-STL

状态：`Partial`

已实现：

| 项目 | 状态 | 当前文件 |
|---|---|---|
| multimodal feature dataset | Done | `src/data/multimodal_dataset.py` |
| feature path 映射 | Done | `src/data/multimodal_dataset.py`, `src/features/feature_store.py` |
| missing feature 过滤 | Done | `src/data/multimodal_dataset.py` |
| modality mask | Done | `src/data/multimodal_dataset.py` |
| multimodal model | Done | `src/models/multimodal_model.py` |
| concat / gated fusion 基础模块 | Partial | `src/models/fusion.py` |
| modality sequence runner | Done | `src/train/modality_runner.py` |
| mod_seq_ft | Done | `src/train/modality_runner.py` |
| mod_seq_kd | Done | `src/train/modality_runner.py` |
| prototype_replay | Done | `src/train/modality_runner.py`, `src/continual/multimodal_memory.py` |
| cmcrd_ours | Done | `src/train/modality_runner.py` |
| confidence-aware CMD | Done | `src/continual/cross_modal_distillation.py` |
| WeightedRandomSampler | Done | `src/train/modality_runner.py`, `configs/modality_stl.yaml` |
| modality retention / forgetting / gain 基础计算 | Partial | `src/train/modality_runner.py` |

仍需补充：

| 缺口 | 状态 | 建议补充 |
|---|---|---|
| full_joint baseline | Missing | 在 `run_modality_stl.py` / `modality_runner.py` 加方法分支 |
| text_static baseline | Missing | 固定 text-only 训练与最终评估 |
| mod_dropout baseline | Missing | 在 model 或 batch 层增加 modality dropout |
| final missing-modality robustness 评估 | Missing | 评估 Text / Text+Audio / Text+Visual / Full |
| Audio-only / Visual-only 可选评估 | Missing | 可作为 appendix |
| modality result 表格自动整理 | Missing | 新增 result summarizer |
| modality retention curve | Missing | 新增 plotting |
| feature coverage 写入独立 CSV | Missing | 见 Feature Audit 部分 |

---

### 2.3 Protocol 3: Dialogue-level Task-Incremental MELD-STL

状态：`Missing`

v2 目标：

```text
Input:
  D = [u1, u2, ..., un]

Output:
  Sentiment sequence = [s1, s2, ..., sn]
  Emotion sequence   = [e1, e2, ..., en]
  Shift sequence     = [q1, q2, ..., qn]
```

需要新增：

| 模块 | 状态 | 建议文件 |
|---|---|---|
| DialogueRecord | Missing | `src/data/dialogue_dataset.py` 或 `src/data/records.py` |
| dialogue-level label sequence 构造 | Missing | `src/data/dialogue_dataset.py` |
| shift ignore_index / label mask | Missing | `src/data/dialogue_dataset.py` |
| dialogue collate + padding | Missing | `src/data/dialogue_dataset.py` / `src/data/collate.py` |
| dialogue sequence model | Missing | `src/models/dialogue_sequence_model.py` |
| 2-layer BiLSTM encoder | Missing | `src/models/dialogue_encoder.py` |
| context-free utterance baseline | Missing | `src/train/dialogue_task_runner.py` |
| hierarchical BiLSTM baseline | Missing | `src/train/dialogue_task_runner.py` |
| ours: replay + KD | Missing | `src/train/dialogue_task_runner.py` |
| dialogue task config | Missing | `configs/dialogue_task_stl.yaml` |
| runner script | Missing | `scripts/run_dialogue_task_stl.py` |
| result CSV | Missing | `outputs/results/dialogue_task_stl_results.csv` |

验收标准：

```text
1. 每个 dialogue 样本内 utterance_keys / labels / masks 长度一致
2. shift 首次出现位置为 ignore_index，不参与 loss 和 metric
3. batch padding 后 sequence_mask 正确
4. sentiment / emotion / shift 三个 sequence task 都能 forward + backward
5. stage 1/2/3 只评估已学习任务
```

---

### 2.4 Protocol 4: Dialogue-level Modality-Incremental MELD-STL

状态：`Missing`

v2 目标：

```text
Task:
  Emotion Sequence Tagging

Stage:
  Dialogue Text
  Dialogue Text + Audio
  Dialogue Text + Audio + Visual
```

需要新增：

| 模块 | 状态 | 建议文件 |
|---|---|---|
| dialogue multimodal feature dataset | Missing | `src/data/dialogue_dataset.py` |
| B x L x D feature loading | Missing | `src/data/dialogue_dataset.py` |
| modality mask: B x L x 3 | Missing | `src/data/dialogue_dataset.py` |
| dialogue modality runner | Missing | `src/train/dialogue_modality_runner.py` |
| dialogue_mod_seq_ft | Missing | `src/train/dialogue_modality_runner.py` |
| dialogue_mod_seq_kd | Missing | `src/train/dialogue_modality_runner.py` |
| dialogue_cmcrd_ours | Missing | `src/train/dialogue_modality_runner.py` |
| dialogue modality config | Missing | `configs/dialogue_modality_stl.yaml` |
| runner script | Missing | `scripts/run_dialogue_modality_stl.py` |
| result CSV | Missing | `outputs/results/dialogue_modality_stl_results.csv` |

验收标准：

```text
1. Text / Text+Audio / Full 三个 dialogue modality stage 可训练
2. 每个 stage 只评估已出现的模态组合
3. modality KD 可对 text-only / text+audio 分支生效
4. confidence-aware CMD 可用于 Full -> partial modality distillation
5. 输出 dialogue_avg_f1 / modality_forgetting / modality_retention
```

---

## 3. Feature Pipeline 实现状态

### 3.1 当前实现

| 模态 | 当前实现 | 当前维度 | 状态 |
|---|---|---:|---|
| Text | hashing text feature | 256 | Partial |
| Audio | Wav2Vec2 mean pooling | 768 | Partial |
| Visual | ResNet-50 frame feature mean pooling | 2048 | Partial/Done |

当前文件：

```text
src/features/extract_text.py
src/features/extract_audio.py
src/features/extract_visual.py
scripts/extract_features.py
src/features/feature_store.py
```

### 3.2 v2 plan 要求

| 模态 | v2 目标 | 目标维度 | 当前状态 |
|---|---|---:|---|
| Text | XLM-RoBERTa-large utterance embedding | 1024 | Missing |
| Audio | openSMILE feature | 6373 | Missing |
| Visual | ResNet-50 frame mean pooling | 2048 | Partial/Done |

### 3.3 需要补充

| 缺口 | 状态 | 建议 |
|---|---|---|
| `extract_text_xlmr.py` | Missing | 新增 XLM-R feature extractor |
| `extract_audio_opensmile.py` | Missing | 新增 openSMILE extractor 或接入已有 openSMILE CLI |
| `feature_audit.py` | Missing | 独立统计 coverage / dim / bad shape |
| `scripts/audit_features.py` | Missing | 输出 feature coverage CSV |
| 正式配置 1024/6373/2048 | Missing | 可新增 `configs/modality_stl_v2.yaml`，保留旧配置用于 debug |
| feature metadata | Missing | 记录 extractor、model path、dim、date、split coverage |

注意：

当前 feature cache 与 v2 正式维度不一致。不要直接覆盖已有 `outputs/features`，建议新建正式缓存目录：

```text
outputs/features_v2/{split}/{modality}/{utterance_key}.npy
```

或在配置中显式区分：

```yaml
feature_root: outputs/features_v2
feature_dims:
  text: 1024
  audio: 6373
  visual: 2048
```

---

## 4. Continual Learning 模块状态

| 模块 | 状态 | 当前文件 | 说明 |
|---|---|---|---|
| Random replay | Done | `src/continual/replay_buffer.py` | Task-STL 可用 |
| Prototype memory | Done | `src/continual/prototype_memory.py` | Task-STL 可用 |
| Multimodal prototype memory | Done | `src/continual/multimodal_memory.py` | Modality-STL 可用 |
| Task KD | Done | `src/continual/distillation.py` | 基础 KL |
| Cross-modal KD | Done | `src/continual/cross_modal_distillation.py` | 支持 confidence weighting |
| WeightedRandomSampler | Done | `src/train/trainer.py`, `src/train/modality_runner.py` | 已替代 `lambda_replay` |
| EWC | Partial | `src/continual/ewc.py` | 非主线，可暂缓 |
| correctness-aware filtering | Missing | 可作为 ablation |
| entropy-based weighting | Missing | 可作为 ablation |
| contrastive alignment | Missing | v2 中明确第一版不做 |
| prototype-level modality alignment | Missing | 可选增强 |

---

## 5. CIL 实现状态

状态：`Missing`

当前情况：

```text
scripts/run_cil.py 仍是占位入口，只打印未实现提示。
configs/class_incremental.yaml 已存在，但未被训练流程使用。
```

需要新增：

| 模块 | 状态 | 建议文件 |
|---|---|---|
| class split builder | Missing | `src/data/cil_builder.py` |
| class-incremental dataset 过滤 | Missing | `src/data/cil_builder.py` |
| shared 7-class head / seen-class masking | Missing | `src/train/cil_runner.py` |
| Task-IL / Class-IL 评估 | Missing | `src/train/cil_runner.py` |
| CIL replay / KD 接入 | Missing | `src/train/cil_runner.py` |
| confusion matrix | Missing | `scripts/plot_cil_confusion.py` |

优先级：

```text
低于 Protocol 1 / Protocol 2 / Dialogue-level。
可作为 appendix 或毕业论文补充实验。
```

---

## 6. Evaluation 与 Plotting 缺口

当前已实现：

| 项目 | 状态 | 当前文件 |
|---|---|---|
| classification metrics | Done | `src/train/metrics.py` |
| Task-STL stage evaluation | Done | `src/train/sequential_runner.py` |
| Modality-STL stage evaluation | Done | `src/train/modality_runner.py` |
| final avg / forgetting 基础装饰 | Partial | `src/train/metrics.py`, `src/train/modality_runner.py` |

缺口：

| 缺口 | 状态 | 建议文件 |
|---|---|---|
| `src/eval` 独立模块 | Missing | 当前 eval 逻辑散在 `src/train` |
| per-class F1 | Missing/Partial | 扩展 `metrics.py` |
| dialogue_avg_f1 | Missing | Dialogue-level 需要 |
| length bucket F1 | Missing | Dialogue-level 需要 |
| speaker bucket F1 | Missing | Dialogue-level 需要 |
| result summarizer | Missing | `scripts/summarize_results.py` |
| forgetting curve | Missing | `scripts/plot_forgetting.py` |
| modality retention curve | Missing | `scripts/plot_modality_retention.py` |
| memory size curve | Missing | `scripts/plot_ablation.py` |
| confidence ablation summary | Missing | `scripts/summarize_ablation.py` |

---

## 7. Baseline 覆盖情况

### 7.1 Task-STL baseline

| Baseline | 状态 |
|---|---|
| Single-task Training | Missing/可由 runner 扩展 |
| Joint Multi-task Training | Done |
| Sequential Fine-tuning | Done |
| Sequential + LwF | Done |
| Sequential + Random Replay | Done |
| Sequential + Prototype Replay | Done |
| Sequential + Prototype Replay + KD | Done |
| EWC | Partial/不建议优先 |

### 7.2 Modality-STL baseline

| Baseline | 状态 |
|---|---|
| Text-only Static | Missing |
| Full Multimodal Joint | Missing |
| Sequential Modality FT | Done |
| Sequential + KD | Done |
| Sequential + Modality Dropout | Missing |
| Sequential + Prototype Replay | Done |
| Ours: Prototype Replay + Confidence-aware CMD | Done |

### 7.3 Dialogue-level baseline

| Baseline | 状态 |
|---|---|
| Context-free Utterance Classifier | Missing |
| Hierarchical BiLSTM | Missing |
| Ours | Missing |

### 7.4 CIL baseline

| Baseline | 状态 |
|---|---|
| Joint Training | Missing |
| Sequential FT | Missing |
| LwF | Missing |
| Experience Replay | Missing |
| Random Replay | Missing |
| Prototype Replay | Missing |
| Prototype Replay + KD | Missing |

---

## 8. Config 与脚本缺口

### 8.1 已有配置

```text
configs/main_stl.yaml
configs/modality_stl.yaml
configs/class_incremental.yaml
```

### 8.2 需要新增配置

```text
configs/paths.yaml
configs/dialogue_task_stl.yaml
configs/dialogue_modality_stl.yaml
configs/cil.yaml
configs/modality_stl_v2.yaml
```

说明：

`class_incremental.yaml` 已存在，但 v2 plan 中建议命名为 `cil.yaml`。可以保留旧文件，同时新增 `cil.yaml` 作为正式入口，避免破坏已有路径。

### 8.3 已有脚本

```text
scripts/smoke_check.py
scripts/prepare_shift_labels.py
scripts/extract_features.py
scripts/run_baselines.py
scripts/run_main_stl.py
scripts/run_modality_stl.py
scripts/run_cil.py
```

### 8.4 需要新增脚本

```text
scripts/audit_features.py
scripts/run_dialogue_task_stl.py
scripts/run_dialogue_modality_stl.py
scripts/run_ablation.py
scripts/summarize_results.py
scripts/plot_forgetting.py
scripts/plot_modality_retention.py
```

---

## 9. README 与文档缺口

当前 `README.md` 已过时的地方：

```text
1. README 仍说 audio / visual extractors are placeholders
   但当前代码已有 wav2vec2 audio extractor 和 ResNet-50 visual extractor。

2. README 仍记录 modality dims 为 text=256, audio=768, visual=2048
   这适合当前 debug pipeline，但不等于 v2 正式特征方案。

3. README 未说明 WeightedRandomSampler 已替代 lambda_replay。

4. README 未说明 dialogue-level protocol 尚未实现。
```

建议后续新增或更新：

```text
docs/implementation_status/MELD_exp_v2_implementation_gap.md
docs/feature_pipeline.md
docs/experiment_commands.md
docs/result_tables.md
```

---

## 10. 建议后续实现顺序

### Step A：补齐 feature audit 与正式特征配置

目标：

```text
1. 新增 scripts/audit_features.py
2. 输出 feature coverage / bad shape / missing count
3. 新增 modality_stl_v2.yaml，明确 1024 / 6373 / 2048
```

原因：

正式多模态实验依赖 feature cache 的覆盖率和维度稳定性。

### Step B：补齐 Modality-STL baseline

目标：

```text
1. full_joint
2. text_static
3. mod_dropout
4. missing-modality robustness
```

原因：

Protocol 2 是 v2 的第二主实验，也是 CMCRD-inspired 方法的核心落点。

### Step C：实现 Dialogue-level dataset 与 model

目标：

```text
1. DialogueRecord / dialogue dataset
2. padding collate
3. DialogueSequenceModel
4. sequence loss with ignore_index
```

原因：

这是 v2 相对旧版最重要的新实验粒度。

### Step D：实现 Protocol 3 和 Protocol 4 runners

目标：

```text
1. dialogue task sequence runner
2. dialogue modality sequence runner
3. dialogue_avg_f1 / bucket metrics
```

原因：

补齐 v2 四条 protocol。

### Step E：实现 CIL

目标：

```text
1. class split builder
2. class-incremental runner
3. Task-IL / Class-IL 评估
4. confusion matrix
```

原因：

CIL 是毕业方向补充实验，但不应阻塞主线。

### Step F：实现 summarizer 与 plotting

目标：

```text
1. 自动生成主表
2. forgetting curve
3. modality retention curve
4. ablation summary
```

原因：

保证结果能直接进入论文写作。

---

## 11. 下一批最推荐 Codex 任务

建议下一批实现从下面三个中选一个，不要一次性全做：

### 任务 1：Feature Audit

```text
实现 scripts/audit_features.py 和 src/features/feature_audit.py。
输出每个 split / modality 的 expected_count、found_count、missing_count、bad_shape_count、dim。
```

优点：

```text
低风险，能立刻提高实验可控性。
```

### 任务 2：Modality-STL baseline 补齐

```text
在 run_modality_stl.py / modality_runner.py 中补 full_joint、text_static、mod_dropout。
同时增加最终 missing-modality robustness 评估。
```

优点：

```text
直接服务主实验表。
```

### 任务 3：Dialogue Dataset + Model

```text
实现 dialogue dataset、padding collate、DialogueSequenceModel。
先只跑单 batch forward/backward，不急着接完整 runner。
```

优点：

```text
为 Protocol 3 / 4 打基础。
```

---

## 12. 当前修改后的重要设计约定

### 12.1 Replay loss 不再单独加权

当前代码已改为：

```text
supervised_ce = mean(CE(current batch), CE(replay batches))
loss = supervised_ce + lambda_kd * KD + lambda_cmd * CMD
```

并使用：

```text
WeightedRandomSampler
```

保证类别以及 replay/current 来源更均衡。

### 12.2 第一版 CMD 只做 confidence-weighted KL

当前不做：

```text
contrastive push-away
negative pair mining
cross-modal transformer alignment
```

### 12.3 训练阶段不读 MP4

训练 runner 只读取 `.npy` feature cache。MP4 只用于离线 feature extraction。

### 12.4 当前 modality feature 仍是 debug pipeline

当前默认：

```text
text=256 hashing
audio=768 wav2vec2
visual=2048 resnet50
```

v2 正式目标：

```text
text=1024 XLM-R
audio=6373 openSMILE
visual=2048 ResNet-50
```

---

## 13. 最小投稿闭环缺口

如果目标是尽快形成可投稿结果，最低还需要补：

```text
1. Protocol 2 的 full_joint / text_static / mod_dropout
2. feature audit 和 coverage 记录
3. result summarizer
4. forgetting curve
5. modality retention curve
6. confidence CMD ablation 表
```

如果目标是完整 v2 方案，还需要额外补：

```text
1. Dialogue-level dataset
2. Dialogue-level BiLSTM model
3. Protocol 3 runner
4. Protocol 4 runner
5. CIL runner
6. CIL confusion matrix
```

