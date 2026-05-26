# MELD Task-STL 当前方案与代码实现说明

更新时间：2026-05-19

本文档说明当前 MELD-STL 实验到底在跑什么、代码如何实现、哪些细节会影响实验结论。这里的重点不是只解释“分类头有没有冻结”，而是把数据划分、模型参数更新、回放、蒸馏、relation loss、OOM 改动、日志记录和当前已知风险都写清楚。

## -1. 任务背景

MELD 是一个多轮对话情感理解数据集。每个样本来自电视剧对话，包含 dialogue、utterance、speaker、文本，以及情绪/情感相关标签。这个项目里当前关注的是从同一份 MELD 数据中构造三个连续学习任务：

1. `sentiment`：判断 utterance 的情感极性，通常是 negative / neutral / positive。
2. `emotion`：判断 utterance 的细粒度情绪类别，例如 joy、anger、sadness 等。
3. `shift`：判断同一 speaker 在当前 utterance 相比上一次发言是否发生情绪转移。

这三个任务都和对话情绪理解有关，但监督信号不同。`sentiment` 是粗粒度情感极性，`emotion` 是细粒度情绪分类，`shift` 更强调上下文和同一说话人的状态变化。因此它们天然适合用来研究：模型学完一个情绪相关任务后，再学习另一个相关任务时，旧任务知识会不会被破坏，以及不同连续学习方法能否缓解这种破坏。

这里的 Task-STL 可以理解为 task-incremental single-task learning 序列。模型不是一开始联合训练三个任务，而是按顺序学习：

```text
stage 1: train sentiment
stage 2: train emotion
stage 3: train shift
```

每个 stage 训练时只把当前任务当作主监督任务。训练完新任务后，再回头评估所有已学任务，例如 stage 3 结束后同时评估 sentiment、emotion、shift。如果 stage 3 后 sentiment/emotion 明显下降，就说明发生了 catastrophic forgetting。

这个实验要回答的核心问题是：

- end-to-end fine-tuning XLM-R 时，顺序学习三个 MELD 任务会遗忘多少？
- 只用 LwF/KD 这类传统蒸馏方法能缓解多少？
- 随机回放旧任务样本是否有帮助？
- confidence-based KD 和 relation loss 是否能在不回放或配合回放时进一步改善稳定性？
- dialogue-level 建模，也就是 `XLM-R + BiLSTM`，是否比单句 utterance-level 更适合这些任务？

因此，当前实验不是单纯追求某一个任务最高分，而是比较不同连续学习策略在三个指标上的平衡：

1. 旧任务保持能力：sentiment/emotion 在后续 stage 后是否还好。
2. 新任务学习能力：shift 是否被过强的旧任务约束压制。
3. 整体平均性能：最终 stage 三个任务的 weighted-F1 平均值。

一个方法如果旧任务保持很好但 shift 很差，说明它可能过度保守；如果 shift 很好但 sentiment/emotion 掉得厉害，说明它仍然遗忘明显。当前 `text_task_sa_cmd` 就是在这个背景下设计的：希望用 confidence KD 和 relation loss 保持旧知识，同时用 replay 和当前任务 CE 维持新任务学习能力。

## 0. `text_task_sa_cmd` 到底是什么方法

`text_task_sa_cmd` 是当前 end-to-end text Task-STL 实验里的主方法名。它不是一个单独的新模型结构，而是一套连续学习训练目标，核心思想是在训练新任务时同时做三件事：

1. 用新任务真实标签学习当前任务。
2. 用旧模型 teacher 约束旧任务输出，减少旧任务遗忘。
3. 用旧任务 memory replay 继续监督旧任务，避免只靠蒸馏。

在 stage `t` 训练当前任务 `T_t` 时，当前模型是 student，上一个 stage 结束后的模型会被复制并冻结为 teacher。对一个当前任务 batch，优化目标可以概括为：

```text
L = CE_current
  + CE_replay
  + lambda_kd * ConfidenceKD_old
  + lambda_rel * RelationLoss_old
```

各项含义如下：

- `CE_current`：当前任务的交叉熵，例如 task2 阶段训练 emotion 时，用 emotion 标签监督 emotion head。
- `CE_replay`：从旧任务 memory 里取样本，用旧任务真实标签继续训练旧任务 head。当前默认 memory selection 是 `random`，不是 hybrid selection。
- `ConfidenceKD_old`：对已经学过的旧任务做 knowledge distillation。teacher 对某个样本越自信，该样本的 KD 权重越高。
- `RelationLoss_old`：约束 student 和 teacher 的样本表示关系，使旧任务表示空间不要剧烈漂移。

更具体地说，在 task2/task3 阶段，student 仍然会用当前输入去跑旧任务 head，并和 teacher 的旧任务输出做对齐。因此 `text_task_sa_cmd` 不是“只训练当前 head”的方法。它会更新共享 XLM-R encoder，也可能更新旧任务 head，因为旧任务 head 参与了 KD/replay/relation 相关计算。

它和几个 baseline 的区别是：

| 方法 | 当前任务 CE | 旧任务 KD | confidence weight | relation loss | replay CE |
| --- | --- | --- | --- | --- | --- |
| `seq_ft` | yes | no | no | no | no |
| `lwf` | yes | yes | no | no | no |
| `text_random_replay` | yes | no | no | no | yes |
| `text_sa_cmd_no_replay` | yes | yes | yes | yes | no |
| `text_task_sa_cmd` | yes | yes | yes | yes | yes |

所以，`text_task_sa_cmd` 可以理解为：

```text
LwF-style old-task distillation
+ teacher-confidence sample weighting
+ student/teacher relation preservation
+ old-task random replay
```

当前代码里，`text_task_sa_cmd` 同时用于 utterance-level runner 和 dialogue-level runner。名字里有 `text`，表示它是 text-only end-to-end 方法；但在 dialogue runner 里它处理的是整段 dialogue，不是单句。

## 1. 当前实验目标

当前实验是在 MELD 上做 Task-Incremental STL，任务顺序固定为：

1. `sentiment`
2. `emotion`
3. `shift`

每个 stage 只训练当前任务，训练完一个任务后进入下一个任务。评估时会在已经学过的任务上分别测试，所以最终 stage 3 会同时汇报 sentiment、emotion、shift。

现在主线实验已经从“固定提取好的特征”转为“end-to-end text fine-tuning”。也就是说，当前 utterance/dialogue end-to-end runner 都会直接把文本输入 XLM-R，而不是读 `outputs/features_v2` 中提前抽取好的 text/audio/visual 特征。

## 2. 最重要的数据细节：三个任务现在使用不重叠 dialogue split

之前一个很关键的问题是：三个任务可能在用同一批 dialogue，只是 label 不同。这会让 Task-Incremental 结果不干净，因为模型在 task2/task3 阶段仍然能看到 task1 的同一段文本，只是换了任务标签。

现在已经加入固定 STL task split：

- 代码：`src/data/stl_task_splits.py`
- 生成脚本：`scripts/prepare_stl_task_splits.py`
- 配置入口：`data.stl_task_split_root: stl_task_splits`
- 实际数据目录：`/data2/yangxy/dataset/MELD/MELD.Raw/stl_task_splits`

每个 split 下按任务存 dialogue id：

```text
stl_task_splits/
  train/task1/dialogue_ids.txt  # sentiment
  train/task2/dialogue_ids.txt  # emotion
  train/task3/dialogue_ids.txt  # shift
  dev/...
  test/...
```

`load_stl_task_split()` 会检查同一个 split 内不同任务的 dialogue id 是否有交集。如果有交集会直接报错。因此当前固定 split 下，train/dev/test 各自内部的 task1/task2/task3 dialogue 是不重叠的。

### 需要注意

`shift` 任务不是所有 dialogue 都有效。代码里 `_valid_dialogue_ids_for_task()` 会要求一个 dialogue 中至少出现一次同一 speaker 的后续 utterance，因为 shift 标签依赖“同一说话人上一次情绪”。因此 split 生成时会先过滤掉没有有效 shift 标签的 dialogue。

## 3. Utterance-level end-to-end runner

入口：

```text
scripts/run_text_task_stl_finetune.py
```

模型：

```text
XLMRTaskSTLModel
  encoder: AutoModel.from_pretrained(xlm-roberta-large)
  heads:
    sentiment: Linear(hidden, 3)
    emotion: Linear(hidden, 7)
    shift: Linear(hidden, 2)
```

输入是单句文本，`context_window=0`。每个样本只是一条 utterance。forward 时取 XLM-R 的 `[CLS]` 位置：

```python
embedding = output.last_hidden_state[:, 0]
logits = self.heads[task_name](embedding)
```

支持的方法：

- `seq_ft`
- `lwf`
- `text_random_replay`
- `text_sa_cmd_no_replay`
- `text_task_sa_cmd`

CLI 支持单方法 `--method`，也支持 suite runner 风格的 `--methods ...`。

## 4. Dialogue-level end-to-end runner

入口：

```text
scripts/run_dialogue_task_stl.py
src/train/dialogue_text_task_runner.py
```

模型：

```text
XLMRDialogueTaskModel
  encoder: AutoModel.from_pretrained(xlm-roberta-large)
  optional dialogue_encoder: BiLSTM
  heads:
    sentiment: Linear(dialogue_dim, 3)
    emotion: Linear(dialogue_dim, 7)
    shift: Linear(dialogue_dim, 2)
```

输入是一个 dialogue 中的多条 utterance。代码会把 `(batch, max_dialogue_len, token_len)` 展平成 `(batch * max_dialogue_len, token_len)` 送入 XLM-R，再 reshape 回 dialogue 序列。如果方法不是 `context_free`，会继续经过 BiLSTM。

当前 `configs/dialogue_task_stl_v2.yaml` 默认：

- `batch_size: 1`
- `grad_accum_steps: 8`
- `epochs: 30`
- `fp16: true`
- `max_length: 128`
- `dialogue_hidden_dim: 256`
- `dialogue_num_layers: 2`

`batch_size: 1` 是 dialogue 端为了避免 OOM 的保守默认值。它不是算法定义的一部分，但会影响训练噪声和速度。用 `grad_accum_steps: 8` 后，优化器看到的有效 batch size 约等于 8 个 dialogue。

## 5. 固定特征 runner 与 end-to-end runner 的关系

项目里仍然保留固定特征版配置和代码，例如 `configs/main_stl_v2.yaml` 中的：

```yaml
modalities:
  active_modalities: [text, audio, visual]
feature_paths:
  feature_root: outputs/features_v2
```

这些字段主要服务于旧的 multimodal fixed-feature 实验。当前 `scripts/run_text_task_stl_finetune.py` 和 `scripts/run_dialogue_task_stl.py` 的 end-to-end text 实验不会加载固定 `.npy` 特征，也不会使用 audio/visual 特征。

因此现在比较结果时要分清：

- `run_text_task_stl_finetune.py`：utterance-level, text-only, XLM-R end-to-end
- `run_dialogue_task_stl.py`：dialogue-level, text-only, XLM-R end-to-end, 可带 BiLSTM
- 旧 main STL runner：fixed-feature multimodal，不应直接和 end-to-end text-only 数字混在一起解释

## 6. 旧任务分类头目前没有显式冻结

这是一个重要细节。

当前 optimizer 是：

```python
optimizer = torch.optim.AdamW(model.parameters(), ...)
```

也就是说，XLM-R encoder、dialogue encoder、所有 task heads 都被交给 optimizer。代码没有在进入新任务时对旧任务 head 设置 `requires_grad_(False)`。

但不同方法下，旧 head 是否实际更新取决于它有没有参与 forward 和 loss：

| 方法 | 旧 head 是否会被用到 | 旧 head 是否可能更新 |
| --- | --- | --- |
| `seq_ft` / `dlg_seq_ft` | 新任务训练时只 forward 当前 head | 通常不会，因为旧 head 没有梯度 |
| `lwf` / `dlg_seq_kd` | 旧任务 KD 会 forward 旧 head | 会 |
| `text_sa_cmd_no_replay` / `dlg_sa_cmd_no_replay` | 旧任务 KD 和 relation 会 forward 旧 head | 会 |
| `text_random_replay` / `dlg_random_replay` | replay batch 会 forward 旧 head | 会 |
| `text_task_sa_cmd` | KD/relation/replay 都可能 forward 旧 head | 会 |

这会影响实验解释。现在的方法不是“只共享 encoder、每个旧任务 head 固定不动”的设定，而是“旧任务 head 可以通过 KD 或 replay 继续被优化”的设定。

如果后续想做更严格的 TIL 消融，可以加入 `freeze_old_heads`：

- 进入新 stage 后冻结 `heads[old_task]`
- 只允许 encoder、dialogue encoder 和当前任务 head 更新
- 或者进一步冻结旧 head 但允许旧 head 用于 KD 目标对齐

这会改变优化目标，不能和当前结果直接等价比较。

## 7. Encoder 也没有冻结

当前 end-to-end runner 的核心就是 fine-tune XLM-R，所以 encoder 参数也在更新。新任务训练时，encoder 的更新会影响所有任务的表示空间。这是遗忘产生的主要来源之一。

因此当前实验衡量的是：

```text
共享 XLM-R encoder + task-specific heads 在连续任务上的遗忘与缓解
```

不是：

```text
固定 encoder 特征 + 只训练小分类器
```

这也是为什么 end-to-end dialogue 端显存压力远大于固定特征 runner。

## 8. 方法定义

### 8.1 `seq_ft` / `dlg_seq_ft`

只用当前任务 supervised CE：

```text
L = CE(current_task)
```

没有 teacher，没有 replay，没有 relation loss。

### 8.2 `lwf` / `dlg_seq_kd`

从 task2 开始，保存上一 stage 模型作为 frozen teacher。训练当前任务时，对旧任务做知识蒸馏：

```text
L = CE(current_task) + lambda_kd * KD(student_old_logits, teacher_old_logits)
```

teacher 是 `copy.deepcopy(model)` 后冻结。student 是当前正在训练的模型。

### 8.3 `text_random_replay` / `dlg_random_replay`

训练当前任务时混入旧任务 memory 中的样本，旧任务 replay batch 使用旧任务真实标签做 CE：

```text
L = average(CE(current_task), CE(replay_task_1), ...)
```

当前默认 replay strategy 是 `random`，即每个 class 随机保留 `memory_per_class` 个样本。

### 8.4 `text_sa_cmd_no_replay` / `dlg_sa_cmd_no_replay`

没有 replay，只做 confidence KD + sample relation loss：

```text
L = CE(current_task)
  + lambda_kd * weighted_KD(old_tasks)
  + lambda_rel * weighted_sample_relation(old_tasks)
```

其中 confidence weight 来自 teacher logits，teacher 越自信的样本权重越大。

这组是重要消融：它回答“不靠回放，只靠蒸馏和 relation 是否有效”。

### 8.5 `text_task_sa_cmd`

当前主方法，包含：

- 当前任务 CE
- 旧任务 confidence KD
- 旧任务 sample relation loss
- 旧任务 replay CE

可写成：

```text
L = CE(current_task)
  + CE(replay)
  + lambda_kd * weighted_KD(old_tasks)
  + lambda_rel * weighted_sample_relation(old_tasks)
```

在 dialogue runner 中，`text_task_sa_cmd` 是为了和 utterance runner 保持名字一致而支持的 alias。它会走 dialogue-level 模型和 dialogue-level 数据，不是单句实验。

## 9. 回放策略

当前 utterance/dialogue end-to-end runner 都支持：

- `random`
- `prototype_nearest`
- `diverse`
- `hybrid`

但你当前明确要先跑随机回放，不用 hybrid selection，所以推荐主实验用：

```text
--replay-strategy random
```

### 9.1 random

按 label 分组，每类随机保留最多 `memory_per_class` 个样本。

utterance 端的样本单位是 utterance。dialogue 端的样本单位是 dialogue。

### 9.2 prototype_nearest

先用当前模型提 embedding，再按 class 计算均值 prototype，选择离 prototype 最近的样本。

### 9.3 diverse

先选离 prototype 较远的点作为起点，再用 farthest-first 思路选更多样本，目标是覆盖更分散的区域。

### 9.4 hybrid

一部分选 prototype_nearest，一部分选 diverse。比例由 `representative_ratio` 控制。

注意：这些 selection 方法只决定 memory 中保留哪些样本，不改变训练 loss 的形式。

## 10. `pg_trd` 不是当前主线回放方法

之前提到的 `pg_trd` 更接近 prototype-guided task relation distillation。它维护的是 prototype bank，用于 relation/prototype alignment，不是“把旧样本拿回来训练”的 replay memory。

当前你要对比的“回放/不回放”主线应优先看：

- 无回放：`seq_ft`, `lwf`, `text_sa_cmd_no_replay`
- 随机回放：`text_random_replay`, `text_task_sa_cmd`

`dlg_task_pg_trd` 可以作为额外方法，但不应被当作随机回放方法。

## 11. Dialogue OOM 修改及其数学影响

dialogue 端之前 OOM 的主要原因是：

1. 一个 batch 中有多个 utterance，XLM-R 实际看到的是 `batch_size * max_dialogue_len` 条文本。
2. KD 对每个旧任务都重新 forward student/teacher，会重复占用显存。
3. replay loss 和 current loss 如果堆在同一张 computation graph 中，会同时保留多份激活。

当前代码做了两类节省：

### 11.1 复用 embedding 做旧任务 head 分类

`XLMRDialogueTaskModel` 加了：

```python
def classify_embedding(self, embedding, task_name):
    return {"logits": self.heads[task_name](embedding), "embedding": embedding}
```

训练当前 batch 时，student 只跑一次 XLM-R 得到 embedding，然后对旧 task head 重用这个 embedding。teacher 也只跑一次 encoder，再复用 teacher embedding。

数学上，这不改变旧任务 head 的 logits 定义。因为原来的旧任务 forward 也是：

```text
embedding = encoder(x)
logits_old = head_old(embedding)
```

现在只是把同一个 `embedding` 复用给多个 head，避免重复计算 encoder。

### 11.2 replay 分开 backward

当前 dialogue `_train_epoch()` 中，current loss 先 backward，随后每个 replay loss 单独 forward/backward。optimizer step 仍按 `grad_accum_steps` 执行。

数学上，如果忽略 dropout 随机性、AMP 数值误差和 batchnorm 这类状态层，下面两种写法对参数梯度是等价的：

```text
backward(L_current + L_replay_1 + L_replay_2)
```

和：

```text
backward(L_current)
backward(L_replay_1)
backward(L_replay_2)
```

因为梯度满足线性性：

```text
grad(L1 + L2) = grad(L1) + grad(L2)
```

这个改动的目的只是降低峰值显存，不是改变目标函数。

需要注意的是，模型中有 dropout，因此“完全逐 bit 相同”不能保证。但这属于训练随机性，不是算法目标发生了变化。

## 12. Loss 权重细节

utterance runner 中，replay 是把 current CE 和所有 replay CE 放到一起求平均：

```python
loss = torch.stack([current_loss, *replay_terms]).mean()
```

dialogue runner 中，为了支持分开 backward，先计算：

```python
supervised_weight = 1.0 / (1 + replay_count)
```

然后 current supervised loss 和每个 replay supervised loss 都乘以这个权重。这样 replay 数量为 `k` 时，supervised CE 部分仍然等价于对 `1 + k` 个 CE 取平均。

KD/relation 当前是在 current batch 上计算，不是在 replay batch 上计算。

## 13. 日志与参数记录

utterance/dialogue runner 都会写：

```text
logs/run_parameters.json
```

其中包含：

- CLI 输入参数或 CLI train overrides
- 完整 config
- effective train 参数
- epochs
- batch size
- grad accumulation
- effective batch size
- lr / weight decay
- memory_per_class
- replay_strategy
- representative_ratio
- device / gpu id / fp16

这对后续整理实验非常重要。看结果 CSV 时，应同时打开同一次 run 的 `logs/run_parameters.json`，否则很容易混淆是 batch size、epoch、replay strategy 还是方法本身导致差异。

## 14. 当前实验结果

本节记录截至 2026-05-19 当前已经写出 CSV 的实验结果。注意这里主要比较 end-to-end text Task-STL；早期 fixed-feature multimodal/modality-incremental 结果放在后面单独说明，不和当前 text-only end-to-end 结果直接混合比较。

### 14.1 Utterance-level end-to-end Task-STL

已完成的主结果文件包括：

```text
outputs/runs/text_task_stl_finetune/20260518_225956_nightly_utt_e2e_ablation_seq_ft/results/text_task_stl_finetune.csv
outputs/runs/text_task_stl_finetune/20260519_003441_nightly_utt_e2e_ablation_lwf/results/text_task_stl_finetune.csv
outputs/runs/text_task_stl_finetune/20260519_030017_nightly_utt_e2e_ablation_text_random_replay/results/text_task_stl_finetune.csv
outputs/runs/text_task_stl_finetune/20260519_052653_nightly_utt_e2e_ablation_text_sa_cmd_no_replay/results/text_task_stl_finetune.csv
outputs/runs/text_task_stl_finetune/20260519_081730_nightly_utt_e2e_ablation_text_task_sa_cmd/results/text_task_stl_finetune.csv
outputs/runs/text_task_stl_finetune/20260519_121947_nightly_utt_e2e_text_task_sa_cmd_prototype_nearest/results/text_task_stl_finetune.csv
```

最终 stage 的 test weighted-F1 摘要：

| method | replay strategy | sentiment | emotion | shift | shift positive-F1 | avg weighted-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `seq_ft` | none | 0.5023 | 0.4119 | 0.5296 | 0.5718 | 0.4813 |
| `lwf` | none | 0.6548 | 0.5873 | 0.5335 | 0.5925 | 0.5919 |
| `text_random_replay` | random | 0.6486 | 0.5213 | 0.5444 | 0.5815 | 0.5714 |
| `text_sa_cmd_no_replay` | none | 0.6562 | 0.5733 | 0.5060 | 0.5557 | 0.5785 |
| `text_task_sa_cmd` | random | 0.6525 | 0.5856 | 0.5323 | 0.5838 | 0.5901 |
| `text_task_sa_cmd` | prototype_nearest | 0.6446 | 0.5865 | 0.5006 | 0.5570 | 0.5772 |

当前解读：

- `seq_ft` 明显遗忘。
- `lwf` 是目前最强 baseline，平均 weighted-F1 略高于 `text_task_sa_cmd`。
- `text_random_replay` 对 shift 有帮助，但 emotion 掉得明显。
- `text_sa_cmd_no_replay` 对旧任务保持较好，但 shift 弱。
- `text_task_sa_cmd` random replay 更均衡，和 LwF 很接近，但目前还没有超过 LwF。
- `prototype_nearest` selection 不如 random，主要是 shift 明显下降。这支持当前先用 random replay 作为主设置。

因此，当前结果能支持“KD/relation/replay 对遗忘有缓解作用”，但还不能支持“当前方法显著优于传统 LwF”。

截至当前，`text_task_sa_cmd --replay-strategy diverse` 仍在运行，还没有写出结果 CSV；因此暂时不把 diverse selection 写入完成结果表。

### 14.2 Dialogue-level end-to-end Task-STL

已完成结果文件：

```text
outputs/runs/dialogue_text_task_stl/20260518_225922_nightly_dialogue_e2e_ablation_dlg_seq_ft/results/dialogue_text_task_stl_results.csv
outputs/runs/dialogue_text_task_stl/20260519_111056_nightly_dialogue_e2e_ablation_retry_dlg_seq_kd/results/dialogue_text_task_stl_results.csv
outputs/runs/dialogue_text_task_stl/20260519_121344_nightly_dialogue_e2e_ablation_retry_dlg_random_replay/results/dialogue_text_task_stl_results.csv
```

最终 stage 的 test weighted-F1 摘要：

| method | sentiment | emotion | shift | shift positive-F1 | avg weighted-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dlg_seq_ft` | 0.4785 | 0.4490 | 0.5666 | 0.6054 | 0.4980 |
| `dlg_seq_kd` | 0.6635 | 0.5922 | 0.5787 | 0.6343 | 0.6115 |
| `dlg_random_replay` | 0.6241 | 0.5200 | 0.5554 | 0.5813 | 0.5665 |

当前解读：

- dialogue 端 `dlg_seq_kd` 目前明显最好，比 `dlg_seq_ft` 和 `dlg_random_replay` 都高。
- `dlg_seq_ft` 的 shift 不低，但 sentiment/emotion 遗忘严重，所以最终平均值低。
- `dlg_random_replay` 相比 `dlg_seq_ft` 有提升，但明显不如 KD。说明在 dialogue 端，当前随机 replay 本身还没有比 teacher KD 更稳。
- 目前还不能判断 dialogue 端 `text_task_sa_cmd` 是否优于 KD，因为 `dlg_sa_cmd_no_replay` 和后续 `text_task_sa_cmd` 还在同一个 suite 中继续运行。

### 14.3 正在运行但尚未完成的实验

当前仍在运行的实验包括：

```text
text_task_sa_cmd --replay-strategy diverse
dialogue suite: dlg_seq_kd dlg_random_replay dlg_sa_cmd_no_replay text_task_sa_cmd
```

dialogue suite 里已经有 `dlg_seq_kd` 和 `dlg_random_replay` 的结果；`dlg_sa_cmd_no_replay` 当前已经进入 emotion 阶段后段，但还没有最终 CSV。等这个 suite 全部跑完后，最关键要看：

- `dlg_sa_cmd_no_replay` 是否能接近或超过 `dlg_seq_kd`
- dialogue `text_task_sa_cmd` 是否能在保持 sentiment/emotion 的同时不压低 shift
- replay 在 dialogue 端是否仍然弱于 KD，还是只有单独 random replay 弱

dialogue end-to-end 已经可以跑 suite runner，并支持随机回放和 `text_task_sa_cmd`。之前 `dlg_seq_kd` 在 sentiment 后进入旧任务 KD 时 OOM，原因是 dialogue XLM-R + KD/replay 显存峰值太高。

目前已经做了：

- `classify_embedding()` 复用 embedding
- teacher/student 旧任务 head 不再重复跑 XLM-R encoder
- replay 分开 backward 降低峰值显存
- 默认 config 改为 `epochs: 30`

建议 dialogue 端继续使用：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py \
  --gpu-id 8 \
  --config configs/dialogue_task_stl_v2.yaml \
  --methods dlg_seq_kd dlg_random_replay dlg_sa_cmd_no_replay text_task_sa_cmd \
  --epochs 30 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --memory-per-class 100 \
  --replay-strategy random \
  --fp16 \
  --run-name nightly_dialogue_e2e_ablation_retry
```

如果仍然 OOM，优先降低 `max_length` 或临时只跑部分方法，不建议先改算法 loss。

## 15. 模态增量实验背景与已有结果

除了当前的 task-incremental Task-STL，我们之前还讨论和实现过 modality-incremental 方向。它和当前任务增量不是同一个问题。

Task-incremental 的学习顺序是任务变了：

```text
sentiment -> emotion -> shift
```

Modality-incremental 的学习顺序是输入模态逐步增加，通常任务本身保持为同一个分类目标：

```text
text -> text+audio -> text+audio+visual
```

也就是说，模态增量要回答的问题是：模型先只学 text，再加入 audio，再加入 visual 时，旧模态能力是否会被破坏，新模态是否真的带来增益。它关注的是 modality forgetting / modality retention / modality gain，而不是 sentiment/emotion/shift 三个任务之间的遗忘。

当前代码里和模态增量相关的 runner/结果主要是 fixed-feature 版本：

```text
outputs/runs/modality_stl_v2/
outputs/runs/dialogue_modality_stl_v2/
```

这些实验使用的是预提取特征，不是当前 text-only XLM-R end-to-end 主线。因此它们可以作为历史参考和方法动机，但不能直接和当前 `text_task_sa_cmd` 的数字放在同一张主比较表里。

已有模态增量结果中，较新的 100 epoch run 摘要如下：

| setting | final eval text | final eval text+audio | final eval text+audio+visual | final avg |
| --- | ---: | ---: | ---: | ---: |
| utterance modality SA-CMD | 0.5320 | 0.5352 | 0.5327 | 0.5333 |
| dialogue modality SA-CMD | 0.4429 | 0.4477 | 0.4473 | 0.4460 |

对应结果文件：

```text
outputs/runs/modality_stl_v2/20260517_153713_ep100_eval10_utt_mod/results/modality_stl_results.csv
outputs/runs/dialogue_modality_stl_v2/20260517_145447_ep100_eval10_dlg_mod/results/dialogue_modality_stl_results.csv
```

另一个 no-replay CE+KD+relation 的历史 run 摘要：

| setting | final eval text | final eval text+audio | final eval text+audio+visual | final avg |
| --- | ---: | ---: | ---: | ---: |
| utterance modality no replay | 0.5222 | 0.5208 | 0.5182 | 0.5204 |
| dialogue modality no replay | 0.4716 | 0.4666 | 0.4667 | 0.4683 |

对应结果文件：

```text
outputs/runs/modality_stl_v2/20260516_230812_final_utt_mod_no_replay_ce_kd1.0_rel1.0/results/modality_stl_results.csv
outputs/runs/dialogue_modality_stl_v2/20260516_233440_final_dlg_mod_no_replay_ce_kd1.0_rel1.0/results/dialogue_modality_stl_results.csv
```

目前对模态增量结果的谨慎解读是：

- utterance 模态增量中，100 epoch SA-CMD final avg 约 0.533，高于 no-replay 约 0.520，说明回放/选择策略可能有帮助。
- dialogue 模态增量中，100 epoch SA-CMD final avg 约 0.446，反而低于 no-replay 约 0.468。这说明 dialogue 模态增量当时的实现或参数不一定稳定，不能简单说 SA-CMD 在所有设置都有效。
- 模态增量当前仍是 fixed-feature multimodal 线索，和现在的 end-to-end text Task-STL 主线不同。后续如果要统一论文叙事，需要决定是把它作为另一个实验设置，还是只保留 task-incremental text end-to-end 主线。

## 16. 当前推荐的 utterance 命令

```bash
python scripts/run_text_task_stl_finetune.py \
  --gpu-id 9 \
  --config configs/main_stl_v2.yaml \
  --methods seq_ft lwf text_random_replay text_sa_cmd_no_replay text_task_sa_cmd \
  --epochs 30 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --memory-per-class 100 \
  --replay-strategy random \
  --fp16 \
  --run-name nightly_utt_e2e_ablation
```

如果要试不同回放 selection，只改 `--replay-strategy`，并保持其他参数不变：

```bash
for s in prototype_nearest diverse hybrid; do
  python scripts/run_text_task_stl_finetune.py \
    --gpu-id 9 \
    --config configs/main_stl_v2.yaml \
    --methods text_task_sa_cmd \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --memory-per-class 100 \
    --replay-strategy $s \
    --representative-ratio 0.5 \
    --fp16 \
    --run-name nightly_utt_e2e_text_task_sa_cmd_${s}
done
```

## 17. 当前仍需补的实验/消融

建议后续按优先级补：

1. dialogue 端完整跑完 `dlg_sa_cmd_no_replay` 和 `text_task_sa_cmd`，再和 `dlg_seq_kd` 比较。
2. 等 utterance `diverse` selection 跑完，补齐 random/prototype_nearest/diverse/hybrid 的 selection 消融。
3. utterance 端跑 3 个 seed，确认 `lwf` 和 `text_task_sa_cmd` 的差距是否稳定。
4. 加 `freeze_old_heads` 消融，明确旧 head 继续训练是否贡献了结果。
5. 加只冻结 XLM-R encoder 或只训练 head 的 sanity check，用来分离“表示漂移”和“分类头漂移”。
6. 对 replay memory size 做消融，例如 20/50/100 per class。
7. 对 `lambda_kd` 和 `lambda_rel` 做小网格，例如 0.5/1.0/2.0。
8. 如果要保留模态增量作为论文实验，需要重新检查 fixed-feature multimodal runner 的数据 split、方法命名和最终指标，避免和当前 text-only end-to-end 主线混淆。

## 18. 目前结果解释时必须声明的假设

写论文或实验报告时，需要明确说明：

- 当前是 text-only end-to-end，不是 multimodal fixed-feature。
- 三个任务使用固定 dialogue-level disjoint split。
- utterance 和 dialogue 实验的数据单位不同：utterance runner 是单句，dialogue runner 是整段 dialogue。
- XLM-R encoder 没有冻结。
- 旧任务 head 没有显式冻结；只是在没有参与 loss 时不会更新。
- replay memory 当前默认按 class 随机选，单位在 utterance/dialogue 两个 runner 中不同。
- KD/relation 当前主要在当前 batch 上对旧任务 logits/embedding 做约束，不是在 replay batch 上做。
- dialogue OOM 优化主要改变计算图执行方式和重复 forward，不应被解释为新算法。

这些点如果不写清楚，后续很容易把“算法效果”“数据 split”“模型是否冻结”“回放单位”“runner 粒度”混在一起解释。
