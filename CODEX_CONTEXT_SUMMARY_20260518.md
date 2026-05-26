# CODEX Context Summary - 2026-05-18

## 1. 当前 MELD-STL 实验目标与进度

当前项目目标是在 MELD 数据集上构建并验证 STL（Sequential Task Learning / Scenario-based Task Learning）实验框架，重点比较 continual learning 方法在不同场景下的表现。

目前已经建立并运行过四类 v2 场景：

1. Utterance-level Task-Incremental MELD-STL
2. Utterance-level Modality-Incremental MELD-STL
3. Dialogue-level Task-Incremental MELD-STL
4. Dialogue-level Modality-Incremental MELD-STL

当前最重要的结果与观察：

- Modality-incremental 场景中，传统 replay CE 不适合直接使用，因为任务标签空间始终是 emotion 7 分类，变化的是输入模态。
- Modality-incremental 更适合使用 old-view KD / relation consistency，而不是 replay CE。
- Task-incremental 场景中，replay CE 是合理的，因为不同阶段学习不同任务。
- Frozen feature 方案的上界偏低。XLM-R frozen feature 的 emotion-only upper bound 约为 0.5066，而 XLM-R end-to-end fine-tuning 的 emotion-only test weighted-F1 达到约 0.5907。
- 因此，后续主线应从 frozen `.npy` feature 方案逐步转向 XLM-R end-to-end fine-tuning。

目前遇到的主要问题：

- 早期 5 epoch 训练明显不充分。
- 100 epoch 下训练 loss 继续下降，但 test weighted-F1 会震荡或下降，说明存在过拟合或任务冲突。
- 当前 Task-STL 的方法效果尚未稳定超过 LwF/KD。
- 当前数据划分方式还没有严格保证 task1/task2/task3 的训练数据互不重合，这会影响任务增量实验的论文规范性。

## 2. Utterance-level 和 Dialogue-level 都需要修改

后续修改不能只改 utterance-level。

必须同时检查和修改：

- `feature_task_runner.py`
- `dialogue_task_runner.py`

原因：

- Utterance-level Task-STL 当前以单条 utterance 为训练与评估单位。
- Dialogue-level Task-STL 当前以 dialogue sequence 为训练输入，但评估仍按 utterance 展开。
- 两个 runner 都应读取固定的 task split 文件，保证任务划分一致、可复现、无数据重叠。

## 3. 最重要结果场景

当前最重要的实验结果应聚焦在：

1. Dialogue-level Task-Incremental MELD-STL
2. Dialogue-level Modality-Incremental MELD-STL

原因：

- MELD 原始任务是 Emotion Recognition in Conversations，dialogue-level context 更符合数据集本意。
- Dialogue-level modality-incremental 目前是较有潜力的结果场景。
- Dialogue-level task-incremental 是后续论文中最需要规范化和提升的场景。

Utterance-level 结果仍可保留，但更适合作为辅助实验或 ablation。

## 4. Task1 / Task2 / Task3 数据不能重合

后续 Task-Incremental 实验中，task1/task2/task3 的训练数据必须互不重合。

不能继续让三个任务共享同一批 train dialogues / utterances，否则任务增量设定会不规范。

推荐设定：

- task1: sentiment
- task2: emotion
- task3: shift

但每个 task 应拥有自己的独立训练 dialogue_ids。

也就是说：

```text
task1 train dialogue_ids ∩ task2 train dialogue_ids = empty
task1 train dialogue_ids ∩ task3 train dialogue_ids = empty
task2 train dialogue_ids ∩ task3 train dialogue_ids = empty
```

dev/test 是否也完全互斥需要后续明确，但为了论文规范，建议 train/dev/test 都按固定 task split 文件读取。

## 5. 划分必须可复现

任务划分必须使用固定 seed。

建议统一使用：

```text
seed = 13
```

划分逻辑必须 deterministic：

- 固定随机种子；
- 固定 dialogue_id 排序；
- 固定 shuffle；
- 输出持久化 split 文件；
- runner 后续只读取 split 文件，不在训练时重新随机划分。

## 6. 建议新建固定 split 目录

希望在 MELD 数据集目录下新建固定 split 文件夹，例如：

```text
/data2/yangxy/dataset/MELD/MELD.Raw/stl_task_splits/
```

推荐结构：

```text
stl_task_splits/
  train/
    task1/
      dialogue_ids.txt
    task2/
      dialogue_ids.txt
    task3/
      dialogue_ids.txt
  dev/
    task1/
      dialogue_ids.txt
    task2/
      dialogue_ids.txt
    task3/
      dialogue_ids.txt
  test/
    task1/
      dialogue_ids.txt
    task2/
      dialogue_ids.txt
    task3/
      dialogue_ids.txt
```

也可以额外保存 metadata，例如：

```text
stl_task_splits/metadata.json
```

metadata 应记录：

- seed；
- task 顺序；
- 每个 split 每个 task 的 dialogue 数量；
- 每个 split 每个 task 的 utterance 数量；
- 生成时间；
- 生成脚本版本或参数。

## 7. 每个 task 保存自己的 dialogue_ids

每个 task 应保存自己的 `dialogue_ids.txt`。

文件内容建议一行一个 dialogue id：

```text
0
1
2
...
```

runner 后续通过这些 dialogue_ids 过滤原始 MELD CSV records。

Dialogue-level runner 直接按 dialogue_id 取完整 dialogue。

Utterance-level runner 则从对应 task 的 dialogue_ids 中展开 utterances。

这样可以保证 utterance-level 和 dialogue-level 使用相同的 task split 定义。

## 8. Runner 后续应读取固定 split 文件

后续 `feature_task_runner.py` 和 `dialogue_task_runner.py` 不应在运行时重新随机划分 task 数据。

它们应：

1. 从 config 中读取 split 根目录，例如：

```yaml
data:
  stl_task_split_root: /data2/yangxy/dataset/MELD/MELD.Raw/stl_task_splits
```

2. 根据当前 split 和 task 读取：

```text
stl_task_splits/{train,dev,test}/task{1,2,3}/dialogue_ids.txt
```

3. 用 dialogue_ids 过滤原始 MELD records。

4. 再构建对应 task examples。

这样实验将具备：

- 可复现性；
- task 数据互斥；
- utterance/dialogue 两种输入级别的一致 split；
- 后续论文可解释性。

## 9. Shift task 要过滤没有有效 shift label 的 dialogue

shift task 需要特别注意。

shift label 是根据同一 speaker 前后 emotion 是否变化构造的。

有些 dialogue 可能没有有效 shift label，例如：

- dialogue 太短；
- 每个 speaker 只出现一次；
- 没有可比较的 same-speaker previous emotion；
- 全部 shift label 为 ignore 或无效。

因此，划分 task3 时必须过滤掉没有有效 shift label 的 dialogue。

推荐规则：

```text
只有至少包含一个有效 shift/no_shift label 的 dialogue 才能进入 task3。
```

否则 dialogue-level shift 训练会出现无有效 supervision 的样本。

Utterance-level shift 也应只展开有效 shift labels 对应的 utterances。

## 10. 下一步需要检查并修改的文件

下一步重点检查和修改：

```text
src/train/feature_task_runner.py
src/train/dialogue_task_runner.py
```

需要完成的事情：

1. 增加读取固定 STL task split 的逻辑。
2. 确保 task1/task2/task3 train 数据互不重合。
3. 确保 utterance-level 和 dialogue-level 使用同一套 dialogue_id split。
4. 确保 shift task 过滤无有效 shift label 的 dialogue。
5. 确保所有划分由固定 seed 生成并持久化。
6. 后续 runner 不再临时随机划分。

建议额外新增一个 split 生成脚本，例如：

```text
scripts/prepare_stl_task_splits.py
```

该脚本负责：

- 读取 MELD 原始 CSV；
- 按 dialogue_id 生成 task1/task2/task3；
- 保证 train 中 task 数据互斥；
- 过滤 task3 中无有效 shift label 的 dialogue；
- 保存 dialogue_ids；
- 保存 metadata。

## 11. 当前不要继续做的事情

当前不建议继续：

- 继续调 `lambda_kd` / `lambda_rel`；
- 继续只在 frozen feature 上堆 loss；
- 继续只看 utterance-level 结果；
- 用 test 最佳 epoch 作为最终模型选择；
- 因为比不过某些 baseline 就删除 baseline；
- 在没有固定 split 的情况下继续跑大量 Task-STL 实验。

当前最重要的是先把 Task-STL 数据划分规范化。

## 12. 给后续对话框的重点结论

后续对话框应优先理解：

```text
当前 MELD-STL 的关键问题不是单个 loss 设计，而是实验设定需要规范化：
1. frozen feature 上界偏低；
2. dialogue-level 结果更重要；
3. Task-STL 必须保证 task1/task2/task3 数据不重合；
4. split 必须固定、可复现、持久化；
5. runner 必须读取固定 split 文件；
6. shift task 必须过滤无有效 shift label 的 dialogue；
7. 下一步重点修改 feature_task_runner.py 和 dialogue_task_runner.py。
```
