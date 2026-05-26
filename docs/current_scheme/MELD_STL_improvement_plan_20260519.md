# MELD-STL 四场景改进计划

日期：2026-05-19

本计划基于两份当前文档：

- `/data2/yangxy/MELD/document/plan/MELD_STL_current_issues_summary_20260518.md`
- `docs/current_scheme/MELD_STL_current_scheme_20260519.md`

当前最重要的判断是：不要再直接围绕某个 loss 权重盲调。现在的瓶颈更像是三件事叠加：

1. evaluation protocol 不够稳，100 epoch 后期震荡明显，需要 dev-based checkpoint selection。
2. Task-STL 里 LwF/KD 本来就是强 baseline，当前 SA-CMD 与它差距很小或略低，必须做更强的 replay/KD/参数隔离，而不是只加 relation loss。
3. Modality-STL 不能当成“旧任务 replay CE”问题；它更接近 old-view behavior retention + modality alignment。

## 0. 当前结果定位

四个并行场景：

```text
S1: Utterance-level Task-STL
S2: Utterance-level Modality-STL
S3: Dialogue-level Task-STL
S4: Dialogue-level Modality-STL
```

当前最关键数值：

| 场景 | 当前最强/主要 baseline | 我们方法状态 | 主要问题 |
| --- | ---: | ---: | --- |
| S1 utterance Task-STL, e2e text | LwF avg W-F1 0.5919 | `text_task_sa_cmd` random 0.5901 | 已非常接近 LwF，但没有稳定超过；需要 seed 与 dev selection |
| S3 dialogue Task-STL, e2e text | `dlg_seq_kd` avg W-F1 0.6115 | SA-CMD dialogue 尚未完整收敛/汇总 | KD 强，random replay 弱；dialogue replay 和 relation 设计还粗 |
| S2 utterance Modality-STL, fixed-feature | SA-CMD final avg 约 0.5333 | no-replay 约 0.5204 | text/audio/visual 特征上界和 modality gain 未诊断清楚 |
| S4 dialogue Modality-STL, fixed-feature | no-replay 约 0.4683 | SA-CMD 约 0.4460 | dialogue + noisy modality + fixed feature 不稳定，优先级应低于 S2/S3 |

阶段性目标：

```text
G0: 先修 evaluation，确认 LwF 与 SA-CMD 的差距是不是统计稳定。
G1: S1 用低风险改动超过或稳定不低于 LwF。
G2: S3 完成当前 suite 后，用 dialogue-specific replay/relation 追 KD。
G3: S2/S4 先重新定义 modality continual objective，再考虑 multimodal 改法。
```

## 1. 文献给出的直接启发

### 1.1 为什么不能低估 LwF/KD

LwF 的核心优点是“不存旧数据，只用新任务数据和蒸馏保持旧能力”。原始 LwF 论文已经说明，在相似新旧任务上，LwF 可能比普通 fine-tuning 更适合保持旧能力。

另外，`In Defense of LwF for Task Incremental Learning` 指出，在 task-incremental setting 下，只要架构、训练增强和实现细节合适，LwF 可以非常强。因此当前 S1/S3 中 LwF/KD 难打并不奇怪。

计划含义：

- 不要把“没超过 LwF”解释成方法必然失败。
- 后续应该把目标设为超过 `LwF+best-dev+same training recipe`，而不是超过一个弱化 LwF。
- 如果方法只比 LwF 高 0.2 到 0.5 个点，必须用 3 seeds 证明。

### 1.2 Replay 不能只做硬标签 CE

iCaRL、DER++、MIR、ER-ACE 一类工作给出的共同信号是：experience replay 有效，但关键不只是“存样本”，而是：

- exemplar 选择要覆盖类内结构，而不是只选 prototype-nearest；
- replay 样本上保留旧模型 logits / dark knowledge 往往比只用 hard label 更稳；
- replay retrieval 可以优先取将被新任务更新严重干扰的样本；
- 新旧任务边界处 representation drift 是主要遗忘来源，应做非对称更新或漂移抑制。

计划含义：

- 当前 `text_task_sa_cmd` 只在 replay batch 上做 hard CE，KD/relation 主要在 current batch 上做，这不是强 replay-KD。
- prototype_nearest 在 S1 已经拉低 shift，说明“近 prototype 样本”可能太容易，不能覆盖边界和少数类。
- 下一步更值得做 `DER-style replay logits`、`replay-batch KD/relation`、`interference-aware/hard replay`。

### 1.3 Transformer continual learning 更适合参数隔离

Adapters、LoRA、lightweight snapshots 一类方法说明：对大 pretrained encoder 做全量 sequential fine-tuning 容易产生共享表示漂移；用小参数模块隔离任务知识，可以显著降低旧知识破坏，同时保持训练可控。

计划含义：

- 对 XLM-R large，单纯全量 fine-tune + KD 可能不是最稳的设置。
- Task-STL 是 task-incremental，有 task id 和 task-specific heads，加入 task-specific adapter/LoRA 是合理设定。
- S1 可先做 `freeze base + task LoRA/adapters + head` sanity check；如果新任务学习不足，再解冻 top-k layers。

### 1.4 ERC 强模型依赖 context、speaker 和 multimodal interaction

MELD、DialogueRNN、DialogueGCN、COSMIC、MMGCN 等工作都强调：

- MELD 是多方对话 multimodal 情绪识别，不只是单句分类。
- speaker state、self/inter-speaker dependency、long-distance context 对 ERC 很重要。
- multimodal ERC 不能只做简单 concat，应该建模跨模态依赖和上下文。

计划含义：

- S3/S4 如果只用 BiLSTM + 简单 relation，很可能不足以打出 dialogue-level 优势。
- shift 任务尤其依赖同 speaker 历史状态，relation loss 应该按 speaker/temporal structure 设计，而不是只做全 batch sample relation。

### 1.5 Modality incremental 的核心是 alignment，不是旧任务 CE

MISA、MAG-BERT、Continual Multimodal Contrastive Learning、modality-incremental MLLM 近期工作都指向同一个问题：新模态带来的退化不只是 forgetting，还包括 modality-specific component 与 shared component 的 misalignment。

计划含义：

- S2/S4 应明确目标：学会新 view，同时保持旧 view behavior。
- replay 如果保留，只应作为 old-view calibration memory，用 KD/consistency，不应作为旧任务 hard-label replay CE 的主解释。
- 新模态应以 text anchor / residual shift / gated fusion 方式加入，避免 audio/visual 噪声破坏 text baseline。

## 2. P0：先统一评估协议

这是所有场景的前置工作。

### 2.1 Dev-based checkpoint selection

当前 100 epoch 下 loss 继续下降但 test W-F1 震荡，不能用 epoch 100，也不能用 test 选 best。

建议：

```text
每个 stage 内每 eval_interval 跑 dev。
selection score = learned tasks 的 mean dev weighted-F1。
可选 tie-breaker = 更低 forgetting 或更高当前任务 dev W-F1。
保存 best_dev_stage{stage}.pt。
最终 test 只汇报 best-dev checkpoint 的结果。
```

对 Task-STL：

```text
stage 1 score = dev(sentiment)
stage 2 score = mean(dev sentiment, dev emotion)
stage 3 score = mean(dev sentiment, dev emotion, dev shift)
```

对 Modality-STL：

```text
stage text score = dev(text-view)
stage text+audio score = mean(dev text-view, dev text+audio-view)
stage text+audio+visual score = mean(dev text, dev text+audio, dev text+audio+visual)
```

优先复用 `scripts/run_text_emotion_finetune.py` 中已有 best-dev 逻辑，扩展到：

- `scripts/run_text_task_stl_finetune.py`
- `src/train/dialogue_text_task_runner.py`
- fixed-feature task/modality runners

### 2.2 统一报告表

每个方法至少汇报：

```text
best_dev_epoch
final_stage test weighted-F1 by task/view
avg weighted-F1
forgetting
retention
current-task plasticity score
3-seed mean/std
```

当前 S1 的 LwF 0.5919 vs SA-CMD 0.5901，差距只有 0.0018。没有 seed/std 之前，不能说方法真的弱于 LwF。

## 3. P1：Upper-bound 和负迁移诊断

### 3.1 Task-STL upper bound

需要同时做 fixed-feature 与 end-to-end text upper bound。

必跑：

```text
emotion-only
sentiment-only
shift-only
task-joint sentiment+emotion+shift
context-free vs dialogue context
```

判断：

```text
如果 emotion-only < 0.55：
  当前 backbone/feature 上界过低，不应继续堆 continual loss。

如果 task-joint 明显低于单任务：
  三任务存在冲突，需要 task routing / adapter / loss balancing。

如果 dialogue joint 没有高于 utterance joint：
  当前 BiLSTM/context 实现没有带来 ERC 文献中预期收益，先修 dialogue modeling。
```

### 3.2 Modality-STL upper bound

对 S2/S4 必须先回答：

```text
text-only joint emotion upper bound 是多少？
text+audio 是否真的高于 text？
text+audio+visual 是否真的高于 text+audio？
dialogue context 是否真的高于 utterance？
```

如果 full modality 不高于 text-only，则当前 audio/visual 是负迁移源，不应把 modality continual 失败归因于 forgetting。

建议新增表：

| view | utterance joint dev/test | dialogue joint dev/test | modality gain |
| --- | ---: | ---: | ---: |
| text | | | baseline |
| text+audio | | | T+A - T |
| text+audio+visual | | | T+A+V - T+A |

## 4. S1：Utterance-level Task-STL 改进

S1 当前最有希望短期出结果，因为 `text_task_sa_cmd` random 已经接近 LwF。

### 4.1 第一批低风险实验

按优先级：

1. `best-dev + 3 seeds`：先确认 LwF 与 SA-CMD 差距是否稳定。
2. `freeze_old_heads`：进入新 stage 后冻结旧 head，只让 encoder/current head 或 adapters 更新。
3. `replay-batch KD/relation`：对 replay batch 同时做 CE + teacher KD + relation，而不是只做 hard CE。
4. `DER-style dark replay`：memory 存旧模型 logits，训练时对 replay logits 做 KL/MSE。
5. `balanced CE`：对 emotion/shift 用 class-balanced loss 或 logit adjustment，所有方法同配方重跑，避免只增强我们方法。

推荐方法名：

```text
text_task_sa_cmd_replay_kd
text_task_sa_cmd_dark_replay
text_task_sa_cmd_freeze_old_heads
```

### 4.2 Replay selection 改法

prototype_nearest 已经不如 random，主要拉低 shift。下一步不要继续押注 nearest。

建议顺序：

```text
random
random + class-balanced quota
diverse / k-center
hard replay: high loss / high entropy / teacher-student disagreement
MIR-lite: 选择被当前 batch 更新前后 logits 变化最大的 memory 样本
```

S1 的 replay 单位是 utterance，因此 MIR-lite 成本可控。实现上可每 N step 在候选 memory 中抽一个小池，估计 student 对旧 task 的 loss 或 KL 增幅，再选 top-k。

### 4.3 KD 权重调度

当前固定 `lambda_kd=1.0, lambda_rel=1.0` 不一定适合 stage 2/3。

建议先做很小的 schedule，不做大网格：

```text
stage 2:
  前 30% steps: lambda_kd 高，保护 sentiment
  后 70% steps: lambda_kd 线性下降，释放 emotion plasticity

stage 3:
  前 30% steps: lambda_kd 高，保护 sentiment/emotion
  后 70% steps: 如果 shift dev 低于 LwF，则降低 kd/relation
```

判断规则：

```text
old dev 掉得多 -> 加 KD/relation
current dev 学不上去 -> 降 KD/relation 或提高 current CE weight
```

### 4.4 参数隔离版本

做一个更强但更清晰的 variant：

```text
XLM-R base frozen
每个 task 一个 LoRA/adapter
每个 task 一个 head
旧 task adapter/head 冻结
新 task 只训练新 adapter/head，必要时训练一个 shared small fusion adapter
```

如果 frozen base 导致 shift/emotion 学不上去，再试：

```text
freeze bottom layers
unfreeze top 2/4 transformer layers
encoder lr = head/adapters lr 的 0.1
```

这个路线即使不作为主方法，也能解释“全量 fine-tune 的表示漂移是否是主要遗忘源”。

## 5. S3：Dialogue-level Task-STL 改进

S3 当前目标不是马上堆新 loss，而是先完成正在跑的 dialogue suite，再按 dialogue 特性改。

### 5.1 完成当前缺口

先补齐：

```text
dlg_sa_cmd_no_replay
text_task_sa_cmd / dlg_text_task_sa_cmd
best-dev selection
3 seeds for dlg_seq_kd and best SA-CMD variant
```

如果 SA-CMD 仍低于 `dlg_seq_kd`：

```text
说明 dialogue 端 replay/relation 没有带来额外收益，优先改 relation 与 memory 单位。
```

### 5.2 Dialogue replay 不能只按 dialogue label 随机

当前 dialogue replay 单位是整段 dialogue。问题是：

- 一个 dialogue 内多数 utterance 可能是 neutral；
- shift positive 稀疏；
- 按 dialogue label 选样本可能没有覆盖关键 speaker transition；
- replay 一个长 dialogue 成本高但有效监督点少。

建议做：

```text
speaker-transition-aware memory
按 task label + shift positive + speaker continuity 分层采样
每个 replay dialogue 记录关键 utterance mask
CE/KD 只对有效 utterance 或高价值 utterance 加权
```

低风险实现：

```text
仍 replay 整段 dialogue，但在 loss 上提高 rare emotion / shift-positive utterance 权重。
```

### 5.3 Speaker-aware relation distillation

当前 relation loss 是 general sample relation。对 dialogue/shift，更合适的是：

```text
same-speaker relation:
  保持同一 speaker 相邻 utterance 的 embedding distance / transition direction

temporal-local relation:
  保持 t-1, t, t+1 邻域表示关系

inter-speaker relation:
  保持同一 dialogue 内不同 speaker 的相对状态
```

建议先实现最简单版本：

```text
对同 speaker 连续 utterance pair:
  teacher_delta = emb_teacher[t] - emb_teacher[prev_same_speaker]
  student_delta = emb_student[t] - emb_student[prev_same_speaker]
  loss = 1 - cosine(student_delta, teacher_delta)
```

这比全 batch relation 更贴合 shift 任务。

### 5.4 Dialogue model upper bound

如果 `dlg_seq_kd` 已经强，但 dialogue joint upper bound 不高，说明 BiLSTM 不是问题核心；如果 dialogue joint 明显应更高但没有达到，应考虑：

```text
speaker embedding
speaker-aware BiLSTM/state
DialogueGCN-style graph encoder
context window ablation
```

不要在 S3 未稳定前引入太重的 COSMIC/commonsense 模块。它适合作为 literature reference 或后续增强，不适合作为第一轮抢救。

## 6. S2：Utterance-level Modality-STL 改进

S2 要先重新定义 objective。

### 6.1 正确目标

学习顺序：

```text
text -> text+audio -> text+audio+visual
```

任务始终是 emotion 7 分类。旧知识不是旧任务，而是旧 view behavior。

推荐 loss：

```text
L = CE(current_view)
  + lambda_view_kd * KD(student_old_view, teacher_old_view)
  + lambda_align * modality alignment / relation consistency
  + optional calibration memory KD
```

不要把 replay CE 当作主机制。memory 可以保留，但解释应是：

```text
old-view calibration memory
not old-task replay memory
```

### 6.2 Text-anchor residual fusion

当前 audio/visual 特征可能噪声大。建议以 text 为 anchor：

```text
h_text = text projection
delta_audio = gate_audio(audio, text) * proj_audio(audio)
delta_visual = gate_visual(visual, text) * proj_visual(visual)
h_fused = h_text + delta_audio + delta_visual
```

好处：

- text-only 能力不容易被新模态破坏；
- audio/visual 只有在 gate 认为有用时才影响预测；
- 可以直接报告 modality gain。

这与 MAG-BERT 的“非文本模态生成对语言表示的 shift”思想一致，但更轻量，适合 fixed-feature。

### 6.3 Modality alignment

参考 MISA/CMCL，建议分成：

```text
private modality feature: 保留每个模态特有信息
shared invariant feature: 对齐 text/audio/visual 的共同情感信息
```

第一轮不要做复杂模型，先加两个简单约束：

```text
1. shared projection contrastive:
   同一 utterance 的 text/audio/visual shared embedding 拉近。

2. old-view embedding KD:
   加新模态后，student 在 text-view 下的 embedding/logits 对齐 teacher text-view。
```

### 6.4 S2 实验顺序

```text
A0: text-only / T+A / T+A+V joint upper bound
A1: current fixed-feature modality no-replay baseline with best-dev
A2: old-view KD only
A3: old-view KD + text-anchor residual gate
A4: old-view KD + gate + contrastive alignment
A5: calibration memory KD, no hard replay CE
```

如果 A0 中 T+A+V 不高于 text-only，则 S2 的论文叙事应转为“modality retention under noisy/incremental modality”，不要声称新模态显著提升 emotion recognition。

## 7. S4：Dialogue-level Modality-STL 改进

S4 当前最不稳，建议降低优先级。只有当 S2 的 modality gain 和 S3 的 dialogue modeling 都比较稳后，再推进 S4。

### 7.1 先做拆解

S4 的失败可能来自三处：

```text
fixed audio/visual feature 噪声
dialogue BiLSTM/context 建模不足
modality incremental objective 不正确
```

拆解实验：

```text
D0: dialogue text-only emotion joint
D1: dialogue T+A joint
D2: dialogue T+A+V joint
D3: utterance fusion frozen -> dialogue encoder train
D4: dialogue encoder frozen -> modality fusion train
```

### 7.2 结构建议

推荐顺序：

```text
先训练/验证 S2 utterance fusion
把 utterance fusion 作为 frozen encoder
再训练 dialogue BiLSTM/graph
最后做 modality incremental old-view KD
```

避免一开始同时更新：

```text
projection + fusion + dialogue encoder + task head
```

否则很难判断退化来自哪里。

### 7.3 Dialogue modality consistency

S4 的 consistency 应包含两层：

```text
utterance-level old-view KD:
  加新模态后，每条 utterance 的 text-view prediction 不变。

dialogue-level temporal consistency:
  整段 dialogue 的 hidden trajectory 不剧烈漂移，尤其同 speaker transition。
```

如果只做最终 logits KD，可能保不住 shift/dialogue trajectory。

## 8. 最小实验矩阵

第一轮不要铺太大。建议 10 组以内确认方向。

### 8.1 S1 第一轮

```text
seed: 13, 21, 42
epochs: 30 or best-dev early stop

methods:
1. lwf
2. text_task_sa_cmd random
3. text_task_sa_cmd random + freeze_old_heads
4. text_task_sa_cmd random + replay_batch_kd
5. text_task_sa_cmd random + dark_replay_logits
```

判断：

```text
如果 3/4/5 任一方法 mean avg W-F1 > LwF mean + 0.005，进入第二轮。
如果只提升 shift 但旧任务掉，调 KD schedule。
如果只保旧任务但 shift 掉，降低 KD/relation 或做 asymmetric replay。
```

### 8.2 S3 第一轮

```text
seed: 13 first, best method 再 3 seeds

methods:
1. dlg_seq_kd
2. dlg_random_replay
3. dlg_sa_cmd_no_replay
4. text_task_sa_cmd random
5. best S1 variant adapted to dialogue
```

额外消融：

```text
speaker-transition replay weighting
speaker-aware relation loss
```

### 8.3 S2 第一轮

```text
methods:
1. text-only joint upper bound
2. T+A joint upper bound
3. T+A+V joint upper bound
4. old-view KD only
5. old-view KD + text-anchor gate
6. old-view KD + text-anchor gate + alignment
```

### 8.4 S4 第一轮

只做诊断，不急着主打：

```text
1. dialogue text-only joint upper bound
2. dialogue T+A joint upper bound
3. dialogue T+A+V joint upper bound
4. old-view KD only
5. old-view KD + frozen S2 utterance fusion
```

## 9. 成功判据

### 9.1 算法成功

```text
S1:
  SA-CMD+ 三 seed mean avg W-F1 >= LwF + 0.005
  且 sentiment/emotion forgetting 不高于 LwF

S3:
  SA-CMD+ 接近或超过 dlg_seq_kd
  shift positive-F1 不低于 KD

S2:
  old-view retention >= no-replay baseline
  modality gain 不为负，或至少不显著负迁移

S4:
  先达到 no-replay baseline，再谈超过 SA-CMD 历史结果
```

### 9.2 论文叙事成功

如果 S1/S3 能稳定超过 LwF/KD：

```text
主线：Task-incremental ERC with stability-plasticity balanced distillation/replay.
```

如果 S1 与 LwF 持平但 S2/S4 有清晰优势：

```text
主线：Task and modality incremental MELD under old-task/old-view retention.
```

如果 frozen multimodal upper bound 低：

```text
主线收窄为 end-to-end text Task-STL；modality 作为分析或附录。
```

## 10. 不建议继续做的事

暂时不建议：

```text
1. 在没有 best-dev/seed 的情况下继续声称某个方法超过 LwF。
2. 继续大网格搜索 lambda_kd/lambda_rel。
3. 在 Modality-STL 里把 replay CE 当作核心贡献。
4. 直接拿 fixed-feature multimodal 数字和 end-to-end text 数字混表。
5. 为了超过 baseline 删除 LwF/KD。
6. 在 S4 未诊断上界前引入复杂 dialogue/multimodal 大模型。
```

## 11. 参考文献与链接

- Poria et al., 2019, MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations, ACL. https://aclanthology.org/P19-1050/
- Majumder et al., 2019, DialogueRNN: An Attentive RNN for Emotion Detection in Conversations, AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/4657
- Ghosal et al., 2019, DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. https://arxiv.org/abs/1908.11540
- Ghosal et al., 2020, COSMIC: COmmonSense knowledge for eMotion Identification in Conversations, Findings EMNLP. https://aclanthology.org/2020.findings-emnlp.224/
- Hu et al., 2021, MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation, ACL-IJCNLP. https://aclanthology.org/2021.acl-long.440/
- Li and Hoiem, 2016/2017, Learning without Forgetting. https://arxiv.org/abs/1606.09282
- Oren and Wolf, 2021, In Defense of the Learning Without Forgetting for Task Incremental Learning. https://arxiv.org/abs/2107.12304
- Rebuffi et al., 2017, iCaRL: Incremental Classifier and Representation Learning, CVPR. https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
- Buzzega et al., 2020, Dark Experience for General Continual Learning: a Strong, Simple Baseline, NeurIPS. https://papers.nips.cc/paper_files/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html
- Aljundi et al., 2019, Online Continual Learning with Maximally Interfered Retrieval. https://arxiv.org/abs/1908.04742
- Caccia et al., 2022, New Insights on Reducing Abrupt Representation Change in Online Continual Learning. https://arxiv.org/abs/2203.03798
- Houlsby et al., 2019, Parameter-Efficient Transfer Learning for NLP, ICML. https://proceedings.mlr.press/v97/houlsby19a.html
- Hu et al., 2022, LoRA: Low-Rank Adaptation of Large Language Models, ICLR. https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/
- Wang et al., 2023, Effective Continual Learning for Text Classification with Lightweight Snapshots, AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/26206
- Rahman et al., 2020, Integrating Multimodal Information in Large Pretrained Transformers, ACL. https://aclanthology.org/2020.acl-main.214/
- Hazarika et al., 2020, MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis. https://arxiv.org/abs/2005.03545
- Liu et al., 2025, Continual Multimodal Contrastive Learning. https://arxiv.org/abs/2503.14963
- Zhang et al., 2025, Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for Multimodal LLMs, EMNLP. https://aclanthology.org/2025.emnlp-main.665.pdf
- Cui et al., 2019, Class-Balanced Loss Based on Effective Number of Samples, CVPR. https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
