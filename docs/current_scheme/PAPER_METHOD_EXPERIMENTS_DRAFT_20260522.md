# MELD Continual Learning 论文方案与实验素材（2026-05-22）

本文档面向后续论文写作，整理当前方案、方法流程、创新点、方法细节、实验细节、已有结果、可画图内容和后续补实验计划。它不是代码交接文档；如果需要上手跑实验，请看：

```text
docs/current_scheme/HANDOFF_PROMPT_20260522.md
```

## 1. 研究背景与问题定义

### 1.1 背景

MELD 是一个多模态多方对话情绪识别数据集，通常用于静态 ERC/MERC 评测：给定完整训练集一次性训练模型，然后在 test set 上评估 emotion recognition。现有 MELD 工作大多关注：

- 上下文建模；
- speaker dependency；
- 多模态融合；
- emotion shift；
- 类别不平衡；
- commonsense knowledge；
- text/audio/visual 表示学习。

但 MELD 上还没有非常标准化的 continual learning / task-incremental learning 评测体系。因此本项目把 MELD 改造成 dialogue-level task-incremental learning benchmark，研究模型如何顺序学习不同对话情绪相关任务，同时保持旧任务能力。

### 1.2 当前任务设置：Dialogue-level Task-IL

主实验是 S3：Text-Only Dialogue Task-STL。

任务顺序为：

```text
Task 1: sentiment classification
Task 2: emotion classification
Task 3: emotion-shift detection
```

每个 task 都是 dialogue-level sequence labeling：输入一段 dialogue 的 utterance 序列，输出每个 utterance 的任务标签。评估时 task id 已知，即分别在 sentiment/emotion/shift head 上评估已学习任务。

可以在论文里这样写：

> We formulate MELD as a dialogue-level task-incremental learning problem. A model sequentially learns sentiment recognition, emotion classification, and emotion-shift detection. At each stage, only data from the current task is available, and the model is evaluated on all tasks learned so far.

### 1.3 为什么这个设置有意义

这个设置模拟真实情绪计算系统的演化：

- 系统可能先学粗粒度 sentiment；
- 再学细粒度 emotion；
- 再学与上下文和 speaker history 相关的 emotion shift；
- 新任务到来时不能丢失旧任务能力。

相比静态 ERC，这个设置更关注：

- stability-plasticity trade-off；
- dialogue representation 是否能跨任务保留；
- old-task forgetting；
- replay/distillation/regularization 对对话任务的适配性。

## 2. 方法总览

当前最合理的论文主线是：

```text
SA-CMD: Confidence-aware Relation Distillation for Dialogue Task-Incremental Emotion Recognition
```

注意：当前结果不支持把 replay 作为主要创新点。更稳的叙事是：

- 主方法：SA-CMD，即 confidence-aware KD + sample relation distillation；
- replay：可选增强模块，但当前不是主要收益来源；
- S6：audio-assisted text-only distillation，是多模态扩展方向。

整体结构：

```text
Stage t:
  1. 训练当前任务 CE
  2. 使用上一阶段模型作为 teacher
  3. 对旧任务做 confidence-aware KD
  4. 保持 teacher/student embedding 的样本关系结构
  5. 可选：加入 replay batch
```

## 3. SA-CMD 方法细节

### 3.1 基础模型

S3 text-only backbone 当前主要使用 BERT-base：

```text
/data2/yangxy/models/bert-base-uncased
```

也跑过 XLM-R-large：

```text
/data2/yangxy/models/xlm-roberta-large
```

模型结构：

```text
utterance text
  -> BERT / XLM-R encoder
  -> utterance embedding
  -> BiLSTM dialogue encoder
  -> task-specific classifier head
```

任务 head：

```text
sentiment head
emotion head
shift head
```

代码位置：

```text
src/train/dialogue_text_task_runner.py
class XLMRDialogueTaskModel
```

虽然类名叫 `XLMRDialogueTaskModel`，但实际通过 `AutoModel.from_pretrained` 支持 BERT/XLM-R。

### 3.2 LwF baseline

LwF 对旧任务进行 logits distillation：

```text
L = L_CE(current task)
  + lambda_kd * L_KD(old task logits)
```

形式化表示：

```text
L_KD = KL( softmax(z_teacher / T) || softmax(z_student / T) )
```

其中：

- `z_teacher` 是上一阶段冻结模型对旧任务 head 的 logits；
- `z_student` 是当前模型对旧任务 head 的 logits；
- `T` 是 temperature。

代码方法名：

```text
dlg_seq_kd
```

### 3.3 SA-CMD：Confidence-aware KD

LwF 的问题是把 teacher 的所有输出等权对待。SA-CMD 引入 teacher confidence：

```text
w_i = max_c softmax(z_teacher_i / T)_c
```

然后对每个样本的 KD loss 加权：

```text
L_CKD = mean_i w_i * KL(p_teacher_i || p_student_i)
```

直觉：

- teacher 对某个样本高置信，说明旧知识更可靠，应强约束；
- teacher 低置信样本可能是困难样本/噪声样本，降低蒸馏权重；
- 避免错误或不确定 teacher target 过度影响 student。

代码位置：

```text
src/losses/sa_cmd.py
confidence_weights
masked_kd_loss
```

### 3.4 SA-CMD：Sample Relation Distillation

除 logits 外，SA-CMD 约束 teacher/student 在 embedding space 中的样本关系。当前实现是 sample relation loss：

```text
L_REL = distance( relation(E_student), relation(E_teacher) )
```

直观上可理解为：

- teacher embedding 中样本之间的相对结构代表旧任务知识；
- student 学新任务时不应破坏旧表示几何关系；
- 这比只对齐 logits 更关注 representation preservation。

代码位置：

```text
src/losses/sa_cmd.py
sample_relation_loss
```

### 3.5 SA-CMD 总 loss

完整 SA-CMD：

```text
L = L_CE
  + lambda_kd * L_CKD
  + lambda_rel * L_REL
```

当前配置：

```text
lambda_kd = 1.0
lambda_rel = 1.0
temperature = 2.0
```

方法名：

```text
text_task_sa_cmd
```

### 3.6 Replay-enhanced SA-CMD

可选 replay 版本：

```text
text_task_sa_cmd_replay_kd
```

额外维护旧任务 replay memory，并在新任务训练时混入旧任务 replay batch。replay batch 上可做：

- hard CE；
- teacher KD；
- relation loss。

当前重要结论：

> Replay-enhanced SA-CMD did not consistently improve over no-replay SA-CMD under the current BERT-based MELD Task-IL setting.

因此论文里 replay 不应作为主贡献，只能作为可选 variant 或分析项。

## 4. S6：Audio-assisted Text-only Distillation

### 4.1 动机

直接 S5 text+audio e2e concat fusion 结果很差，说明把 raw audio 直接拼接进最终模型容易造成负迁移。更合理的做法是把 audio 用作训练阶段辅助信号：

```text
Teacher: text + audio
Student: text only
Inference: text only
```

这样 audio 不作为测试时必要输入，只在训练阶段通过 teacher 提供额外情绪信息。

可以在论文中写：

> Instead of directly fusing noisy acoustic features into the final classifier, we use audio as a training-time auxiliary teacher. A text-only student is distilled from a text-audio teacher, allowing acoustic cues to regularize textual representations without requiring audio at inference time.

### 4.2 S6 模型流程

训练阶段：

```text
Text -> BERT encoder --------\
                             -> Text+Audio Teacher -> logits_teacher, emb_teacher
Audio -> wav2vec2/HuBERT ----/

Text -> BERT student ---------> logits_student, emb_student

Loss:
  CE(student, y)
  + KD(student logits, teacher logits)
  + relation(student emb, teacher emb)
  + old-task KD/relation for continual learning
```

推理阶段：

```text
Text -> Text-only student -> prediction
Audio not required
```

### 4.3 S6 与 CMCRD 的关系

当前 S6 借鉴了 CMCRD 的“训练时多模态、测试时单模态”思想，但还不是完整 CMCRD。

当前已有：

- text+audio teacher；
- text-only student；
- teacher-student KD；
- confidence-aware relation distillation。

当前缺少：

- explicit cross-modal contrastive representation distillation；
- InfoNCE loss；
- modality-specific projection heads；
- teacher/student contrastive alignment objective。

后续若要更贴近 CMCRD，可以新增：

```text
L = CE
  + lambda_kd * KD
  + lambda_rel * relation
  + lambda_ctr * InfoNCE(student_text_repr, teacher_text_audio_repr)
```

## 5. 创新点写法

### 5.1 推荐贡献点

建议论文贡献点写成：

```text
1. We introduce a dialogue-level task-incremental formulation of MELD, where sentiment recognition, emotion classification, and emotion-shift detection are learned sequentially. This setting evaluates continual learning under conversational context and speaker-dependent emotional dynamics.

2. We propose SA-CMD, a confidence-aware relation distillation framework for continual conversational emotion recognition. SA-CMD emphasizes reliable teacher predictions and preserves sample-level relational structures in dialogue representations, extending conventional logit-based LwF.

3. We further explore an audio-assisted text-only distillation setting, where a text-audio teacher transfers multimodal emotional cues to a text-only student. This design avoids test-time dependency on acoustic inputs while leveraging audio as auxiliary supervision during training.
```

### 5.2 不建议这样写

当前不要写：

```text
Our method achieves state-of-the-art on MELD.
```

因为 BERT seed13 上 LwF 仍略高。

也不要写：

```text
Replay is the key contribution.
```

因为当前 replay 版本没有稳定提升。

更稳的表述：

```text
SA-CMD achieves competitive performance against a strong LwF baseline and outperforms several replay- and regularization-based continual learning baselines. The results suggest that confidence-aware relational distillation is a promising alternative to memory-heavy replay in dialogue-level MELD continual learning.
```

## 6. 实验设置

### 6.1 数据集

数据集：

```text
MELD
```

数据路径：

```text
/data2/yangxy/dataset/MELD/MELD.Raw
```

固定 task split：

```text
stl_task_splits
```

每个 task 的 train split：

```text
346 dialogues
```

dev/test：

```text
dev: sentiment 39, emotion 38, shift 37
test: sentiment 94, emotion 93, shift 93
```

### 6.2 任务

```text
Task 1: sentiment
Task 2: emotion
Task 3: shift
```

评估时对所有已学任务评估：

```text
after Task 1: evaluate sentiment
after Task 2: evaluate sentiment, emotion
after Task 3: evaluate sentiment, emotion, shift
```

### 6.3 Backbone

当前主要报告 BERT-base：

```text
/data2/yangxy/models/bert-base-uncased
```

原因：

- XLM-R-large 参数量大，MELD 数据量较小，可能过拟合；
- BERT-base 更轻，训练更快；
- 当前 BERT S3 结果整体更稳定。

XLM-R-large 作为对照：

```text
/data2/yangxy/models/xlm-roberta-large
```

S6 audio backbone：

```text
/data2/yangxy/models/wav2vec2-base
```

### 6.4 训练超参

BERT S3：

```text
epochs = 30
batch_size = 2
grad_accum_steps = 4
effective batch size = 8
lr = 2e-5
weight_decay = 0.01
max_length = 128
fp16 = true
seed = 13
```

XLM-R S3：

```text
batch_size = 1
grad_accum_steps = 8
```

S6：

```text
epochs = 30
teacher_epochs = 10
batch_size = 1
grad_accum_steps = 8
text backbone = bert-base-uncased
audio backbone = wav2vec2-base
max_audio_seconds = 4
```

### 6.5 评估指标

主指标：

```text
Final average weighted-F1
```

辅助指标：

```text
Final average accuracy
Forgetting
Retention
Macro-F1
Shift positive F1
```

论文中建议主表报告：

```text
Final Avg W-F1
Final Avg Acc
Forgetting
```

## 7. Baselines

### 7.1 Lower / Upper

```text
dlg_seq_ft
hier_bilstm joint
```

注意：当前 joint upper 低于 LwF，不可信，需要后续复查。

### 7.2 Distillation baseline

```text
dlg_seq_kd
```

即 LwF。

### 7.3 Replay baselines

```text
dlg_er
dlg_icarl
dlg_der
dlg_derpp
```

### 7.4 Parameter regularization baselines

```text
dlg_ewc
dlg_mas
dlg_si
```

### 7.5 Structure / pruning baseline

```text
dlg_packnet
```

当前实现是 PackNet-inspired parameter pruning/masking/freezing，不是严格 subnet-per-task eval。

### 7.6 Our variants

```text
text_task_sa_cmd
text_task_sa_cmd_freeze_old_heads
text_task_sa_cmd_replay_kd
text_task_sa_cmd_replay_kd_freeze_old_heads
prototype_nearest_klmap
diverse_klmap
hybrid_klmap
```

## 8. 当前实验结果

### 8.1 BERT-base S3 主结果

| 方法 | Final avg W-F1 | Final avg Acc | 备注 |
|---|---:|---:|---|
| `dlg_seq_kd` / LwF | **0.6232** | **0.6320** | 当前最高 |
| `text_task_sa_cmd_freeze_old_heads` | 0.6191 | 0.6275 | SA-CMD + freeze |
| `text_task_sa_cmd` | 0.6190 | 0.6274 | SA-CMD 本体 |
| `prototype/diverse/hybrid_klmap` | 0.6164 | 0.6263 | 旧结果 suspect，需重跑 |
| `text_task_sa_cmd_replay_kd random` | 0.6158 | 0.6250 | SA-CMD + replay |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 0.6134 | 0.6225 | replay + freeze |
| `dlg_icarl` NME | 0.6079 | 0.6050 | replay baseline |
| `dlg_derpp` | 0.5969 | 0.5990 | DER++ |
| `dlg_der` | 0.5943 | 0.5964 | DER |
| `hier_bilstm` joint | 0.5924 | 0.5914 | joint upper 偏低 |
| `dlg_er` | 0.5912 | 0.5913 | ER |
| `dlg_packnet` | 0.5622 | 0.5589 | PackNet |
| `dlg_mas` | 0.5534 | 0.5485 | MAS |
| `dlg_seq_ft` | 0.5174 | 0.5096 | lower |
| `dlg_ewc` | 0.5168 | 0.5183 | EWC |
| `dlg_si` | 0.3943 | 0.4433 | SI，日志出现 `nan` |

结论：

- LwF 当前最强。
- SA-CMD 与 LwF 接近，强于多个 replay/regularization baseline。
- replay/freeze 没有带来稳定提升。
- 当前上界偏低，需要复查。

### 8.2 XLM-R-large S3 结果

| 方法 | Final avg W-F1 | Final avg Acc | 备注 |
|---|---:|---:|---|
| `dlg_icarl` NME | **0.6147** | **0.6209** | XLM-R seed13 最强 |
| `text_task_sa_cmd_replay_kd random` | 0.6084 | - | 主方法旧结果 |
| `prototype_nearest/diverse` | 0.6079 | 0.6185 | 相同 |
| `prototype/diverse/hybrid_klmap` | 0.6076 | 0.6177 | 旧结果 suspect |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 0.6042 | - | freeze 无明显帮助 |
| `dlg_seq_kd` | 0.6002 | - | KD baseline |
| `hier_bilstm` joint | 0.5964 | 0.5987 | 上界偏低 |
| `dlg_er` | 0.5943 | - | ER |
| `dlg_seq_ft` | 0.4716 | 0.4805 | lower |

结论：

- XLM-R 下 iCaRL NME 最强。
- XLM-R 参数大，训练慢，不作为当前主线。
- BERT 结果更适合作为论文主表。

### 8.3 S5 fixed-feature 结果

| 方法 | Final avg W-F1 | Final avg Acc | 备注 |
|---|---:|---:|---|
| `dlg_task_sa_cmd` | **0.5233** | **0.5315** | fixed S5 最好 |
| `dlg_task_pg_trd` | 0.5171 | 0.5290 | 接近 KD |
| `dlg_seq_kd` | 0.5171 | 0.5273 | KD |
| `dlg_seq_ft` | 0.4937 | 0.5048 | lower |

### 8.4 S5 e2e 结果

| 方法 | Final avg W-F1 | Final avg Acc | 备注 |
|---|---:|---:|---|
| `s5_e2e_ta_seq_kd` | **0.3408** | 0.4915 | 三者最高但仍失败 |
| `s5_e2e_ta_seq_ft` | 0.3369 | 0.4901 | 没学好 |
| `s5_e2e_ta_sa_cmd` | 0.3272 | 0.4910 | 比 KD 更差 |

结论：

- S5 e2e 当前不能作为主结果。
- 问题不是 forgetting，而是基础 text+audio e2e 学不起来。
- S6 是对 S5 的修正方向。

### 8.5 S6 当前状态

截至本文档创建时，S6 wav2vec2 两条实验仍在跑：

```text
s6_text_student_ta_teacher
s6_text_student_ta_teacher_sa
```

查看：

```bash
pgrep -af 'run_dialogue_text_audio_teacher_student_stl.py|s6_text_student'
tail -80 outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_gpu4.log
tail -80 outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_sa_gpu5.log
```

注意：日志名里的 gpu 编号和实际 `--gpu-id` 有过不一致，以进程参数为准。

## 9. 可画图内容

### 9.1 Figure 1：MELD Task-IL setting

画法：

```text
Dialogue utterances
  -> Task 1 sentiment
  -> Task 2 emotion
  -> Task 3 emotion shift
```

突出：

- sequential learning；
- after each task evaluate all learned tasks；
- dialogue-level sequence labeling。

### 9.2 Figure 2：SA-CMD framework

建议画：

```text
Teacher
  old-task logits -> confidence weights -> weighted KD
  embeddings -> relation matrix

Student
  current-task CE
  old-task logits -> weighted KD
  embeddings -> relation matrix

Loss = CE + CKD + relation
```

配色：

- teacher 蓝色；
- student 橙色；
- loss 绿色；
- confidence/relation 模块紫色。

### 9.3 Figure 3：S6 audio-assisted text-only distillation

画法：

```text
Training:
Text + Audio -> Teacher
Text only    -> Student
Teacher -> KD/relation -> Student

Inference:
Text only -> Student -> prediction
```

重点标注：

```text
Audio is used only during training.
```

### 9.4 Figure 4：主结果柱状图

只放核心方法：

```text
seq_ft
ER
DER
DER++
iCaRL
PackNet
SA-CMD
LwF
```

Y 轴：

```text
Final Avg Weighted-F1
```

注意：LwF 最高，不要画成 ours 最高的叙事。可以强调 SA-CMD 竞争性和优于 replay/regularization baseline。

### 9.5 Figure 5：消融图

等补完 component ablation 后画：

```text
KD only
KD + confidence
KD + relation
KD + confidence + relation
```

如果 full 仍低于 LwF，则消融图不要强行画成递增贡献，改成表格更稳。

## 10. 当前不足与后续改进

### 10.1 必须补的核心消融

现在缺真正 component-level ablation。需要新增并跑：

```text
KD only / LwF
KD + confidence
KD + relation
KD + confidence + relation
```

当前已有的 replay/freeze ablation 不能说明 confidence/relation 各自贡献。

### 10.2 多 seed

至少跑：

```text
seed 13 / 21 / 42
```

方法：

```text
dlg_seq_kd
text_task_sa_cmd
text_task_sa_cmd_replay_kd
```

### 10.3 修正 joint upper

当前 joint upper 低于 LwF，不符合直觉。需要查：

- joint 训练是否 task sampling 不均；
- head/loss 是否正确；
- eval 是否和 sequential 一致；
- epoch 是否足够；
- 是否需要 class-balanced loss。

### 10.4 KLMap 重跑

旧 KLMap 结果 suspect。已加入 replay selection digest，重跑时建议：

```text
memory_per_class = 30 or 50
klmap_dim = 50
```

### 10.5 S6 改进

当前 S6 可跑，但还需：

- 拆出真正 text-only student，降低显存；
- 加 InfoNCE / contrastive representation distillation；
- 尝试 HuBERT；
- 调 `lambda_ta_kd`, `lambda_ta_rel`；
- 评估 teacher 本身的 text+audio 上限。

## 11. 论文实验表建议

### Table 1：Main results on MELD Task-IL

列：

```text
Method | Type | Final Avg W-F1 | Final Avg Acc | Forgetting
```

行：

```text
seq_ft
ER
iCaRL
DER
DER++
PackNet
EWC
MAS
SI
LwF
SA-CMD
SA-CMD + replay
```

### Table 2：Ablation of SA-CMD

列：

```text
Confidence | Relation | Replay | W-F1 | Acc
```

行：

```text
KD only
KD + confidence
KD + relation
Full SA-CMD
Full SA-CMD + replay
```

### Table 3：Backbone comparison

列：

```text
Backbone | Method | W-F1 | Acc
```

行：

```text
BERT-base LwF
BERT-base SA-CMD
XLM-R LwF
XLM-R SA-CMD
```

### Table 4：Audio-assisted distillation

等 S6 跑完后填：

```text
Text-only baseline
Text+audio direct fusion S5
S6 teacher-student
S6 teacher-student + SA-CMD
```

## 12. 可直接写进论文的结果分析草稿

### 12.1 BERT S3 主结果分析

> On the BERT-base backbone, SA-CMD achieves a final average weighted-F1 of 0.6190, substantially outperforming replay-based baselines such as ER, iCaRL, DER, DER++, and PackNet, as well as regularization-based methods including EWC, MAS, and SI. Although LwF obtains the best result under the current seed, the small performance gap indicates that confidence-aware relational distillation is competitive with strong logit distillation while avoiding reliance on memory replay.

### 12.2 Replay 结果分析

> Replay-enhanced variants do not consistently improve over the no-replay SA-CMD variant. This suggests that, under the current dialogue-level MELD Task-IL setting, the main performance gain comes from distillation-based representation preservation rather than memory replay. Replay selection and replay loss design remain important directions for future improvement.

### 12.3 S5/S6 分析

> Direct end-to-end text-audio fusion performs poorly in our preliminary S5 experiments, indicating that raw acoustic inputs may introduce noisy or unstable signals when directly fused into the classifier. We therefore explore an audio-assisted text-only distillation setting, where a wav2vec2-based text-audio teacher transfers acoustic emotional cues to a text-only student. This design aims to leverage audio during training while preserving robust text-only inference.

## 13. 关键代码与命令

主 S3：

```bash
python scripts/run_dialogue_task_stl.py --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd --backbone /data2/yangxy/models/bert-base-uncased
```

S6：

```bash
python scripts/run_dialogue_text_audio_teacher_student_stl.py --config configs/dialogue_task_stl_v2.yaml --methods s6_text_student_ta_teacher_sa --backbone /data2/yangxy/models/bert-base-uncased --audio-encoder pretrained --audio-backbone /data2/yangxy/models/wav2vec2-base
```

常用 nohup：

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 2 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd --epochs 30 --batch-size 2 --grad-accum-steps 4 --backbone /data2/yangxy/models/bert-base-uncased --fp16 --run-name s3_bert_base_text_task_sa_cmd_seed13_gpu2_20260522 > outputs/command_logs/s3_bert_base_text_task_sa_cmd_gpu2.log 2>&1 &
```

