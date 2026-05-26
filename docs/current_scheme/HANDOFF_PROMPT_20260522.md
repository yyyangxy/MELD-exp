# MELD Continual Learning 实验交接 Prompt（2026-05-22）

这份文档用于把当前 MELD 连续学习项目完整交接给一个没接触过项目的新同学或新的 Codex/ChatGPT 对话。请先完整阅读，再动代码或启动实验。

## 0. 一句话概况

当前项目在 MELD 上做 dialogue-level continual / task-incremental emotion recognition。主线是 **S3 Text-Only Task-IL**，任务顺序为：

```text
Task 1: sentiment
Task 2: emotion
Task 3: emotion shift
```

当前最稳定的实验设置是 **BERT-base text-only S3**。方法核心不是 replay，而是 **SA-CMD：confidence-aware KD + sample relation distillation**。目前结果显示 SA-CMD 接近但没有超过 LwF；replay 在当前配置下没有稳定收益。正在探索 **S6：text+audio teacher -> text-only student**，用 wav2vec2 audio teacher 辅助 text-only student。

项目路径：

```text
/data2/yangxy/MELD/MELD-exp
```

数据路径：

```text
/data2/yangxy/dataset/MELD/MELD.Raw
```

本地模型路径：

```text
/data2/yangxy/models/bert-base-uncased
/data2/yangxy/models/xlm-roberta-large
/data2/yangxy/models/wav2vec2-base
```

## 1. 当前最重要的结论

1. **S3-BERT 是当前主线。**
   - LwF / `dlg_seq_kd`: 0.6232 W-F1，是当前 BERT 已完成结果里最高。
   - `text_task_sa_cmd`: 0.6190 W-F1，接近 LwF，强于 iCaRL/DER/PackNet/EWC/MAS/SI。
   - `text_task_sa_cmd_replay_kd random`: 0.6158 W-F1，replay 没有提升。

2. **当前 replay 不应作为主要创新点。**
   - ER、DER、DER++、iCaRL、PackNet 都没有超过 LwF/SA-CMD。
   - KLMap 三种策略之前结果完全一样，后来确认那批结果不能作为策略比较，需要重跑。

3. **S5 end-to-end text+audio 目前失败。**
   - `s5_e2e_ta_seq_kd` 只有 0.3408 W-F1。
   - 原因不是遗忘，而是前两个任务本身没学好。
   - 原 S5 使用 raw waveform + 简单 `RawAudioEncoder`，不接近 MELD 原文的 openSMILE 特征，也不是强 audio encoder。

4. **S6 是当前正在尝试的新方向。**
   - 训练 text+audio teacher，评估 text-only student。
   - 已支持 wav2vec2/HuBERT 类 pretrained audio encoder。
   - 目前有两条 S6 wav2vec2 实验正在跑。

## 2. 环境和通用注意事项

当前 conda 环境通常是：

```bash
conda activate yangxinyao
```

由于服务器 shell/sandbox 特性，之前 Codex 运行 shell 经常需要 escalated 权限。人工在终端跑不需要管这个。

常用 nohup 环境变量：

```bash
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

原因：

- 避免 tokenizer / rayon thread pool panic。
- 降低多进程 CPU 线程争用。
- 缓解 CUDA memory fragmentation。

当前 git worktree 很脏，有大量历史修改和未跟踪文件。**不要随便 reset / checkout / 删除文件**。还有一些早期误粘命令产生的未跟踪空文件，如 `--batch-size`、`--config`、`--methods` 等，不要在没确认前清理。

## 3. 实验设置

### 3.1 S3：Dialogue Text-Only Task-STL

核心设置：

```text
输入: dialogue utterance text
输出: 每个 utterance 的 task label
任务: sentiment -> emotion -> shift
评估: task id known，按已学任务分别评估
主配置: configs/dialogue_task_stl_v2.yaml
主脚本: scripts/run_dialogue_task_stl.py
主 runner: src/train/dialogue_text_task_runner.py
```

任务拆分使用固定 dialogue split：

```text
data.stl_task_split_root: stl_task_splits
train: 每个 task 346 dialogues
dev: sentiment 39, emotion 38, shift 37
test: sentiment 94, emotion 93, shift 93
```

默认训练细节：

```text
seed: 13
epochs: 30
batch-size: BERT 通常 2，XLM-R 通常 1
grad-accum-steps: BERT 通常 4，XLM-R 通常 8
fp16: true
max_length: 128
lr: 2e-5
weight_decay: 0.01
```

指标：

```text
accuracy
weighted_f1
macro_f1
positive_f1_for_shift
final_avg_weighted_f1
final_avg_accuracy
forgetting
retention
```

结果 CSV 路径格式：

```text
outputs/runs/dialogue_text_task_stl/<timestamp>_<run-name>/results/dialogue_text_task_stl_results.csv
```

### 3.2 S5：Text+Audio Task-STL

有两个 S5 分支：

1. **S5 fixed feature**
   - 脚本：`scripts/run_dialogue_multimodal_task_stl.py`
   - runner：`src/train/dialogue_task_runner.py`
   - 使用已提取的 text/audio feature。
   - 结果比 e2e 稳定，但整体不如 S3-BERT。

2. **S5 end-to-end text+audio**
   - 脚本：`scripts/run_dialogue_text_audio_task_e2e_stl.py`
   - runner：`src/train/dialogue_text_audio_task_e2e_runner.py`
   - 原来 audio 是 raw waveform -> `RawAudioEncoder`，效果很差。
   - 现在该模型已支持 `audio_encoder_type=pretrained`，可接 wav2vec2/HuBERT，但 S5 e2e 不是当前主线。

### 3.3 S6：Text+Audio Teacher -> Text-Only Student

这是新方向，目标是用 audio 辅助 text，而不是推理时依赖 audio。

```text
Teacher: text + audio
Student: text only
Audio encoder: wav2vec2/HuBERT pretrained
评估: text-only student
```

脚本：

```text
scripts/run_dialogue_text_audio_teacher_student_stl.py
```

runner：

```text
src/train/dialogue_text_audio_teacher_student_runner.py
```

支持方法：

```text
s6_text_student_ta_teacher
s6_text_student_ta_teacher_sa
```

区别：

- `s6_text_student_ta_teacher`: CE + text/audio teacher KD
- `s6_text_student_ta_teacher_sa`: 上面再加 teacher confidence weighting + relation distillation

注意：S6 目前是初版，不是完整 CMCRD。它没有显式 InfoNCE / cross-modal contrastive loss；只是 text+audio teacher 到 text-only student 的 KD/relation 蒸馏。后续如果要更贴近 CMCRD，需要加 contrastive representation distillation。

## 4. 方法和 baseline

### 4.1 LwF / KD baseline

方法名：

```text
dlg_seq_kd
```

形式：

```text
L = CE(new task) + lambda_kd * KD(old task logits)
```

当前 BERT 上最强：0.6232 W-F1。

### 4.2 SA-CMD：当前方法核心

方法名：

```text
text_task_sa_cmd
```

核心思想：

```text
L = CE
  + lambda_kd * confidence-weighted KD
  + lambda_rel * sample relation distillation
```

和 LwF 的区别：

- LwF 只蒸馏旧任务 logits。
- SA-CMD 根据 teacher confidence 给样本加权。
- SA-CMD 额外保持 teacher/student embedding 的样本关系结构。

当前已有消融不是完整 component-level ablation，只是 replay/freeze 变体：

```text
text_task_sa_cmd
text_task_sa_cmd_freeze_old_heads
text_task_sa_cmd_replay_kd
text_task_sa_cmd_replay_kd_freeze_old_heads
```

缺少真正核心消融，后续应新增：

```text
text_task_sa_cmd_no_rel      # KD + confidence, no relation
text_task_sa_cmd_no_conf     # KD + relation, no confidence
KD only                      # 对齐 dlg_seq_kd / LwF
```

### 4.3 Replay variants

普通 replay：

```text
dlg_er
dlg_random_replay
```

iCaRL：

```text
dlg_icarl
```

DER / DER++：

```text
dlg_der
dlg_derpp
```

PackNet：

```text
dlg_packnet
```

注意：当前 PackNet 是 parameter mask / prune / freeze 风格的实现，接近原论文思想，但不是完整独立 subnet-per-task evaluation 的严格 PackNet。

### 4.4 KLMap replay selection

支持策略：

```text
prototype_nearest_klmap
diverse_klmap
hybrid_klmap
```

参数：

```text
--klmap-dim 50
```

作用：先训练小 MLP 将高维 embedding 降到低维选择特征，再做 nearest/diverse/hybrid sample selection。

重要状态：

- 之前 BERT 和 XLM-R 的三种 KLMap 结果完全一样。
- 后来检查发现 checkpoint 也完全一样，因此那批结果不能作为有效策略比较。
- 已修改代码：以后每次 replay selection 会保存：

```text
outputs/runs/.../replay_selections/*.json
```

日志里也会写 `digest=...`。重跑后可用 digest 判断三种策略是否真的选了不同样本。

建议重跑 KLMap 时把 `memory-per-class` 降到 30 或 50；100/class 覆盖样本太多，选择策略差异会被冲淡。

### 4.5 正则化 baseline

方法：

```text
dlg_ewc
dlg_mas
dlg_si
```

重要参数：

```text
--regularizer-scope all 或 non_encoder
--importance-max-batches 10
```

XLM-R large 上 `all` 容易爆显存；BERT 可以用 `all`。当前 BERT 结果里 MAS 最好，EWC 接近下界，SI 出现 `nan`，不可靠。

## 5. 当前结果汇总

### 5.1 S3 Text-Only：BERT-base

| 方法 | Final avg W-F1 | Final avg acc | 备注 |
|---|---:|---:|---|
| `dlg_seq_kd` / LwF | **0.6232** | **0.6320** | 当前 BERT 最强 |
| `text_task_sa_cmd_freeze_old_heads` | 0.6191 | 0.6275 | SA-CMD + freeze |
| `text_task_sa_cmd` | 0.6190 | 0.6274 | SA-CMD，无 replay |
| `prototype/diverse/hybrid_klmap` | 0.6164 | 0.6263 | 旧结果 suspect，需重跑 |
| `text_task_sa_cmd_replay_kd random` | 0.6158 | 0.6250 | 主 replay 版本 |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 0.6134 | 0.6225 | replay + freeze 更差 |
| `dlg_icarl` NME | 0.6079 | 0.6050 | 低于 SA-CMD |
| `dlg_derpp` | 0.5969 | 0.5990 | DER 系最好 |
| `dlg_der` | 0.5943 | 0.5964 | 略低于 DER++ |
| `hier_bilstm` joint | 0.5924 | 0.5914 | “上界”偏低，可疑 |
| `dlg_er` | 0.5912 | 0.5913 | BERT ER 已完成 |
| `dlg_packnet` | 0.5622 | 0.5589 | 偏弱 |
| `dlg_mas` | 0.5534 | 0.5485 | 正则化 baseline 中最好 |
| `dlg_seq_ft` | 0.5174 | 0.5096 | 下界 |
| `dlg_ewc` | 0.5168 | 0.5183 | 接近下界 |
| `dlg_si` | 0.3943 | 0.4433 | 日志有 `nan`，不可靠 |

关键路径：

```text
outputs/runs/dialogue_text_task_stl/20260521_214909_s3_bert_base_lwf_seq_kd_seed13_gpu2_20260521_seed13/results/dialogue_text_task_stl_results.csv
outputs/runs/dialogue_text_task_stl/20260521_233955_s3_bert_base_ablate_no_replay_no_kd_seed13_gpu2_20260521_seed13/results/dialogue_text_task_stl_results.csv
outputs/runs/dialogue_text_task_stl/20260522_101804_s3_bert_base_er_seed13_gpu3_20260522_seed13/results/dialogue_text_task_stl_results.csv
```

### 5.2 S3 Text-Only：XLM-R-large

| 方法 | Final avg W-F1 | Final avg acc | 备注 |
|---|---:|---:|---|
| `dlg_icarl` NME | **0.6147** | **0.6209** | XLM-R seed13 最强 |
| `text_task_sa_cmd_replay_kd random` | 0.6084 | - | 主方法旧主线 |
| `prototype_nearest/diverse` | 0.6079 | 0.6185 | 两者相同 |
| `prototype/diverse/hybrid_klmap` | 0.6076 | 0.6177 | 旧结果 suspect |
| `text_task_sa_cmd_replay_kd_freeze_old_heads` | 0.6042 | - | freeze 无明显帮助 |
| `dlg_seq_kd` | 0.6002 | - | KD baseline |
| `hier_bilstm` joint | 0.5964 | 0.5987 | 上界仍偏低 |
| `dlg_er` | 0.5943 | - | replay baseline |
| `dlg_seq_ft` | 0.4716 | 0.4805 | 下界 |
| `dlg_ewc` non-encoder | 0.4435 | 0.5059 | 不强 |
| `dlg_si` non-encoder | 0.4350 | 0.5014 | 不强 |
| `dlg_mas` non-encoder | 0.3366 | 0.4896 | 很差 |

### 5.3 S5 Text+Audio：Fixed Feature

| 方法 | Final avg W-F1 | Final avg acc | 备注 |
|---|---:|---:|---|
| `dlg_task_sa_cmd` | **0.5233** | **0.5315** | S5 fixed 最好 |
| `dlg_task_pg_trd` | 0.5171 | 0.5290 | 接近 KD |
| `dlg_seq_kd` | 0.5171 | 0.5273 | KD 有提升 |
| `dlg_seq_ft` | 0.4937 | 0.5048 | 下界 |

### 5.4 S5 Text+Audio：End-to-End

| 方法 | Final avg W-F1 | Final avg acc | 备注 |
|---|---:|---:|---|
| `s5_e2e_ta_seq_kd` | **0.3408** | 0.4915 | 三者中最高，但整体失败 |
| `s5_e2e_ta_seq_ft` | 0.3369 | 0.4901 | 基本没学好 |
| `s5_e2e_ta_sa_cmd` | 0.3272 | 0.4910 | 比 KD 更差 |

### 5.5 S6 当前状态

截至 2026-05-22 12:13 左右，S6 wav2vec2 两条还在跑：

```text
PID 75395: --gpu-id 5, method=s6_text_student_ta_teacher
log: outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_gpu4.log
进度: emotion task student epoch 7/30 左右

PID 80455: --gpu-id 6, method=s6_text_student_ta_teacher_sa
log: outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_sa_gpu5.log
进度: emotion task teacher epoch 7/10 左右
```

注意：run-name/log 名里的 gpu4/gpu5 和实际 `--gpu-id 5/6` 有命名偏差，以进程参数为准。

检查命令：

```bash
pgrep -af 'run_dialogue_text_audio_teacher_student_stl.py|s6_text_student'
tail -80 outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_gpu4.log
tail -80 outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_sa_gpu5.log
```

## 6. 代码地图

### 6.1 数据和 split

```text
src/data/meld_csv.py
```

读取 MELD CSV。这里修过文本 mojibake，5 月 20 日前的结果大多是探索性，不建议作为正式结论。

```text
src/data/stl_task_splits.py
scripts/prepare_stl_task_splits.py
```

固定 dialogue-level task split。

### 6.2 S3 text-only

```text
scripts/run_dialogue_task_stl.py
src/train/dialogue_text_task_runner.py
configs/dialogue_task_stl_v2.yaml
```

`dialogue_text_task_runner.py` 包含：

- `XLMRDialogueTaskModel`，名字保留 XLMR 但可用 BERT。
- LwF / KD。
- SA-CMD。
- replay / iCaRL / DER / DER++ / PackNet。
- EWC / MAS / SI。
- KLMap replay selection。
- replay selection digest 记录。

### 6.3 Loss

```text
src/losses/sa_cmd.py
src/losses/task_relation.py
```

关键函数：

- `masked_kd_loss`
- `confidence_weights`
- `sample_relation_loss`
- task relation / prototype relation 相关 loss

### 6.4 S5 fixed feature

```text
scripts/run_dialogue_multimodal_task_stl.py
src/train/dialogue_task_runner.py
src/models/dialogue_model.py
configs/dialogue_multimodal_task_stl_v2.yaml
```

### 6.5 S5 e2e text+audio

```text
scripts/run_dialogue_text_audio_task_e2e_stl.py
src/train/dialogue_text_audio_task_e2e_runner.py
src/train/dialogue_modality_e2e_runner.py
```

`dialogue_modality_e2e_runner.py` 里有：

- `RawAudioEncoder`
- `_load_audio`

`dialogue_text_audio_task_e2e_runner.py` 现在支持：

- raw audio encoder
- pretrained audio encoder，传 wav2vec2/HuBERT 路径

### 6.6 S6 text/audio teacher-student

```text
scripts/run_dialogue_text_audio_teacher_student_stl.py
src/train/dialogue_text_audio_teacher_student_runner.py
```

重要 caveat：

- S6 student 当前也会实例化 audio encoder（因为复用了同一个 `DialogueTextAudioTaskE2EModel`），但 student 训练/评估时 `audio_enabled=False`，所以不会实际 forward audio。这样会浪费显存，后续最好拆成真正 text-only student model。
- S6 不是完整 CMCRD，还没有 contrastive InfoNCE。

### 6.7 wav2vec2 feature extractor

```text
src/features/extract_audio.py
```

已有离线 wav2vec2 feature extractor，默认 `facebook/wav2vec2-base`，但当前 S6 是在线端到端使用 wav2vec2 model，不走这个离线 feature extractor。

## 7. 常用命令

### 7.1 查看结果

```bash
tail -1 outputs/runs/dialogue_text_task_stl/<run>/results/dialogue_text_task_stl_results.csv
```

批量找结果：

```bash
find outputs/runs -path '*results/*.csv' -newermt '2026-05-20' | sort
```

查看运行进程：

```bash
pgrep -af 'run_dialogue_task_stl.py|run_dialogue_text_audio_task_e2e_stl.py|run_dialogue_text_audio_teacher_student_stl.py'
gpustat
```

### 7.2 S3 BERT LwF

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 2 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods dlg_seq_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --backbone /data2/yangxy/models/bert-base-uncased --fp16 --run-name s3_bert_base_lwf_seq_kd_seed13_gpu2_20260522 > outputs/command_logs/s3_bert_base_lwf_gpu2.log 2>&1 &
```

### 7.3 S3 BERT SA-CMD

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 2 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd --epochs 30 --batch-size 2 --grad-accum-steps 4 --backbone /data2/yangxy/models/bert-base-uncased --fp16 --run-name s3_bert_base_text_task_sa_cmd_seed13_gpu2_20260522 > outputs/command_logs/s3_bert_base_text_task_sa_cmd_gpu2.log 2>&1 &
```

### 7.4 S3 BERT replay-KD

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 3 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 100 --replay-batch-kd --replay-strategy random --backbone /data2/yangxy/models/bert-base-uncased --fp16 --run-name s3_bert_base_ours_replay_kd_random_seed13_gpu3_20260522 > outputs/command_logs/s3_bert_base_ours_replay_kd_random_gpu3.log 2>&1 &
```

### 7.5 重跑 KLMap（建议 memory 30 或 50）

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_task_stl.py --gpu-id 0 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods text_task_sa_cmd_replay_kd --epochs 30 --batch-size 2 --grad-accum-steps 4 --memory-per-class 50 --replay-batch-kd --replay-strategy prototype_nearest_klmap --klmap-dim 50 --backbone /data2/yangxy/models/bert-base-uncased --fp16 --run-name s3_bert_base_ours_proto_klmap_mem50_seed13_gpu0_20260522 > outputs/command_logs/s3_bert_base_proto_klmap_mem50_gpu0.log 2>&1 &
```

把 `--replay-strategy` 换成：

```text
diverse_klmap
hybrid_klmap
```

跑完检查：

```bash
find outputs/runs/dialogue_text_task_stl -path '*replay_selections/*.json' | sort
```

### 7.6 S6 wav2vec2 teacher-student

非 SA：

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_text_audio_teacher_student_stl.py --gpu-id 5 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods s6_text_student_ta_teacher --epochs 30 --teacher-epochs 10 --batch-size 1 --grad-accum-steps 8 --backbone /data2/yangxy/models/bert-base-uncased --audio-encoder pretrained --audio-backbone /data2/yangxy/models/wav2vec2-base --max-audio-seconds 4 --fp16 --run-name s6_bert_base_wav2vec2_text_student_ta_teacher_seed13_gpu5_20260522 > outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_gpu5.log 2>&1 &
```

SA：

```bash
mkdir -p outputs/command_logs && nohup env TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dialogue_text_audio_teacher_student_stl.py --gpu-id 6 --seed 13 --config configs/dialogue_task_stl_v2.yaml --methods s6_text_student_ta_teacher_sa --epochs 30 --teacher-epochs 10 --batch-size 1 --grad-accum-steps 8 --backbone /data2/yangxy/models/bert-base-uncased --audio-encoder pretrained --audio-backbone /data2/yangxy/models/wav2vec2-base --max-audio-seconds 4 --fp16 --run-name s6_bert_base_wav2vec2_text_student_ta_teacher_sa_seed13_gpu6_20260522 > outputs/command_logs/s6_bert_base_wav2vec2_text_student_ta_teacher_sa_gpu6.log 2>&1 &
```

## 8. 论文叙事建议

不要把当前工作写成“提出 replay 方法”。更稳的创新点是：

1. **MELD dialogue-level Task-IL 设置。**
   - 将 MELD 构造成 sentiment -> emotion -> shift 的任务增量学习问题。
   - 评估 forgetting/retention/final avg。

2. **SA-CMD：confidence-aware relation distillation。**
   - 相比 LwF，不把所有 teacher prediction 等权看待。
   - teacher 高置信样本更强蒸馏，低置信样本减弱。
   - 保持 teacher/student dialogue representation 的 sample relation。

3. **Audio-assisted text-only distillation（S6）。**
   - audio 作为训练阶段 teacher signal。
   - 推理/评估仍 text-only，避免 MELD audio 噪声直接污染预测。
   - 这更接近 CMCRD 思路，但当前还不是完整 CMCRD。

贡献点可以写：

```text
1. We formulate MELD emotion recognition as a dialogue-level task-incremental learning problem over sentiment, emotion, and emotion-shift tasks.

2. We propose SA-CMD, a confidence-aware relation distillation framework that emphasizes reliable teacher predictions and preserves sample-level relational structure in dialogue representations.

3. We extend SA-CMD to an audio-assisted text-only distillation setting, where a text-audio teacher transfers multimodal emotional cues to a text-only student without requiring audio at inference time.
```

结果表述要谨慎：

- 可以说 SA-CMD competitive with LwF，并超过多种 replay/regularization baseline。
- 不能说当前 SA-CMD 全面超过 LwF，因为 BERT seed13 上 LwF 仍更高。
- replay 目前不要作为核心贡献；它是 optional module。

## 9. 后续必须做的实验

优先级从高到低：

1. **补真正的 SA-CMD component ablation。**
   - `KD only`: LwF
   - `KD + confidence`
   - `KD + relation`
   - `KD + confidence + relation`: full SA-CMD

2. **多 seed 验证。**
   - 至少 `seed 13/21/42`
   - 方法：`dlg_seq_kd`, `text_task_sa_cmd`, `text_task_sa_cmd_replay_kd`

3. **修/解释 joint upper。**
   - 当前 BERT/XLM-R joint upper 都低于 LwF，不正常。
   - 需要确认 joint 训练是否 epoch、sampler、loss、task head、评估逻辑有问题。

4. **重跑 KLMap。**
   - 使用新代码的 replay digest。
   - memory-per-class 建议 30/50。
   - 确认 prototype/diverse/hybrid 选择 ID 不同。

5. **等待并分析 S6 wav2vec2。**
   - 比较 `s6_text_student_ta_teacher` vs `s6_text_student_ta_teacher_sa`。
   - 如果 S6 比 S3 text-only 高，说明 audio teacher 有用。
   - 如果不高，考虑换 HuBERT 或加真正 CMCRD contrastive loss。

6. **S6 代码优化。**
   - 拆出真正 text-only student，避免 student 也加载 audio encoder。
   - 给 S6 增加 `lambda_ta_kd/lambda_ta_rel` 网格。
   - 加 InfoNCE / contrastive representation distillation，使其更接近 CMCRD。

## 10. 当前风险点

1. **上界不可信。**
   - joint upper 低于 LwF，论文里不能直接用作理论上界。

2. **KLMap 旧结果不可用。**
   - checkpoint 完全一样，必须重跑。

3. **S5 e2e 不可靠。**
   - raw audio encoder 太弱；不能用它证明 audio 没用。

4. **S6 仍是初版。**
   - 当前不是完整 CMCRD。
   - student 浪费显存加载 audio encoder。

5. **repo 状态复杂。**
   - 有大量未跟踪和已修改文件。
   - 不要做 destructive git 操作。

## 11. 新同学上手顺序

1. 进入项目：

```bash
cd /data2/yangxy/MELD/MELD-exp
conda activate yangxinyao
```

2. 看当前运行：

```bash
gpustat
pgrep -af 'run_dialogue'
```

3. 读主代码：

```text
src/train/dialogue_text_task_runner.py
src/losses/sa_cmd.py
scripts/run_dialogue_task_stl.py
src/train/dialogue_text_audio_teacher_student_runner.py
```

4. 先复现/确认 BERT S3 表。

5. 等 S6 跑完，整理结果。

6. 再做 component ablation 和多 seed。

