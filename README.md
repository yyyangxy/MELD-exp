# MELD-exp Baseline

This directory contains runnable baselines for MELD sequence task learning and CMCRD-inspired modality-sequential learning.

Implemented main tasks:

- `sentiment`: 3-way sentiment recognition.
- `emotion`: 7-way emotion recognition.
- `shift`: same-speaker emotion shift detection, built independently inside each split.

Implemented methods:

Task-STL:

- `joint`
- `seq_ft`
- `lwf`
- `random_replay`
- `prototype_replay`
- `proto_replay_kd`

Modality-STL:

- `mod_seq_ft`
- `mod_seq_kd`
- `prototype_replay`
- `cmcrd_ours`

## Feasibility Notes

The full plan is feasible in stages. The local repo only has MELD CSV annotations, so the Task-STL baseline does not depend on missing feature pickles or local MP4 files. Modality-STL expects pre-extracted `.npy` features and does not extract audio/visual features from MP4 during training.

The current baseline uses CSV text directly:

```text
token ids -> embedding/BiLSTM -> speaker embedding -> concat MLP -> task head
```

This is enough to validate split handling, shift labels, task order, evaluation tables, replay memory, prototype selection, and KD plumbing. The CMCRD-style runner then uses feature-level text/audio/visual inputs to test modality retention.

## Setup

```bash
cd /mnt/d/code/MELD/MELD-exp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default config uses `data.data_root_mode: auto`, preferring:

```text
/data2/yangxy/dataset/MELD/MELD.Raw
```

and falling back to:

```text
../MELD-master/data/MELD
```

## Quick Checks

This check only needs the Python standard library:

```bash
python3 scripts/smoke_check.py --config configs/main_stl.yaml
```

Expected CSV row counts:

```text
train=9989, dev=1109, test=2610
```

Prepare shift-label CSVs:

```bash
python3 scripts/prepare_shift_labels.py --config configs/main_stl.yaml
```

## Run Main Baselines

```bash
python3 scripts/run_main_stl.py --config configs/main_stl.yaml --method joint
python3 scripts/run_main_stl.py --config configs/main_stl.yaml --method seq_ft
python3 scripts/run_main_stl.py --config configs/main_stl.yaml --method lwf
python3 scripts/run_main_stl.py --config configs/main_stl.yaml --method random_replay
python3 scripts/run_main_stl.py --config configs/main_stl.yaml --method prototype_replay
python3 scripts/run_main_stl.py --config configs/main_stl.yaml --method proto_replay_kd
```

Or run all:

```bash
python3 scripts/run_baselines.py --config configs/main_stl.yaml
```

Results are appended to:

```text
outputs/results/main_stl_results.csv
```

Checkpoints and logs are written to:

```text
outputs/checkpoints/
outputs/logs/
```

## Run Modality-Sequential Baselines

Modality-STL fixes the task to emotion recognition and trains stages in this order:

```text
Text -> Text+Audio -> Text+Audio+Visual
```

Required feature cache layout:

```text
outputs/features/{split}/{modality}/{diaX_uttY}.npy
```

Default dimensions in `configs/modality_stl.yaml` are:

```text
text=256, audio=768, visual=2048
```

Run:

```bash
python3 scripts/run_modality_stl.py --config configs/modality_stl.yaml --method mod_seq_ft
python3 scripts/run_modality_stl.py --config configs/modality_stl.yaml --method mod_seq_kd
python3 scripts/run_modality_stl.py --config configs/modality_stl.yaml --method prototype_replay
python3 scripts/run_modality_stl.py --config configs/modality_stl.yaml --method cmcrd_ours
```

Results are appended to:

```text
outputs/results/modality_stl_results.csv
```

`cmcrd_ours` uses prototype replay plus confidence-weighted cross-modal KL distillation.

## Feature Cache

A simple text hashing feature cache is available for pipeline testing:

```bash
python3 scripts/extract_features.py --config configs/main_stl.yaml --modalities text
```

It writes:

```text
outputs/features/{split}/text/{diaX_uttY}.npy
```

Audio and visual feature extractors are intentionally placeholders in this baseline.
