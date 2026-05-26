from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.data.datasets import build_speaker_vocab
from src.data.meld_csv import read_all_splits
from src.data.multimodal_dataset import (
    collate_multimodal_batch,
    filter_examples_by_modalities,
    MeldMultimodalFeatureDataset,
)
from src.data.task_builder import build_all_tasks
from src.models.multimodal_model import MultimodalSTLModel
from src.models.stl_model import TASK_NUM_LABELS
from src.train.metrics import compute_classification_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MELD upper-bound diagnostics on frozen v2 features.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl_v2.yaml"))
    parser.add_argument("--mode", choices=["emotion_only", "task_joint"], required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_diagnostic(args.config, args.mode, args.run_name)


def run_diagnostic(config_path: str | Path, mode: str, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    if run_name:
        config.setdefault("run", {})["name"] = run_name
        config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "upper_bound_diagnostics"
    output_dir = resolve_experiment_output_dir(config)
    setup_logging(output_dir / "logs" / f"{mode}.log")
    seed_everything(int(config.get("seed", 13)))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    continual_cfg = config.get("continual", {})
    modality_cfg = config.get("modalities", {})

    task_order = ["emotion"] if mode == "emotion_only" else list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    all_modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
    active_modalities = list(modality_cfg.get("active_modalities", all_modalities))
    feature_dims = {key: int(value) for key, value in modality_cfg.get("feature_dims", {}).items()}
    feature_root = str(resolve_path(modality_cfg.get("feature_root", train_cfg.get("output_dir", "outputs")), PROJECT_ROOT))
    epochs = int(train_cfg.get("epochs", 100))
    eval_interval = int(train_cfg.get("eval_interval", 10))
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    sampler = str(continual_cfg.get("sampler", "") or "")
    device = torch.device("cuda" if str(train_cfg.get("device", "auto")) == "auto" and torch.cuda.is_available() else train_cfg.get("device", "cpu"))

    split_records = read_all_splits(resolve_data_root(config), warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)))
    task_examples = build_all_tasks(split_records, task_order, context_window=int(model_cfg.get("context_window", 3)))
    speaker_to_id = build_speaker_vocab(split_records["train"])
    model = MultimodalSTLModel(
        feature_dims=feature_dims,
        num_speakers=len(speaker_to_id),
        config=config,
        task_num_labels={task: TASK_NUM_LABELS[task] for task in task_order},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    criterion = nn.CrossEntropyLoss()

    train_loaders = {
        task: _loader(task_examples["train"][task], feature_root, feature_dims, speaker_to_id, active_modalities, all_modalities, batch_size, True, num_workers, sampler)
        for task in task_order
    }
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        iterators = {task: iter(loader) for task, loader in train_loaders.items()}
        active = set(task_order)
        while active:
            for task in list(task_order):
                if task not in active:
                    continue
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    active.remove(task)
                    continue
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, task_name=task, active_modalities=active_modalities)
                loss = criterion(output["logits"], batch["label"])
                loss.backward()
                grad_clip = float(train_cfg.get("grad_clip", 5.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                steps += 1
        print(f"{mode} epoch={epoch}/{epochs} loss={epoch_loss / max(steps, 1):.4f}")
        if eval_interval > 0 and epoch % eval_interval == 0:
            for split in ["dev", "test"]:
                rows.extend(_evaluate(model, task_examples[split], task_order, feature_root, feature_dims, speaker_to_id, active_modalities, all_modalities, batch_size, num_workers, device, mode, split, epoch))

    result_path = output_dir / "results" / f"{mode}_diagnostics.csv"
    _write_rows(result_path, rows)
    print(f"results={result_path}")
    return result_path


def _loader(examples, feature_root, feature_dims, speaker_to_id, active_modalities, all_modalities, batch_size, shuffle, num_workers, sampler):
    examples, _ = filter_examples_by_modalities(examples, feature_root, active_modalities)
    dataset = MeldMultimodalFeatureDataset(examples, feature_root, feature_dims, speaker_to_id, active_modalities, all_modalities, allow_missing=False)
    weighted_sampler = None
    if shuffle and sampler == "weighted_random":
        labels = [example.label for example in examples]
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long)).float()
        counts[counts == 0] = 1.0
        weights = torch.tensor([1.0 / counts[label].item() for label in labels], dtype=torch.double)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if weighted_sampler is None else False, sampler=weighted_sampler, num_workers=num_workers, collate_fn=collate_multimodal_batch)


@torch.no_grad()
def _evaluate(model, examples_by_task, task_order, feature_root, feature_dims, speaker_to_id, active_modalities, all_modalities, batch_size, num_workers, device, mode, split, epoch):
    rows = []
    model.eval()
    for task in task_order:
        loader = _loader(examples_by_task[task], feature_root, feature_dims, speaker_to_id, active_modalities, all_modalities, batch_size, False, num_workers, "")
        y_true, y_pred = [], []
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch, task_name=task, active_modalities=active_modalities)
            y_true.extend(batch["label"].detach().cpu().tolist())
            y_pred.extend(output["logits"].argmax(dim=-1).detach().cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred, TASK_NUM_LABELS[task], positive_label=1 if task == "shift" else None)
        rows.append({"mode": mode, "split": split, "epoch": epoch, "task": task, **metrics})
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["mode", "split", "epoch", "task", "accuracy", "weighted_f1", "macro_f1", "positive_f1_for_shift"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: {k: v.to(device) if hasattr(v, "to") else v for k, v in value.items()} if isinstance(value, dict) else value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


if __name__ == "__main__":
    main()
