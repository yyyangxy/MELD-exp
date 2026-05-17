from __future__ import annotations

from pathlib import Path

import torch

from src.data.datasets import Vocabulary, build_speaker_vocab
from src.data.meld_csv import read_all_splits
from src.data.task_builder import build_all_tasks
from src.models.stl_model import MultiTaskSTLModel
from src.train.evaluator import evaluate
from src.train.metrics import decorate_final_metrics
from src.train.sequential_runner import _append_rows, _output_dir, _resolve_device
from src.train.trainer import build_loader, train_joint
from src.utils.logging import setup_logging
from src.utils.paths import ensure_dir, load_config, resolve_data_root
from src.utils.seed import seed_everything


def run_joint_experiment(config_path: str | Path, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    if run_name:
        config.setdefault("run", {})["name"] = run_name
        config.setdefault("run", {})["enabled"] = True
    output_dir = _output_dir(config)
    setup_logging(output_dir / "logs" / "joint.log")
    seed_everything(int(config.get("seed", 13)))

    data_root = resolve_data_root(config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    context_window = int(model_cfg.get("context_window", 3))
    eval_split = str(data_cfg.get("eval_split", "test"))

    split_records = read_all_splits(
        data_root,
        warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)),
    )
    task_examples = build_all_tasks(split_records, task_order, context_window=context_window)
    vocabulary = Vocabulary.build(
        example.context_text for example in task_examples["train"]["sentiment"]
    )
    speaker_to_id = build_speaker_vocab(split_records["train"])
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    model = MultiTaskSTLModel(
        vocab_size=len(vocabulary),
        num_speakers=len(speaker_to_id),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    max_length = int(model_cfg.get("max_length", 96))
    num_workers = int(train_cfg.get("num_workers", 0))
    use_context = bool(model_cfg.get("use_context", True))
    train_loaders = {
        task_name: build_loader(
            task_examples["train"][task_name],
            vocabulary,
            speaker_to_id,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True,
            num_workers=num_workers,
            use_context=use_context,
        )
        for task_name in task_order
    }
    train_joint(
        model,
        train_loaders=train_loaders,
        optimizer=optimizer,
        device=device,
        epochs=int(train_cfg.get("epochs", 5)),
        grad_clip=float(train_cfg.get("grad_clip", 5.0)),
    )

    rows = []
    for task_name in task_order:
        loader = build_loader(
            task_examples[eval_split][task_name],
            vocabulary,
            speaker_to_id,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
            num_workers=num_workers,
            use_context=use_context,
        )
        metrics = evaluate(model, loader, task_name, device)
        rows.append(
            {
                "method": "joint",
                "stage": "joint",
                "task": task_name,
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "positive_f1_for_shift": (
                    metrics["positive_f1_for_shift"]
                    if metrics["positive_f1_for_shift"] is not None
                    else ""
                ),
                "final_avg": "",
                "forgetting": "",
                "retention": "",
            }
        )

    rows = decorate_final_metrics(rows, task_order)
    result_path = output_dir / "results" / "main_stl_results.csv"
    _append_rows(result_path, rows)
    torch.save(model.state_dict(), ensure_dir(output_dir / "checkpoints") / "joint.pt")
    return result_path
