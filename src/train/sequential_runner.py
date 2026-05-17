from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch

from src.continual.distillation import clone_frozen_teacher
from src.continual.prototype_memory import PrototypeMemory
from src.continual.replay_buffer import RandomReplayBuffer
from src.data.datasets import Vocabulary, build_speaker_vocab
from src.data.meld_csv import read_all_splits
from src.data.task_builder import TaskExample, build_all_tasks
from src.models.stl_model import MultiTaskSTLModel
from src.train.evaluator import evaluate
from src.train.metrics import decorate_final_metrics
from src.train.trainer import build_loader, train_one_task
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
SEQUENTIAL_METHODS = {"seq_ft", "lwf", "random_replay", "prototype_replay", "proto_replay_kd"}


def run_sequential_experiment(config_path: str | Path, method: str, run_name: str | None = None) -> Path:
    if method not in SEQUENTIAL_METHODS:
        raise ValueError(f"Unknown sequential method '{method}'. Expected {sorted(SEQUENTIAL_METHODS)}")

    config = load_config(config_path)
    if run_name:
        config.setdefault("run", {})["name"] = run_name
        config.setdefault("run", {})["enabled"] = True
    output_dir = _output_dir(config)
    setup_logging(output_dir / "logs" / f"{method}.log")
    seed_everything(int(config.get("seed", 13)))

    data_root = resolve_data_root(config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    continual_cfg = config.get("continual", {})
    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    context_window = int(model_cfg.get("context_window", 3))
    eval_split = str(data_cfg.get("eval_split", "test"))

    LOGGER.info("Loading MELD data from %s", data_root)
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
    sampler = str(continual_cfg.get("sampler", train_cfg.get("sampler", "")) or "")

    teacher = None
    learned_tasks: list[str] = []
    rows: list[dict[str, object]] = []
    memory = _build_memory(method, continual_cfg, batch_size, device)

    for stage_index, task_name in enumerate(task_order, start=1):
        train_loader = build_loader(
            task_examples["train"][task_name],
            vocabulary,
            speaker_to_id,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True,
            num_workers=num_workers,
            use_context=use_context,
            sampler=sampler,
        )
        replay_loaders = _build_replay_loaders(
            memory,
            learned_tasks,
            vocabulary,
            speaker_to_id,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            use_context=use_context,
            sampler=sampler,
        )

        stats = train_one_task(
            model=model,
            train_loader=train_loader,
            task_name=task_name,
            optimizer=optimizer,
            device=device,
            epochs=int(train_cfg.get("epochs", 5)),
            grad_clip=float(train_cfg.get("grad_clip", 5.0)),
            method=method,
            old_task_names=list(learned_tasks),
            teacher=teacher,
            replay_loaders=replay_loaders,
            lambda_kd=float(continual_cfg.get("lambda_kd", 0.5)),
            temperature=float(continual_cfg.get("temperature", 2.0)),
        )
        LOGGER.info("Stage %s task=%s loss=%.4f steps=%s", stage_index, task_name, stats.loss, stats.steps)

        learned_tasks.append(task_name)
        _update_memory(
            memory,
            method,
            task_name,
            task_examples["train"][task_name],
            model,
            vocabulary,
            speaker_to_id,
            max_length=max_length,
            use_context=use_context,
        )
        if method in {"lwf", "proto_replay_kd"}:
            teacher = clone_frozen_teacher(model, device)

        _save_checkpoint(model, output_dir, method, stage_index, task_name)
        rows.extend(
            _evaluate_stage(
                model,
                task_examples[eval_split],
                learned_tasks,
                vocabulary,
                speaker_to_id,
                batch_size,
                max_length,
                num_workers,
                use_context,
                device,
                method,
                stage=f"stage_{stage_index}_{task_name}",
            )
        )

    rows = decorate_final_metrics(rows, task_order)
    result_path = output_dir / "results" / "main_stl_results.csv"
    _append_rows(result_path, rows)
    LOGGER.info("Wrote results to %s", result_path)
    return result_path


def _build_memory(method: str, continual_cfg: dict, batch_size: int, device: torch.device):
    memory_per_class = int(continual_cfg.get("memory_per_class", 20))
    if method == "random_replay":
        return RandomReplayBuffer(memory_per_class=memory_per_class, seed=int(continual_cfg.get("seed", 13)))
    if method in {"prototype_replay", "proto_replay_kd"}:
        return PrototypeMemory(memory_per_class=memory_per_class, batch_size=batch_size, device=device)
    return None


def _build_replay_loaders(
    memory,
    learned_tasks: list[str],
    vocabulary: Vocabulary,
    speaker_to_id: dict[str, int],
    batch_size: int,
    max_length: int,
    num_workers: int,
    use_context: bool,
    sampler: str = "",
) -> dict[str, torch.utils.data.DataLoader]:
    if memory is None:
        return {}
    loaders = {}
    for task_name in learned_tasks:
        examples = memory.examples_for(task_name)
        if examples:
            loaders[task_name] = build_loader(
                examples,
                vocabulary,
                speaker_to_id,
                batch_size=batch_size,
                max_length=max_length,
                shuffle=True,
                num_workers=num_workers,
                use_context=use_context,
                sampler=sampler,
            )
    return loaders


def _update_memory(
    memory,
    method: str,
    task_name: str,
    examples: list[TaskExample],
    model,
    vocabulary: Vocabulary,
    speaker_to_id: dict[str, int],
    max_length: int,
    use_context: bool,
) -> None:
    if memory is None:
        return
    if method == "random_replay":
        memory.update(task_name, examples)
    elif method in {"prototype_replay", "proto_replay_kd"}:
        memory.update(
            task_name,
            examples,
            model=model,
            vocabulary=vocabulary,
            speaker_to_id=speaker_to_id,
            max_length=max_length,
            use_context=use_context,
        )


def _evaluate_stage(
    model,
    eval_examples_by_task: dict[str, list[TaskExample]],
    learned_tasks: list[str],
    vocabulary: Vocabulary,
    speaker_to_id: dict[str, int],
    batch_size: int,
    max_length: int,
    num_workers: int,
    use_context: bool,
    device: torch.device,
    method: str,
    stage: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_name in learned_tasks:
        loader = build_loader(
            eval_examples_by_task[task_name],
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
                "method": method,
                "stage": stage,
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
    return rows


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "method",
        "stage",
        "task",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "positive_f1_for_shift",
        "final_avg",
        "forgetting",
        "retention",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(model, output_dir: Path, method: str, stage_index: int, task_name: str) -> None:
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    torch.save(
        model.state_dict(),
        checkpoint_dir / f"{method}_stage{stage_index}_{task_name}.pt",
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _output_dir(config: dict) -> Path:
    return resolve_experiment_output_dir(config)
