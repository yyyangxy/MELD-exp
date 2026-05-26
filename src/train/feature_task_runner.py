from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.continual.distillation import kd_loss
from src.continual.multimodal_memory import MultimodalPrototypeMemory
from src.continual.replay_buffer import RandomReplayBuffer
from src.continual.task_prototype_bank import TaskPrototypeBank
from src.data.datasets import build_speaker_vocab
from src.data.meld_csv import read_all_splits
from src.data.multimodal_dataset import (
    MeldMultimodalFeatureDataset,
    collate_multimodal_batch,
    filter_examples_by_modalities,
    summarize_feature_coverage,
)
from src.data.task_builder import TaskExample, build_all_tasks
from src.data.stl_task_splits import (
    filter_task_examples_by_stl_split,
    load_stl_task_split,
    log_stl_task_split_summary,
    resolve_stl_task_split_root,
)
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.losses.task_relation import prototype_alignment_loss, task_relation_distillation_loss
from src.models.multimodal_model import MultimodalSTLModel
from src.models.stl_model import TASK_NUM_LABELS
from src.train.metrics import compute_classification_metrics, decorate_final_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
FEATURE_TASK_METHODS = {
    "joint",
    "seq_ft",
    "lwf",
    "random_replay",
    "prototype_replay",
    "proto_replay_kd",
    "utt_task_sa_cmd",
    "utt_task_pg_trd",
}


def run_feature_task_experiment(config_path: str | Path, method: str, run_name: str | None = None) -> Path:
    if method not in FEATURE_TASK_METHODS:
        raise ValueError(f"Unknown feature Task-STL method '{method}'. Expected {sorted(FEATURE_TASK_METHODS)}")

    config = load_config(config_path)
    if run_name:
        config.setdefault("run", {})["name"] = run_name
        config.setdefault("run", {})["enabled"] = True
    output_dir = resolve_experiment_output_dir(config)
    setup_logging(output_dir / "logs" / f"{method}.log")
    seed_everything(int(config.get("seed", 13)))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    continual_cfg = config.get("continual", {})
    modality_cfg = config.get("modalities", {})

    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    all_modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
    active_modalities = list(modality_cfg.get("active_modalities", all_modalities))
    feature_dims = {key: int(value) for key, value in modality_cfg.get("feature_dims", {}).items()}
    feature_root = str(resolve_path(modality_cfg.get("feature_root", train_cfg.get("output_dir", "outputs")), PROJECT_ROOT))
    eval_split = str(data_cfg.get("eval_split", "test"))
    context_window = int(model_cfg.get("context_window", 3))

    data_root = resolve_data_root(config)
    split_records = read_all_splits(
        data_root,
        warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)),
    )
    task_examples = build_all_tasks(split_records, task_order, context_window=context_window)
    task_split_root = resolve_stl_task_split_root(data_cfg, data_root)
    if task_split_root is not None:
        task_split = load_stl_task_split(task_split_root, task_order, split_records.keys())
        task_examples = filter_task_examples_by_stl_split(task_examples, task_split)
        log_stl_task_split_summary(task_examples=task_examples)
        LOGGER.info("Using fixed STL task split root: %s", task_split.root)
    else:
        LOGGER.warning("No data.stl_task_split_root configured; feature Task-STL uses full split data for every task.")
    speaker_to_id = build_speaker_vocab(split_records["train"])
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    for split, examples_by_task in task_examples.items():
        examples = examples_by_task[task_order[0]]
        LOGGER.info(
            "%s feature coverage under %s: %s",
            split,
            feature_root,
            summarize_feature_coverage(examples, feature_root, all_modalities),
        )

    model = MultimodalSTLModel(
        feature_dims=feature_dims,
        num_speakers=len(speaker_to_id),
        config=config,
        task_num_labels={task: TASK_NUM_LABELS[task] for task in task_order},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    sampler = str(continual_cfg.get("sampler", train_cfg.get("sampler", "")) or "")

    rows = (
        _run_joint(
            model,
            optimizer,
            task_examples,
            task_order,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
            active_modalities,
            batch_size,
            num_workers,
            sampler,
            train_cfg,
            eval_split,
            device,
            method,
        )
        if method == "joint"
        else _run_sequence(
            model,
            optimizer,
            task_examples,
            task_order,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
            active_modalities,
            batch_size,
            num_workers,
            sampler,
            train_cfg,
            continual_cfg,
            eval_split,
            device,
            method,
            output_dir,
        )
    )

    rows = decorate_final_metrics(rows, task_order)
    result_path = output_dir / "results" / "main_stl_results.csv"
    _append_rows(result_path, rows)
    _save_checkpoint(model, output_dir, method, "final")
    LOGGER.info("Wrote feature Task-STL results to %s", result_path)
    return result_path


def _run_sequence(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task_examples: dict[str, dict[str, list[TaskExample]]],
    task_order: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    active_modalities: list[str],
    batch_size: int,
    num_workers: int,
    sampler: str,
    train_cfg: dict,
    continual_cfg: dict,
    eval_split: str,
    device: torch.device,
    method: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    criterion = nn.CrossEntropyLoss()
    teacher: nn.Module | None = None
    learned_tasks: list[str] = []
    rows: list[dict[str, object]] = []
    memory = _build_memory(method, continual_cfg, batch_size, device)
    prototype_bank = TaskPrototypeBank() if method == "utt_task_pg_trd" else None

    for stage_index, task_name in enumerate(task_order, start=1):
        train_examples, missing = filter_examples_by_modalities(
            task_examples["train"][task_name],
            feature_root,
            active_modalities,
        )
        if missing:
            LOGGER.warning("Task %s skipped missing train features: %s", task_name, missing)
        loader = _build_loader(
            train_examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            active_modalities,
            all_modalities,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            sampler=sampler,
        )
        replay_loaders = _build_replay_loaders(
            memory,
            learned_tasks,
            feature_root,
            feature_dims,
            speaker_to_id,
            active_modalities,
            all_modalities,
            batch_size,
            num_workers,
            sampler,
        )
        loss = _train_one_feature_task(
            model,
            loader,
            task_name,
            optimizer,
            device,
            criterion,
            method,
            learned_tasks,
            teacher,
            replay_loaders,
            prototype_bank,
            active_modalities,
            epochs=int(train_cfg.get("epochs", 5)),
            grad_clip=float(train_cfg.get("grad_clip", 5.0)),
            lambda_kd=float(continual_cfg.get("lambda_kd", 1.0)),
            lambda_rel=float(continual_cfg.get("lambda_rel", continual_cfg.get("lambda_kd", 1.0))),
            temperature=float(continual_cfg.get("temperature", 2.0)),
            eval_interval=int(train_cfg.get("eval_interval", 0)),
            eval_callback=lambda epoch, stage_index=stage_index, task_name=task_name: rows.extend(
                _evaluate_stage(
                    model,
                    task_examples[eval_split],
                    [*learned_tasks, task_name],
                    feature_root,
                    feature_dims,
                    speaker_to_id,
                    active_modalities,
                    all_modalities,
                    batch_size,
                    num_workers,
                    device,
                    method,
                    f"stage_{stage_index}_{task_name}_ep{epoch}",
                )
            ),
        )
        LOGGER.info("Stage %s task=%s loss=%.4f", stage_index, task_name, loss)

        learned_tasks.append(task_name)
        _update_memory(
            memory,
            method,
            task_name,
            train_examples,
            active_modalities,
            model,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
        )
        _update_prototype_bank(
            prototype_bank,
            method,
            task_name,
            train_examples,
            active_modalities,
            model,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
            batch_size,
            num_workers,
            device,
        )
        if method in {"lwf", "proto_replay_kd", "utt_task_sa_cmd", "utt_task_pg_trd"}:
            teacher = _clone_frozen(model, device)

        _save_checkpoint(model, output_dir, method, f"stage{stage_index}_{task_name}")
        rows.extend(
            _evaluate_stage(
                model,
                task_examples[eval_split],
                learned_tasks,
                feature_root,
                feature_dims,
                speaker_to_id,
                active_modalities,
                all_modalities,
                batch_size,
                num_workers,
                device,
                method,
                f"stage_{stage_index}_{task_name}",
            )
        )
    return rows


def _run_joint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task_examples: dict[str, dict[str, list[TaskExample]]],
    task_order: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    active_modalities: list[str],
    batch_size: int,
    num_workers: int,
    sampler: str,
    train_cfg: dict,
    eval_split: str,
    device: torch.device,
    method: str,
) -> list[dict[str, object]]:
    criterion = nn.CrossEntropyLoss()
    train_loaders = {}
    for task_name in task_order:
        examples, _ = filter_examples_by_modalities(task_examples["train"][task_name], feature_root, active_modalities)
        train_loaders[task_name] = _build_loader(
            examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            active_modalities,
            all_modalities,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            sampler=sampler,
        )

    for _ in range(int(train_cfg.get("epochs", 5))):
        iterators = {task: iter(loader) for task, loader in train_loaders.items()}
        active = set(task_order)
        model.train()
        while active:
            for task_name in list(task_order):
                if task_name not in active:
                    continue
                try:
                    batch = next(iterators[task_name])
                except StopIteration:
                    active.remove(task_name)
                    continue
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, task_name=task_name, active_modalities=active_modalities)
                loss = criterion(output["logits"], batch["label"])
                loss.backward()
                grad_clip = float(train_cfg.get("grad_clip", 5.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

    return _evaluate_stage(
        model,
        task_examples[eval_split],
        task_order,
        feature_root,
        feature_dims,
        speaker_to_id,
        active_modalities,
        all_modalities,
        batch_size,
        num_workers,
        device,
        method,
        "joint",
    )


def _train_one_feature_task(
    model: nn.Module,
    loader: DataLoader,
    task_name: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    method: str,
    old_task_names: list[str],
    teacher: nn.Module | None,
    replay_loaders: dict[str, DataLoader],
    prototype_bank: TaskPrototypeBank | None,
    active_modalities: list[str],
    epochs: int,
    grad_clip: float,
    lambda_kd: float,
    lambda_rel: float,
    temperature: float,
    eval_interval: int = 0,
    eval_callback=None,
) -> float:
    replay_iters = {task: _infinite(loader) for task, loader in replay_loaders.items()}
    total_loss = 0.0
    steps = 0
    for epoch_index in range(1, epochs + 1):
        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_replay_kd_loss = 0.0
        epoch_relation_loss = 0.0
        epoch_steps = 0
        model.train()
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task_name=task_name, active_modalities=active_modalities)
            supervised_terms = [criterion(output["logits"], batch["label"])]
            kd_terms = []

            relation_terms = []
            student_logits_by_task = {}
            teacher_logits_by_task = {}
            if method in {"lwf", "utt_task_sa_cmd", "utt_task_pg_trd"} and teacher is not None:
                for old_task in old_task_names:
                    with torch.no_grad():
                        teacher_output = teacher(batch, task_name=old_task, active_modalities=active_modalities)
                    student_output = model(batch, task_name=old_task, active_modalities=active_modalities)
                    student_logits_by_task[old_task] = student_output["logits"]
                    teacher_logits_by_task[old_task] = teacher_output["logits"]
                    weights = confidence_weights(teacher_output["logits"]) if method in {"utt_task_sa_cmd", "utt_task_pg_trd"} else None
                    kd_terms.append(
                        masked_kd_loss(
                            student_output["logits"],
                            teacher_output["logits"],
                            temperature=temperature,
                            weights=weights,
                        )
                    )
                    if method in {"utt_task_sa_cmd", "utt_task_pg_trd"}:
                        relation_terms.append(
                            sample_relation_loss(
                                student_output["embedding"],
                                teacher_output["embedding"],
                                weights=weights,
                            )
                        )
                    if method == "utt_task_pg_trd" and prototype_bank is not None:
                        relation_terms.append(
                            prototype_alignment_loss(
                                student_output["embedding"],
                                teacher_output["embedding"],
                                prototype_bank.prototypes_for(old_task, device),
                            )
                        )

            if method == "utt_task_pg_trd" and student_logits_by_task:
                relation_terms.append(
                    task_relation_distillation_loss(
                        student_logits_by_task,
                        teacher_logits_by_task,
                        temperature=temperature,
                    )
                )

            replay_kd_terms = []
            for replay_task, iterator in replay_iters.items():
                replay_batch = _move_batch(next(iterator), device)
                replay_output = model(replay_batch, task_name=replay_task, active_modalities=active_modalities)
                supervised_terms.append(criterion(replay_output["logits"], replay_batch["label"]))
                if method in {"proto_replay_kd", "utt_task_sa_cmd", "utt_task_pg_trd"} and teacher is not None:
                    with torch.no_grad():
                        teacher_output = teacher(replay_batch, task_name=replay_task, active_modalities=active_modalities)
                    weights = confidence_weights(teacher_output["logits"]) if method in {"utt_task_sa_cmd", "utt_task_pg_trd"} else None
                    replay_kd_terms.append(
                        masked_kd_loss(
                            replay_output["logits"],
                            teacher_output["logits"],
                            temperature=temperature,
                            weights=weights,
                        )
                    )
                    if method == "utt_task_pg_trd" and prototype_bank is not None:
                        relation_terms.append(
                            prototype_alignment_loss(
                                replay_output["embedding"],
                                teacher_output["embedding"],
                                prototype_bank.prototypes_for(replay_task, device),
                            )
                        )

            ce_loss = torch.stack(supervised_terms).mean()
            kd_loss_value = torch.stack(kd_terms).mean() if kd_terms else ce_loss.new_zeros(())
            replay_kd_loss_value = torch.stack(replay_kd_terms).mean() if replay_kd_terms else ce_loss.new_zeros(())
            relation_loss_value = torch.stack(relation_terms).mean() if relation_terms else ce_loss.new_zeros(())

            loss = ce_loss
            if kd_terms:
                loss = loss + lambda_kd * kd_loss_value
            if replay_kd_terms:
                loss = loss + lambda_kd * replay_kd_loss_value
            if relation_terms:
                loss = loss + lambda_rel * relation_loss_value
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_float = float(loss.detach().cpu())
            total_loss += loss_float
            epoch_total_loss += loss_float
            epoch_ce_loss += float(ce_loss.detach().cpu())
            epoch_kd_loss += float(kd_loss_value.detach().cpu())
            epoch_replay_kd_loss += float(replay_kd_loss_value.detach().cpu())
            epoch_relation_loss += float(relation_loss_value.detach().cpu())
            epoch_steps += 1
            steps += 1
        LOGGER.info(
            "Epoch %d/%d task=%s method=%s total_loss=%.4f ce=%.4f kd=%.4f replay_kd=%.4f relation=%.4f",
            epoch_index,
            epochs,
            task_name,
            method,
            epoch_total_loss / max(epoch_steps, 1),
            epoch_ce_loss / max(epoch_steps, 1),
            epoch_kd_loss / max(epoch_steps, 1),
            epoch_replay_kd_loss / max(epoch_steps, 1),
            epoch_relation_loss / max(epoch_steps, 1),
        )
        if eval_interval > 0 and eval_callback is not None and epoch_index % eval_interval == 0:
            eval_callback(epoch_index)
    return total_loss / max(steps, 1)


def _evaluate_stage(
    model: nn.Module,
    eval_examples_by_task: dict[str, list[TaskExample]],
    learned_tasks: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    active_modalities: list[str],
    all_modalities: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    method: str,
    stage: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model.eval()
    for task_name in learned_tasks:
        examples, missing = filter_examples_by_modalities(eval_examples_by_task[task_name], feature_root, active_modalities)
        if missing:
            LOGGER.warning("Eval task %s skipped missing features: %s", task_name, missing)
        loader = _build_loader(
            examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            active_modalities,
            all_modalities,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                output = model(batch, task_name=task_name, active_modalities=active_modalities)
                y_true.extend(batch["label"].detach().cpu().tolist())
                y_pred.extend(output["logits"].argmax(dim=-1).detach().cpu().tolist())
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            num_labels=TASK_NUM_LABELS[task_name],
            positive_label=1 if task_name == "shift" else None,
        )
        rows.append(
            {
                "method": method,
                "stage": stage,
                "task": task_name,
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "positive_f1_for_shift": metrics["positive_f1_for_shift"] or "",
                "final_avg": "",
                "forgetting": "",
                "retention": "",
            }
        )
    return rows


def _build_memory(method: str, continual_cfg: dict, batch_size: int, device: torch.device):
    memory_per_class = int(continual_cfg.get("memory_per_class", 100))
    if method == "random_replay":
        return RandomReplayBuffer(memory_per_class=memory_per_class, seed=int(continual_cfg.get("seed", 13)))
    if method in {"prototype_replay", "proto_replay_kd", "utt_task_sa_cmd", "utt_task_pg_trd"}:
        return MultimodalPrototypeMemory(
            memory_per_class=memory_per_class,
            batch_size=batch_size,
            device=device,
            memory_strategy=str(continual_cfg.get("memory_strategy", "prototype_nearest")),
            representative_ratio=float(continual_cfg.get("representative_ratio", 0.5)),
            kmeans_iters=int(continual_cfg.get("kmeans_iters", 10)),
            seed=int(continual_cfg.get("seed", 13)),
        )
    return None


def _update_memory(
    memory,
    method: str,
    task_name: str,
    examples: list[TaskExample],
    active_modalities: list[str],
    model: nn.Module,
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
) -> None:
    if memory is None:
        return
    if method == "random_replay":
        memory.update(task_name, examples)
    elif method in {"prototype_replay", "proto_replay_kd", "utt_task_sa_cmd", "utt_task_pg_trd"}:
        memory.update(
            task_name,
            examples,
            active_modalities=active_modalities,
            model=model,
            feature_root=feature_root,
            feature_dims=feature_dims,
            speaker_to_id=speaker_to_id,
            all_modalities=all_modalities,
        )


@torch.no_grad()
def _update_prototype_bank(
    prototype_bank: TaskPrototypeBank | None,
    method: str,
    task_name: str,
    examples: list[TaskExample],
    active_modalities: list[str],
    model: nn.Module,
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    if prototype_bank is None or method != "utt_task_pg_trd" or not examples:
        return

    loader = _build_loader(
        examples,
        feature_root,
        feature_dims,
        speaker_to_id,
        active_modalities,
        all_modalities,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    was_training = model.training
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch, task_name=task_name, active_modalities=active_modalities)
        embeddings.append(output["embedding"].detach().cpu())
        labels.append(batch["label"].detach().cpu())
    if was_training:
        model.train()
    if embeddings:
        prototype_bank.update_from_embeddings(task_name, torch.cat(embeddings, dim=0), torch.cat(labels, dim=0))


def _build_replay_loaders(
    memory,
    learned_tasks: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    active_modalities: list[str],
    all_modalities: list[str],
    batch_size: int,
    num_workers: int,
    sampler: str,
) -> dict[str, DataLoader]:
    if memory is None:
        return {}
    loaders = {}
    for task_name in learned_tasks:
        examples = memory.examples_for(task_name)
        if examples:
            loaders[task_name] = _build_loader(
                examples,
                feature_root,
                feature_dims,
                speaker_to_id,
                active_modalities,
                all_modalities,
                batch_size,
                shuffle=True,
                num_workers=num_workers,
                sampler=sampler,
            )
    return loaders


def _build_loader(
    examples: list[TaskExample],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    active_modalities: list[str],
    all_modalities: list[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: str = "",
) -> DataLoader:
    dataset = MeldMultimodalFeatureDataset(
        examples,
        feature_root=feature_root,
        feature_dims=feature_dims,
        speaker_to_id=speaker_to_id,
        active_modalities=active_modalities,
        all_modalities=all_modalities,
        allow_missing=False,
    )
    weighted_sampler = _build_weighted_sampler(examples) if shuffle and sampler == "weighted_random" else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if weighted_sampler is None else False,
        sampler=weighted_sampler,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
    )


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


def _save_checkpoint(model: nn.Module, output_dir: Path, method: str, suffix: str) -> None:
    torch.save(model.state_dict(), ensure_dir(output_dir / "checkpoints") / f"{method}_{suffix}.pt")


def _clone_frozen(model: nn.Module, device: torch.device) -> nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                nested_key: nested_value.to(device) if hasattr(nested_value, "to") else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _build_weighted_sampler(examples: list[TaskExample]) -> WeightedRandomSampler | None:
    if not examples:
        return None
    labels = [example.label for example in examples]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long)).float()
    counts[counts == 0] = 1.0
    weights = torch.tensor([1.0 / counts[label].item() for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
