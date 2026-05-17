from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.continual.distillation import kd_loss
from src.data.datasets import build_speaker_vocab
from src.data.dialogue_dataset import (
    IGNORE_INDEX,
    DialogueExample,
    MeldDialogueFeatureDataset,
    build_dialogue_examples,
    collate_dialogue_batch,
    filter_dialogues_by_modalities,
)
from src.data.meld_csv import read_all_splits
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.models.dialogue_model import DialogueMultimodalSTLModel
from src.models.stl_model import TASK_NUM_LABELS
from src.train.metrics import compute_classification_metrics, decorate_final_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
DIALOGUE_TASK_METHODS = {"context_free", "hier_bilstm", "dlg_seq_ft", "dlg_seq_kd", "dlg_ours", "dlg_task_sa_cmd"}


def run_dialogue_task_experiment(config_path: str | Path, method: str, run_name: str | None = None) -> Path:
    if method not in DIALOGUE_TASK_METHODS:
        raise ValueError(f"Unknown dialogue Task-STL method '{method}'. Expected {sorted(DIALOGUE_TASK_METHODS)}")

    config = load_config(config_path)
    if run_name:
        config.setdefault("run", {})["name"] = run_name
        config.setdefault("run", {})["enabled"] = True
    output_dir = resolve_experiment_output_dir(config)
    setup_logging(output_dir / "logs" / f"{method}.log")
    seed_everything(int(config.get("seed", 13)))

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    continual_cfg = config.get("continual", {})
    modality_cfg = config.get("modalities", {})
    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    all_modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
    active_modalities = list(modality_cfg.get("active_modalities", all_modalities))
    feature_dims = {key: int(value) for key, value in modality_cfg.get("feature_dims", {}).items()}
    feature_root = str(resolve_path(modality_cfg.get("feature_root", train_cfg.get("output_dir", "outputs")), PROJECT_ROOT))
    eval_split = str(data_cfg.get("eval_split", "test"))

    split_records = read_all_splits(
        resolve_data_root(config),
        warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)),
    )
    dialogue_examples = {split: build_dialogue_examples(records) for split, records in split_records.items()}
    speaker_to_id = build_speaker_vocab(split_records["train"])
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    use_dialogue_encoder = method != "context_free"
    model = DialogueMultimodalSTLModel(
        feature_dims=feature_dims,
        num_speakers=len(speaker_to_id),
        config=config,
        task_num_labels={task: TASK_NUM_LABELS[task] for task in task_order},
        use_dialogue_encoder=use_dialogue_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    if method in {"context_free", "hier_bilstm"}:
        rows = _run_joint_dialogue(
            model,
            optimizer,
            dialogue_examples,
            task_order,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
            active_modalities,
            train_cfg,
            continual_cfg,
            eval_split,
            device,
            method,
        )
    else:
        rows = _run_sequence_dialogue(
            model,
            optimizer,
            dialogue_examples,
            task_order,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
            active_modalities,
            train_cfg,
            continual_cfg,
            eval_split,
            device,
            method,
            output_dir,
        )

    rows = decorate_final_metrics(rows, task_order)
    result_path = output_dir / "results" / "dialogue_task_stl_results.csv"
    _append_rows(result_path, rows)
    _save_checkpoint(model, output_dir, method, "final")
    LOGGER.info("Wrote dialogue Task-STL results to %s", result_path)
    return result_path


def _run_joint_dialogue(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dialogue_examples: dict[str, list[DialogueExample]],
    task_order: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    active_modalities: list[str],
    train_cfg: dict,
    continual_cfg: dict,
    eval_split: str,
    device: torch.device,
    method: str,
) -> list[dict[str, object]]:
    train_examples, missing = filter_dialogues_by_modalities(dialogue_examples["train"], feature_root, active_modalities)
    if missing:
        LOGGER.warning("Joint dialogue train skipped dialogues with missing features: %s", missing)
    loader = _build_loader(
        train_examples,
        feature_root,
        feature_dims,
        speaker_to_id,
        active_modalities,
        all_modalities,
        int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        sampler=str(continual_cfg.get("sampler", "")),
        sampler_task="emotion",
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for _ in range(int(train_cfg.get("epochs", 5))):
        model.train()
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for task_name in task_order:
                output = model(batch, task_name=task_name, active_modalities=active_modalities)
                losses.append(_sequence_ce(criterion, output["logits"], batch["labels"][task_name]))
            loss = torch.stack(losses).mean()
            loss.backward()
            _clip_and_step(model, optimizer, float(train_cfg.get("grad_clip", 5.0)))
    return _evaluate_dialogue_tasks(
        model,
        dialogue_examples[eval_split],
        task_order,
        feature_root,
        feature_dims,
        speaker_to_id,
        active_modalities,
        all_modalities,
        train_cfg,
        device,
        method,
        "joint",
    )


def _run_sequence_dialogue(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dialogue_examples: dict[str, list[DialogueExample]],
    task_order: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    active_modalities: list[str],
    train_cfg: dict,
    continual_cfg: dict,
    eval_split: str,
    device: torch.device,
    method: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    teacher: nn.Module | None = None
    learned_tasks: list[str] = []
    rows: list[dict[str, object]] = []
    train_examples, missing = filter_dialogues_by_modalities(dialogue_examples["train"], feature_root, active_modalities)
    if missing:
        LOGGER.warning("Sequential dialogue train skipped dialogues with missing features: %s", missing)

    for stage_index, task_name in enumerate(task_order, start=1):
        loader = _build_loader(
            train_examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            active_modalities,
            all_modalities,
            int(train_cfg.get("batch_size", 16)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            sampler=str(continual_cfg.get("sampler", "")),
            sampler_task=task_name,
        )
        for _ in range(int(train_cfg.get("epochs", 5))):
            model.train()
            for batch in loader:
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, task_name=task_name, active_modalities=active_modalities)
                loss = _sequence_ce(criterion, output["logits"], batch["labels"][task_name])
                if method in {"dlg_seq_kd", "dlg_ours", "dlg_task_sa_cmd"} and teacher is not None:
                    kd_terms = []
                    relation_terms = []
                    for old_task in learned_tasks:
                        with torch.no_grad():
                            teacher_output = teacher(batch, task_name=old_task, active_modalities=active_modalities)
                        student_output = model(batch, task_name=old_task, active_modalities=active_modalities)
                        valid_mask = batch["labels"][old_task] != IGNORE_INDEX
                        weights = (
                            confidence_weights(teacher_output["logits"], valid_mask)
                            if method == "dlg_task_sa_cmd"
                            else None
                        )
                        kd_terms.append(
                            masked_kd_loss(
                                student_output["logits"],
                                teacher_output["logits"],
                                mask=valid_mask,
                                temperature=float(continual_cfg.get("temperature", 2.0)),
                                weights=weights,
                            )
                        )
                        if method == "dlg_task_sa_cmd":
                            relation_terms.append(
                                sample_relation_loss(
                                    student_output["embedding"],
                                    teacher_output["embedding"],
                                    mask=valid_mask,
                                    weights=weights,
                                )
                            )
                    if kd_terms:
                        loss = loss + float(continual_cfg.get("lambda_kd", 1.0)) * torch.stack(kd_terms).mean()
                    if relation_terms:
                        loss = loss + float(continual_cfg.get("lambda_rel", continual_cfg.get("lambda_cmd", 1.0))) * torch.stack(relation_terms).mean()
                loss.backward()
                _clip_and_step(model, optimizer, float(train_cfg.get("grad_clip", 5.0)))

        learned_tasks.append(task_name)
        if method in {"dlg_seq_kd", "dlg_ours", "dlg_task_sa_cmd"}:
            teacher = _clone_frozen(model, device)
        _save_checkpoint(model, output_dir, method, f"stage{stage_index}_{task_name}")
        rows.extend(
            _evaluate_dialogue_tasks(
                model,
                dialogue_examples[eval_split],
                learned_tasks,
                feature_root,
                feature_dims,
                speaker_to_id,
                active_modalities,
                all_modalities,
                train_cfg,
                device,
                method,
                f"stage_{stage_index}_{task_name}",
            )
        )
    return rows


@torch.no_grad()
def _evaluate_dialogue_tasks(
    model: nn.Module,
    examples: list[DialogueExample],
    learned_tasks: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    active_modalities: list[str],
    all_modalities: list[str],
    train_cfg: dict,
    device: torch.device,
    method: str,
    stage: str,
) -> list[dict[str, object]]:
    available, missing = filter_dialogues_by_modalities(examples, feature_root, active_modalities)
    if missing:
        LOGGER.warning("Dialogue eval skipped dialogues with missing features: %s", missing)
    loader = _build_loader(
        available,
        feature_root,
        feature_dims,
        speaker_to_id,
        active_modalities,
        all_modalities,
        int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    rows: list[dict[str, object]] = []
    model.eval()
    for task_name in learned_tasks:
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch, task_name=task_name, active_modalities=active_modalities)
            labels = batch["labels"][task_name]
            mask = labels != IGNORE_INDEX
            y_true.extend(labels[mask].detach().cpu().tolist())
            y_pred.extend(output["logits"].argmax(dim=-1)[mask].detach().cpu().tolist())
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
                "num_eval_dialogues": len(available),
                "num_eval_utterances": len(y_true),
            }
        )
    return rows


def _build_loader(
    examples: list[DialogueExample],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    active_modalities: list[str],
    all_modalities: list[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: str = "",
    sampler_task: str = "emotion",
) -> DataLoader:
    dataset = MeldDialogueFeatureDataset(
        examples,
        feature_root=feature_root,
        feature_dims=feature_dims,
        speaker_to_id=speaker_to_id,
        active_modalities=active_modalities,
        all_modalities=all_modalities,
    )
    weighted_sampler = _build_weighted_sampler(examples, sampler_task) if shuffle and sampler == "weighted_random" else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if weighted_sampler is None else False,
        sampler=weighted_sampler,
        num_workers=num_workers,
        collate_fn=collate_dialogue_batch,
    )


def _build_weighted_sampler(examples: list[DialogueExample], task_name: str) -> WeightedRandomSampler | None:
    if not examples:
        return None
    labels = []
    for example in examples:
        seq = getattr(example, f"{task_name}_labels")
        valid = [label for label in seq if label != IGNORE_INDEX]
        labels.append(valid[0] if valid else 0)
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long)).float()
    counts[counts == 0] = 1.0
    weights = torch.tensor([1.0 / counts[label].item() for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _sequence_ce(criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _sequence_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    mask = labels.reshape(-1) != IGNORE_INDEX
    if not torch.any(mask):
        return student_logits.sum() * 0.0
    return kd_loss(
        student_logits.reshape(-1, student_logits.shape[-1])[mask],
        teacher_logits.reshape(-1, teacher_logits.shape[-1])[mask],
        temperature=temperature,
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
        "num_eval_dialogues",
        "num_eval_utterances",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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


def _clip_and_step(model: nn.Module, optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


def _save_checkpoint(model: nn.Module, output_dir: Path, method: str, suffix: str) -> None:
    torch.save(model.state_dict(), ensure_dir(output_dir / "checkpoints") / f"{method}_{suffix}.pt")


def _clone_frozen(model: nn.Module, device: torch.device) -> nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
