from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.continual.cross_modal_distillation import cross_modal_kd_loss
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
from src.train.metrics import compute_classification_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
DIALOGUE_MODALITY_METHODS = {"dlg_mod_seq_ft", "dlg_mod_seq_kd", "dlg_ours", "dlg_modality_sa_cmd"}


def run_dialogue_modality_experiment(config_path: str | Path, method: str, run_name: str | None = None) -> Path:
    if method not in DIALOGUE_MODALITY_METHODS:
        raise ValueError(f"Unknown dialogue modality method '{method}'. Expected {sorted(DIALOGUE_MODALITY_METHODS)}")

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
    all_modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
    stage_order = list(modality_cfg.get("stage_order", ["text", "text_audio", "full"]))
    stages = _parse_stages(modality_cfg, stage_order)
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

    model = DialogueMultimodalSTLModel(
        feature_dims=feature_dims,
        num_speakers=len(speaker_to_id),
        config=config,
        task_num_labels={"emotion": 7},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    teacher_by_stage: dict[str, nn.Module] = {}
    learned_stages: list[str] = []
    rows: list[dict[str, object]] = []

    for stage_name in stage_order:
        active_modalities = stages[stage_name]
        train_examples, missing = filter_dialogues_by_modalities(dialogue_examples["train"], feature_root, active_modalities)
        if missing:
            LOGGER.warning("Dialogue modality stage %s skipped missing train dialogues: %s", stage_name, missing)
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
        )
        loss = _train_stage(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            method=method,
            stage_name=stage_name,
            active_modalities=active_modalities,
            learned_stages=learned_stages,
            stages=stages,
            teacher_by_stage=teacher_by_stage,
            epochs=int(train_cfg.get("epochs", 5)),
            grad_clip=float(train_cfg.get("grad_clip", 5.0)),
            lambda_kd=float(continual_cfg.get("lambda_kd", 1.0)),
            lambda_cmd=float(continual_cfg.get("lambda_cmd", 1.0)),
            lambda_rel=float(continual_cfg.get("lambda_rel", continual_cfg.get("lambda_cmd", 1.0))),
            temperature=float(continual_cfg.get("temperature", 2.0)),
        )
        LOGGER.info("Finished dialogue modality stage=%s loss=%.4f", stage_name, loss)
        learned_stages.append(stage_name)
        if method in {"dlg_mod_seq_kd", "dlg_ours", "dlg_modality_sa_cmd"}:
            teacher_by_stage[stage_name] = _clone_frozen(model, device)
        _save_checkpoint(model, output_dir, method, stage_name)
        rows.extend(
            _evaluate_stage(
                model,
                dialogue_examples[eval_split],
                learned_stages,
                stages,
                feature_root,
                feature_dims,
                speaker_to_id,
                all_modalities,
                train_cfg,
                device,
                method,
                current_stage=stage_name,
            )
        )

    rows = _decorate_modality_metrics(rows, stage_order)
    result_path = output_dir / "results" / "dialogue_modality_stl_results.csv"
    _append_rows(result_path, rows)
    LOGGER.info("Wrote dialogue modality results to %s", result_path)
    return result_path


def _train_stage(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    stage_name: str,
    active_modalities: list[str],
    learned_stages: list[str],
    stages: dict[str, list[str]],
    teacher_by_stage: dict[str, nn.Module],
    epochs: int,
    grad_clip: float,
    lambda_kd: float,
    lambda_cmd: float,
    lambda_rel: float,
    temperature: float,
) -> float:
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    steps = 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task_name="emotion", active_modalities=active_modalities)
            supervised_terms = [_sequence_ce(criterion, output["logits"], batch["labels"]["emotion"])]
            if method == "dlg_modality_sa_cmd":
                for old_stage in learned_stages:
                    supervised_output = model(batch, task_name="emotion", active_modalities=stages[old_stage])
                    supervised_terms.append(_sequence_ce(criterion, supervised_output["logits"], batch["labels"]["emotion"]))
            loss = torch.stack(supervised_terms).mean()

            kd_terms = []
            relation_terms = []
            if method in {"dlg_mod_seq_kd", "dlg_ours", "dlg_modality_sa_cmd"}:
                for old_stage in learned_stages:
                    teacher = teacher_by_stage.get(old_stage)
                    if teacher is None:
                        continue
                    old_modalities = stages[old_stage]
                    with torch.no_grad():
                        teacher_output = teacher(batch, task_name="emotion", active_modalities=old_modalities)
                    student_output = model(batch, task_name="emotion", active_modalities=old_modalities)
                    valid_mask = batch["labels"]["emotion"] != IGNORE_INDEX
                    weights = (
                        confidence_weights(teacher_output["logits"], valid_mask)
                        if method == "dlg_modality_sa_cmd"
                        else None
                    )
                    kd_terms.append(
                        masked_kd_loss(
                            student_output["logits"],
                            teacher_output["logits"],
                            mask=valid_mask,
                            temperature=temperature,
                            weights=weights,
                        )
                    )
                    if method == "dlg_modality_sa_cmd":
                        relation_terms.append(
                            sample_relation_loss(
                                student_output["embedding"],
                                teacher_output["embedding"],
                                mask=valid_mask,
                                weights=weights,
                            )
                        )
            if kd_terms:
                loss = loss + lambda_kd * torch.stack(kd_terms).mean()
            if relation_terms:
                loss = loss + lambda_rel * torch.stack(relation_terms).mean()

            cmd_terms = []
            if method == "dlg_ours" and learned_stages:
                teacher_logits = output["logits"].detach()
                for old_stage in learned_stages:
                    old_modalities = stages[old_stage]
                    student_logits = model(batch, task_name="emotion", active_modalities=old_modalities)["logits"]
                    cmd_terms.append(_sequence_kd(student_logits, teacher_logits, batch["labels"]["emotion"], temperature))
            if cmd_terms:
                loss = loss + lambda_cmd * torch.stack(cmd_terms).mean()

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def _evaluate_stage(
    model: nn.Module,
    examples: list[DialogueExample],
    learned_stages: list[str],
    stages: dict[str, list[str]],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    train_cfg: dict,
    device: torch.device,
    method: str,
    current_stage: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model.eval()
    for eval_stage in learned_stages:
        eval_modalities = stages[eval_stage]
        available, missing = filter_dialogues_by_modalities(examples, feature_root, eval_modalities)
        if missing:
            LOGGER.warning("Dialogue modality eval %s skipped missing dialogues: %s", eval_stage, missing)
        loader = _build_loader(
            available,
            feature_root,
            feature_dims,
            speaker_to_id,
            eval_modalities,
            all_modalities,
            int(train_cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch, task_name="emotion", active_modalities=eval_modalities)
            labels = batch["labels"]["emotion"]
            mask = labels != IGNORE_INDEX
            y_true.extend(labels[mask].detach().cpu().tolist())
            y_pred.extend(output["logits"].argmax(dim=-1)[mask].detach().cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred, num_labels=7)
        rows.append(
            {
                "method": method,
                "stage": current_stage,
                "train_modalities": "+".join(stages[current_stage]),
                "eval_modalities": "+".join(eval_modalities),
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "final_avg": "",
                "modality_forgetting": "",
                "modality_retention": "",
                "modality_gain": "",
                "num_eval_dialogues": len(available),
                "num_eval_utterances": len(y_true),
            }
        )
    return rows


def _decorate_modality_metrics(rows: list[dict[str, object]], stage_order: list[str]) -> list[dict[str, object]]:
    if not rows:
        return rows
    final_stage = stage_order[-1]
    by_eval: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_eval.setdefault(str(row["eval_modalities"]), []).append(row)
    final_rows = [row for row in rows if row["stage"] == final_stage]
    final_avg = sum(float(row["weighted_f1"]) for row in final_rows) / len(final_rows) if final_rows else 0.0
    final_scores = {str(row["eval_modalities"]): float(row["weighted_f1"]) for row in final_rows}
    for row in rows:
        if row["stage"] != final_stage:
            continue
        eval_key = str(row["eval_modalities"])
        history = by_eval.get(eval_key, [])
        best = max(float(item["weighted_f1"]) for item in history) if history else 0.0
        final = final_scores.get(eval_key, 0.0)
        row["final_avg"] = final_avg
        row["modality_forgetting"] = max(0.0, best - final)
        row["modality_retention"] = final / best if best > 0 else 0.0
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
) -> DataLoader:
    dataset = MeldDialogueFeatureDataset(
        examples,
        feature_root=feature_root,
        feature_dims=feature_dims,
        speaker_to_id=speaker_to_id,
        active_modalities=active_modalities,
        all_modalities=all_modalities,
    )
    weighted_sampler = _build_weighted_sampler(examples) if shuffle and sampler == "weighted_random" else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if weighted_sampler is None else False,
        sampler=weighted_sampler,
        num_workers=num_workers,
        collate_fn=collate_dialogue_batch,
    )


def _build_weighted_sampler(examples: list[DialogueExample]) -> WeightedRandomSampler | None:
    if not examples:
        return None
    labels = [example.emotion_labels[0] if example.emotion_labels else 0 for example in examples]
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
    return cross_modal_kd_loss(
        student_logits.reshape(-1, student_logits.shape[-1])[mask],
        teacher_logits.reshape(-1, teacher_logits.shape[-1])[mask],
        temperature=temperature,
        confidence_weighted=False,
    )


def _parse_stages(modality_cfg: dict, stage_order: list[str]) -> dict[str, list[str]]:
    configured = modality_cfg.get("stages", {})
    if not configured:
        configured = {
            "text": ["text"],
            "text_audio": ["text", "audio"],
            "full": ["text", "audio", "visual"],
        }
    return {stage: list(configured[stage]) for stage in stage_order}


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "method",
        "stage",
        "train_modalities",
        "eval_modalities",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "final_avg",
        "modality_forgetting",
        "modality_retention",
        "modality_gain",
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
