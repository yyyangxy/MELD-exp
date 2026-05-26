from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.continual.cross_modal_distillation import cross_modal_kd_loss
from src.continual.multimodal_memory import MultimodalPrototypeMemory
from src.data.datasets import build_speaker_vocab
from src.data.meld_csv import read_all_splits
from src.data.multimodal_dataset import (
    MeldMultimodalFeatureDataset,
    collate_multimodal_batch,
    filter_examples_by_modalities,
    summarize_feature_coverage,
)
from src.data.task_builder import TaskExample, build_task_examples
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.models.multimodal_model import MultimodalSTLModel
from src.train.metrics import compute_classification_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
MODALITY_METHODS = {"mod_seq_ft", "mod_seq_kd", "prototype_replay", "cmcrd_ours", "utt_modality_sa_cmd"}


def run_modality_experiment(config_path: str | Path, method: str, run_name: str | None = None) -> Path:
    if method not in MODALITY_METHODS:
        raise ValueError(f"Unknown modality method '{method}'. Expected {sorted(MODALITY_METHODS)}")

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
    modality_cfg = config.get("modalities", {})

    all_modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
    stage_order = list(modality_cfg.get("stage_order", ["text", "text_audio", "full"]))
    stages = _parse_stages(modality_cfg, stage_order)
    feature_dims = {key: int(value) for key, value in modality_cfg.get("feature_dims", {}).items()}
    feature_root = str(resolve_path(modality_cfg.get("feature_root", train_cfg.get("output_dir", "outputs")), PROJECT_ROOT))
    eval_split = str(data_cfg.get("eval_split", "test"))

    split_records = read_all_splits(
        data_root,
        warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)),
    )
    examples_by_split = {
        split: build_task_examples(records, "emotion", context_window=int(model_cfg.get("context_window", 3)))
        for split, records in split_records.items()
    }
    speaker_to_id = build_speaker_vocab(split_records["train"])
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    for split, examples in examples_by_split.items():
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
        task_num_labels={"emotion": 7},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    sampler = str(continual_cfg.get("sampler", train_cfg.get("sampler", "")) or "")
    memory = (
        MultimodalPrototypeMemory(
            memory_per_class=int(continual_cfg.get("memory_per_class", 20)),
            batch_size=batch_size,
            device=device,
            memory_strategy=str(continual_cfg.get("memory_strategy", "prototype_nearest")),
            representative_ratio=float(continual_cfg.get("representative_ratio", 0.5)),
            kmeans_iters=int(continual_cfg.get("kmeans_iters", 10)),
            seed=int(continual_cfg.get("seed", 13)),
        )
        if method in {"prototype_replay", "cmcrd_ours"}
        else None
    )

    teacher_by_stage: dict[str, nn.Module] = {}
    learned_stages: list[str] = []
    rows: list[dict[str, object]] = []

    for stage_name in stage_order:
        active_modalities = stages[stage_name]
        train_examples, missing_counts = filter_examples_by_modalities(
            examples_by_split["train"],
            feature_root,
            active_modalities,
        )
        if missing_counts:
            LOGGER.warning("Stage %s skipped missing train features: %s", stage_name, missing_counts)
        if not train_examples:
            raise RuntimeError(
                f"No train examples have all required features for stage {stage_name}: {active_modalities}"
            )

        train_loader = _build_loader(
            train_examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            active_modalities,
            all_modalities,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            sampler=sampler,
        )
        replay_loaders = _build_replay_loaders(
            memory,
            learned_stages,
            feature_root,
            feature_dims,
            speaker_to_id,
            all_modalities,
            batch_size,
            int(train_cfg.get("num_workers", 0)),
            sampler,
        )

        loss = _train_stage(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            method=method,
            stage_name=stage_name,
            active_modalities=active_modalities,
            learned_stages=learned_stages,
            stages=stages,
            teacher_by_stage=teacher_by_stage,
            replay_loaders=replay_loaders,
            epochs=int(train_cfg.get("epochs", 5)),
            grad_clip=float(train_cfg.get("grad_clip", 5.0)),
            lambda_kd=float(continual_cfg.get("lambda_kd", 0.5)),
            lambda_cmd=float(continual_cfg.get("lambda_cmd", 0.5)),
            lambda_rel=float(continual_cfg.get("lambda_rel", continual_cfg.get("lambda_cmd", 1.0))),
            temperature=float(continual_cfg.get("temperature", 2.0)),
            eval_interval=int(train_cfg.get("eval_interval", 0)),
            eval_callback=lambda epoch, stage_name=stage_name: rows.extend(
                _with_stage_label(
                    _evaluate_stage(
                        model,
                        examples_by_split[eval_split],
                        [*learned_stages, stage_name],
                        stages,
                        feature_root,
                        feature_dims,
                        speaker_to_id,
                        all_modalities,
                        batch_size,
                        int(train_cfg.get("num_workers", 0)),
                        device,
                        method,
                        current_stage=stage_name,
                    ),
                    f"{stage_name}_ep{epoch}",
                )
            ),
        )
        LOGGER.info("Finished modality stage %s active=%s loss=%.4f", stage_name, active_modalities, loss)

        learned_stages.append(stage_name)
        if memory is not None:
            memory.update(
                stage_name,
                train_examples,
                active_modalities=active_modalities,
                model=model,
                feature_root=feature_root,
                feature_dims=feature_dims,
                speaker_to_id=speaker_to_id,
                all_modalities=all_modalities,
            )
        if method in {"mod_seq_kd", "cmcrd_ours", "utt_modality_sa_cmd"}:
            teacher_by_stage[stage_name] = _clone_frozen(model, device)

        _save_checkpoint(model, output_dir, method, stage_name)
        rows.extend(
            _evaluate_stage(
                model,
                examples_by_split[eval_split],
                learned_stages,
                stages,
                feature_root,
                feature_dims,
                speaker_to_id,
                all_modalities,
                batch_size,
                int(train_cfg.get("num_workers", 0)),
                device,
                method,
                current_stage=stage_name,
            )
        )

    rows = _decorate_modality_metrics(rows, stage_order)
    result_path = output_dir / "results" / "modality_stl_results.csv"
    _append_rows(result_path, rows)
    LOGGER.info("Wrote modality results to %s", result_path)
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
    replay_loaders: dict[str, DataLoader],
    epochs: int,
    grad_clip: float,
    lambda_kd: float,
    lambda_cmd: float,
    lambda_rel: float,
    temperature: float,
    eval_interval: int = 0,
    eval_callback=None,
) -> float:
    criterion = nn.CrossEntropyLoss()
    replay_iters = {name: _infinite(loader) for name, loader in replay_loaders.items()}
    total_loss = 0.0
    steps = 0

    for epoch_index in range(1, epochs + 1):
        epoch_total_loss = 0.0
        epoch_steps = 0
        model.train()
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task_name="emotion", active_modalities=active_modalities)
            current_ce = criterion(output["logits"], batch["label"])
            supervised_terms = [current_ce]
            if method == "utt_modality_sa_cmd":
                for old_stage in learned_stages:
                    supervised_output = model(
                        batch,
                        task_name="emotion",
                        active_modalities=stages[old_stage],
                    )
                    supervised_terms.append(criterion(supervised_output["logits"], batch["label"]))

            kd_terms = []
            relation_terms = []
            if method in {"mod_seq_kd", "cmcrd_ours", "utt_modality_sa_cmd"}:
                for old_stage in learned_stages:
                    teacher = teacher_by_stage.get(old_stage)
                    if teacher is None:
                        continue
                    old_modalities = stages[old_stage]
                    with torch.no_grad():
                        teacher_output = teacher(
                            batch,
                            task_name="emotion",
                            active_modalities=old_modalities,
                        )
                    student_output = model(
                        batch,
                        task_name="emotion",
                        active_modalities=old_modalities,
                    )
                    weights = (
                        confidence_weights(teacher_output["logits"])
                        if method == "utt_modality_sa_cmd"
                        else None
                    )
                    kd_terms.append(
                        masked_kd_loss(
                            student_output["logits"],
                            teacher_output["logits"],
                            temperature=temperature,
                            weights=weights,
                        )
                    )
                    if method == "utt_modality_sa_cmd":
                        relation_terms.append(
                            sample_relation_loss(
                                student_output["embedding"],
                                teacher_output["embedding"],
                                weights=weights,
                            )
                        )
            replay_terms = []
            if method in {"prototype_replay", "cmcrd_ours"}:
                for replay_stage, iterator in replay_iters.items():
                    replay_batch = _move_batch(next(iterator), device)
                    replay_output = model(
                        replay_batch,
                        task_name="emotion",
                        active_modalities=stages[replay_stage],
                    )
                    replay_terms.append(criterion(replay_output["logits"], replay_batch["label"]))
            if replay_terms:
                supervised_terms.extend(replay_terms)
            loss = torch.stack(supervised_terms).mean()
            if kd_terms:
                loss = loss + lambda_kd * torch.stack(kd_terms).mean()
            if relation_terms:
                loss = loss + lambda_rel * torch.stack(relation_terms).mean()

            cmd_terms = []
            if method == "cmcrd_ours" and learned_stages:
                teacher_logits = output["logits"].detach()
                for old_stage in learned_stages:
                    partial_modalities = stages[old_stage]
                    student_logits = model(
                        batch,
                        task_name="emotion",
                        active_modalities=partial_modalities,
                    )["logits"]
                    cmd_terms.append(
                        cross_modal_kd_loss(
                            student_logits,
                            teacher_logits,
                            temperature=temperature,
                            confidence_weighted=True,
                        )
                    )
            if cmd_terms:
                loss = loss + lambda_cmd * torch.stack(cmd_terms).mean()

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            loss_float = float(loss.detach().cpu())
            total_loss += loss_float
            epoch_total_loss += loss_float
            epoch_steps += 1
            steps += 1
        LOGGER.info(
            "Epoch %d/%d stage=%s method=%s total_loss=%.4f",
            epoch_index,
            epochs,
            stage_name,
            method,
            epoch_total_loss / max(epoch_steps, 1),
        )
        if eval_interval > 0 and eval_callback is not None and epoch_index % eval_interval == 0:
            eval_callback(epoch_index)

    return total_loss / max(steps, 1)


@torch.no_grad()
def _evaluate_stage(
    model: nn.Module,
    eval_examples: list[TaskExample],
    learned_stages: list[str],
    stages: dict[str, list[str]],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    method: str,
    current_stage: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model.eval()
    for eval_stage in learned_stages:
        eval_modalities = stages[eval_stage]
        available_examples, missing_counts = filter_examples_by_modalities(
            eval_examples,
            feature_root,
            eval_modalities,
        )
        if missing_counts:
            LOGGER.warning("Eval %s skipped missing features: %s", eval_stage, missing_counts)
        loader = _build_loader(
            available_examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            eval_modalities,
            all_modalities,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch, task_name="emotion", active_modalities=eval_modalities)
            y_true.extend(batch["label"].detach().cpu().tolist())
            y_pred.extend(output["logits"].argmax(dim=-1).detach().cpu().tolist())
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
                "num_eval_examples": len(available_examples),
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
    final_avg = (
        sum(float(row["weighted_f1"]) for row in final_rows) / len(final_rows)
        if final_rows
        else 0.0
    )
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

    text_score = final_scores.get("text")
    text_audio_score = final_scores.get("text+audio")
    full_score = final_scores.get("text+audio+visual")
    for row in final_rows:
        eval_key = str(row["eval_modalities"])
        if eval_key == "text+audio" and text_score is not None:
            row["modality_gain"] = float(row["weighted_f1"]) - text_score
        elif eval_key == "text+audio+visual" and text_audio_score is not None:
            row["modality_gain"] = float(row["weighted_f1"]) - text_audio_score
        elif eval_key == "text+audio+visual" and full_score is not None and text_score is not None:
            row["modality_gain"] = full_score - text_score
    return rows


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


def _build_replay_loaders(
    memory: MultimodalPrototypeMemory | None,
    learned_stages: list[str],
    feature_root: str,
    feature_dims: dict[str, int],
    speaker_to_id: dict[str, int],
    all_modalities: list[str],
    batch_size: int,
    num_workers: int,
    sampler: str = "",
) -> dict[str, DataLoader]:
    if memory is None:
        return {}
    loaders: dict[str, DataLoader] = {}
    for stage_name in learned_stages:
        examples = memory.examples_for(stage_name)
        if not examples:
            continue
        loaders[stage_name] = _build_loader(
            examples,
            feature_root,
            feature_dims,
            speaker_to_id,
            memory.stage_modalities[stage_name],
            all_modalities,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            sampler=sampler,
        )
    return loaders


def _parse_stages(modality_cfg: dict, stage_order: list[str]) -> dict[str, list[str]]:
    configured = modality_cfg.get("stages", {})
    if not configured:
        configured = {
            "text": ["text"],
            "text_audio": ["text", "audio"],
            "full": ["text", "audio", "visual"],
        }
    stages = {stage: list(configured[stage]) for stage in stage_order}
    return stages


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
        "num_eval_examples",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _with_stage_label(rows: list[dict[str, object]], stage_label: str) -> list[dict[str, object]]:
    for row in rows:
        row["stage"] = stage_label
    return rows


def _save_checkpoint(model: nn.Module, output_dir: Path, method: str, stage_name: str) -> None:
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    torch.save(model.state_dict(), checkpoint_dir / f"{method}_{stage_name}.pt")


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


def _output_dir(config: dict) -> Path:
    return resolve_experiment_output_dir(config)
