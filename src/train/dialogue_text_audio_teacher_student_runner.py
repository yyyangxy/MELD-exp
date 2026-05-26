from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.datasets import build_speaker_vocab
from src.data.dialogue_dataset import IGNORE_INDEX, build_dialogue_examples
from src.data.meld_csv import read_all_splits
from src.data.stl_task_splits import (
    build_dialogue_examples_by_task_split,
    load_stl_task_split,
    log_stl_task_split_summary,
    resolve_stl_task_split_root,
)
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.models.stl_model import TASK_NUM_LABELS
from src.train.dialogue_text_audio_task_e2e_runner import (
    DialogueTextAudioTaskE2EModel,
    _build_loader,
    _clone_frozen,
    _move_batch,
    _resolve_device,
    _resolve_optional_model_path,
    _resolve_text_model_path,
    _save_checkpoint,
    _sequence_ce,
)
from src.train.metrics import compute_classification_metrics, decorate_final_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
S6_TEXT_AUDIO_TEACHER_STUDENT_METHODS = {
    "s6_text_student_ta_teacher",
    "s6_text_student_ta_teacher_sa",
}


def run_s6_text_audio_teacher_student_experiment(
    config_path: str | Path,
    method: str,
    run_name: str | None = None,
    train_overrides: dict[str, Any] | None = None,
) -> Path:
    if method not in S6_TEXT_AUDIO_TEACHER_STUDENT_METHODS:
        raise ValueError(f"Unknown S6 text/audio teacher-student method '{method}'.")

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = load_config(config_path)
    if train_overrides:
        _apply_overrides(config, train_overrides)
    config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "dialogue_text_audio_teacher_student_stl"
    if run_name:
        config.setdefault("run", {})["name"] = run_name
    output_dir = resolve_experiment_output_dir(config)
    setup_logging(output_dir / "logs" / f"{method}.log")
    _write_run_parameters(output_dir, config, method, train_overrides or {})
    seed_everything(int(config.get("seed", 13)))

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    continual_cfg = config.get("continual", {})
    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    data_root = resolve_data_root(config)
    split_records = read_all_splits(data_root, warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)))
    raw_dialogues = {split: build_dialogue_examples(records) for split, records in split_records.items()}
    split_root = resolve_stl_task_split_root(data_cfg, data_root)
    if split_root is not None:
        task_split = load_stl_task_split(split_root, task_order, split_records.keys())
        dialogue_examples = build_dialogue_examples_by_task_split(raw_dialogues, task_order, task_split)
        LOGGER.info("Using fixed STL task split root: %s", task_split.root)
    else:
        dialogue_examples = build_dialogue_examples_by_task_split(raw_dialogues, task_order, None)
        LOGGER.warning("No data.stl_task_split_root configured; S6 uses full split data for every task.")
    log_stl_task_split_summary(dialogue_examples=dialogue_examples)

    records_by_split_key = {split: {r.utterance_key: r for r in records} for split, records in split_records.items()}
    speaker_to_id = build_speaker_vocab(split_records["train"])
    text_model_path = _resolve_text_model_path(str(config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large")))
    audio_model_path = _resolve_optional_model_path(config.get("feature_paths", {}).get("audio_model_path"))
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    student = _build_model(text_model_path, audio_model_path, speaker_to_id, task_order, model_cfg).to(device)
    student_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    teacher_epochs = int(train_cfg.get("teacher_epochs", max(1, epochs // 2)))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    total_steps = max(1, sum(len(dialogue_examples["train"][task]) for task in task_order) * epochs // max(grad_accum_steps, 1))
    student_scheduler = get_linear_schedule_with_warmup(student_optimizer, int(total_steps * 0.1), total_steps)
    student_scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda"))

    text_teacher = None
    learned_tasks: list[str] = []
    rows: list[dict[str, object]] = []
    for stage_index, task in enumerate(task_order, start=1):
        loader = _build_loader(
            dialogue_examples["train"][task],
            records_by_split_key["train"],
            tokenizer,
            speaker_to_id,
            int(train_cfg.get("batch_size", 1)),
            shuffle=True,
            sampler=str(continual_cfg.get("sampler", "")),
            sampler_task=task,
            model_cfg=model_cfg,
        )
        multimodal_teacher = _build_model(text_model_path, audio_model_path, speaker_to_id, task_order, model_cfg).to(device)
        _train_current_task_teacher(
            multimodal_teacher,
            loader,
            task,
            device,
            train_cfg,
            teacher_epochs,
            grad_accum_steps,
            LOGGER,
        )
        multimodal_teacher = _clone_frozen(multimodal_teacher, device)

        for epoch in range(1, epochs + 1):
            loss = _train_student_epoch(
                student,
                multimodal_teacher,
                text_teacher,
                loader,
                learned_tasks,
                task,
                student_optimizer,
                student_scheduler,
                student_scaler,
                device,
                train_cfg,
                continual_cfg,
                method,
                grad_accum_steps,
            )
            LOGGER.info("Epoch %d/%d task=%s method=%s student_loss=%.4f", epoch, epochs, task, method, loss)
        learned_tasks.append(task)
        text_teacher = _clone_frozen(student, device)
        _save_checkpoint(student, output_dir, method, f"stage{stage_index}_{task}_student_text_only")
        _save_checkpoint(multimodal_teacher, output_dir, method, f"stage{stage_index}_{task}_ta_teacher")
        rows.extend(
            _evaluate_text_only(
                student,
                dialogue_examples[str(data_cfg.get("eval_split", "test"))],
                records_by_split_key[str(data_cfg.get("eval_split", "test"))],
                tokenizer,
                speaker_to_id,
                learned_tasks,
                device,
                train_cfg,
                model_cfg,
                method,
                f"stage_{stage_index}_{task}",
            )
        )

    rows = decorate_final_metrics(rows, task_order)
    result_path = output_dir / "results" / "dialogue_text_audio_teacher_student_results.csv"
    _write_rows(result_path, rows)
    LOGGER.info("Wrote S6 text/audio teacher-student results to %s", result_path)
    return result_path


def _build_model(
    text_model_path: str,
    audio_model_path: str | None,
    speaker_to_id: dict[str, int],
    task_order: list[str],
    model_cfg: dict[str, Any],
) -> DialogueTextAudioTaskE2EModel:
    return DialogueTextAudioTaskE2EModel(
        text_model_path=text_model_path,
        num_speakers=len(speaker_to_id),
        task_order=task_order,
        audio_encoder_type=str(model_cfg.get("audio_encoder_type", "raw")),
        audio_model_path=audio_model_path,
        text_dim=int(model_cfg.get("e2e_text_dim", 512)),
        audio_dim=int(model_cfg.get("e2e_audio_dim", 256)),
        fusion_dim=int(model_cfg.get("fusion_output_dim", 256)),
        speaker_dim=int(model_cfg.get("speaker_embedding_dim", 32)),
        dialogue_hidden_dim=int(model_cfg.get("dialogue_hidden_dim", 256)),
        dialogue_num_layers=int(model_cfg.get("dialogue_num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def _train_current_task_teacher(
    model,
    loader,
    task,
    device,
    train_cfg,
    epochs,
    grad_accum_steps,
    logger,
) -> None:
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("teacher_lr", train_cfg.get("lr", 2e-5))),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    total_steps = max(1, len(loader) * epochs // max(grad_accum_steps, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        steps = 0
        for step, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            _set_audio_enabled(batch, True)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
                output = model(batch, task)
                loss = _sequence_ce(criterion, output["logits"], batch["labels"][task])
                loss = loss / max(grad_accum_steps, 1)
            scaler.scale(loss).backward()
            if step % max(grad_accum_steps, 1) == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                grad_clip = float(train_cfg.get("grad_clip", 1.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * max(grad_accum_steps, 1)
            steps += 1
        logger.info("Teacher epoch %d/%d task=%s loss=%.4f", epoch, epochs, task, total / max(steps, 1))


def _train_student_epoch(
    student,
    multimodal_teacher,
    text_teacher,
    loader,
    learned_tasks,
    task,
    optimizer,
    scheduler,
    scaler,
    device,
    train_cfg,
    continual_cfg,
    method,
    grad_accum_steps,
) -> float:
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
            _set_audio_enabled(batch, False)
            student_out = student(batch, task)
            loss = _sequence_ce(criterion, student_out["logits"], batch["labels"][task])
            current_mask = batch["labels"][task] != IGNORE_INDEX
            with torch.no_grad():
                _set_audio_enabled(batch, True)
                ta_teacher_out = multimodal_teacher(batch, task)
                _set_audio_enabled(batch, False)
            current_weights = confidence_weights(ta_teacher_out["logits"], current_mask) if method == "s6_text_student_ta_teacher_sa" else None
            loss = loss + float(continual_cfg.get("lambda_ta_kd", continual_cfg.get("lambda_kd", 1.0))) * masked_kd_loss(
                student_out["logits"],
                ta_teacher_out["logits"],
                mask=current_mask,
                temperature=float(continual_cfg.get("temperature", 2.0)),
                weights=current_weights,
            )
            if method == "s6_text_student_ta_teacher_sa":
                loss = loss + float(continual_cfg.get("lambda_ta_rel", continual_cfg.get("lambda_rel", 1.0))) * sample_relation_loss(
                    student_out["embedding"],
                    ta_teacher_out["embedding"],
                    mask=current_mask,
                    weights=current_weights,
                )

            if text_teacher is not None and learned_tasks:
                old_kd_terms = []
                old_rel_terms = []
                with torch.no_grad():
                    text_teacher_base = text_teacher(batch, learned_tasks[0])
                for old_task in learned_tasks:
                    teacher_old = text_teacher.classify_embedding(text_teacher_base["embedding"], old_task)
                    student_old = student.classify_embedding(student_out["embedding"], old_task)
                    old_mask = batch["labels"][old_task] != IGNORE_INDEX
                    old_weights = confidence_weights(teacher_old["logits"], old_mask) if method == "s6_text_student_ta_teacher_sa" else None
                    old_kd_terms.append(
                        masked_kd_loss(
                            student_old["logits"],
                            teacher_old["logits"],
                            mask=old_mask,
                            temperature=float(continual_cfg.get("temperature", 2.0)),
                            weights=old_weights,
                        )
                    )
                    if method == "s6_text_student_ta_teacher_sa":
                        old_rel_terms.append(sample_relation_loss(student_old["embedding"], teacher_old["embedding"], mask=old_mask, weights=old_weights))
                loss = loss + float(continual_cfg.get("lambda_kd", 1.0)) * torch.stack(old_kd_terms).mean()
                if old_rel_terms:
                    loss = loss + float(continual_cfg.get("lambda_rel", 1.0)) * torch.stack(old_rel_terms).mean()
            loss = loss / max(grad_accum_steps, 1)
        scaler.scale(loss).backward()
        if step % max(grad_accum_steps, 1) == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            grad_clip = float(train_cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu()) * max(grad_accum_steps, 1)
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def _evaluate_text_only(model, examples_by_task, records_by_key, tokenizer, speaker_to_id, learned_tasks, device, train_cfg, model_cfg, method, stage):
    rows = []
    model.eval()
    for task in learned_tasks:
        loader = _build_loader(
            examples_by_task[task],
            records_by_key,
            tokenizer,
            speaker_to_id,
            int(train_cfg.get("batch_size", 1)),
            shuffle=False,
            sampler="",
            sampler_task=task,
            model_cfg=model_cfg,
        )
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loader:
            batch = _move_batch(batch, device)
            _set_audio_enabled(batch, False)
            output = model(batch, task)
            labels = batch["labels"][task]
            mask = labels != IGNORE_INDEX
            y_true.extend(labels[mask].detach().cpu().tolist())
            y_pred.extend(output["logits"].argmax(dim=-1)[mask].detach().cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred, TASK_NUM_LABELS[task], positive_label=1 if task == "shift" else None)
        rows.append(
            {
                "method": method,
                "stage": stage,
                "task": task,
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "positive_f1_for_shift": metrics["positive_f1_for_shift"] or "",
                "final_avg": "",
                "final_avg_weighted_f1": "",
                "final_avg_accuracy": "",
                "forgetting": "",
                "retention": "",
            }
        )
    return rows


def _set_audio_enabled(batch: dict[str, Any], enabled: bool) -> None:
    value = 1.0 if enabled else 0.0
    batch["audio_enabled"] = torch.full_like(batch["audio_enabled"], value)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
        "final_avg_weighted_f1",
        "final_avg_accuracy",
        "forgetting",
        "retention",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    train_cfg = config.setdefault("train", {})
    model_cfg = config.setdefault("model", {})
    feature_cfg = config.setdefault("feature_paths", {})
    continual_cfg = config.setdefault("continual", {})
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "seed":
            config["seed"] = int(value)
        elif key == "text_model_path":
            feature_cfg["text_model_path"] = str(value)
        elif key == "audio_model_path":
            feature_cfg["audio_model_path"] = str(value)
        elif key == "audio_encoder_type":
            model_cfg["audio_encoder_type"] = str(value)
        elif key in {"max_length", "max_audio_seconds", "audio_sample_rate", "audio_cache_root"}:
            model_cfg[key] = value
        elif key in {"lambda_ta_kd", "lambda_ta_rel", "lambda_kd", "lambda_rel", "temperature"}:
            continual_cfg[key] = value
        else:
            train_cfg[key] = value


def _write_run_parameters(output_dir: Path, config: dict[str, Any], method: str, train_overrides: dict[str, Any]) -> None:
    payload = {
        "method": method,
        "cli_train_overrides": {k: v for k, v in train_overrides.items() if v is not None},
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "student_eval": "text_only",
        "teacher_train": "text_audio",
    }
    path = ensure_dir(output_dir / "logs") / "run_parameters.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
