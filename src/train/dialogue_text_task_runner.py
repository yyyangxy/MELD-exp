from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.dialogue_dataset import IGNORE_INDEX, DialogueExample, build_dialogue_examples
from src.data.meld_csv import read_all_splits
from src.data.stl_task_splits import (
    build_dialogue_examples_by_task_split,
    load_stl_task_split,
    log_stl_task_split_summary,
    resolve_stl_task_split_root,
)
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.losses.task_relation import prototype_alignment_loss, task_relation_distillation_loss
from src.continual.task_prototype_bank import TaskPrototypeBank
from src.models.stl_model import TASK_NUM_LABELS
from src.train.metrics import compute_classification_metrics, decorate_final_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
DIALOGUE_TEXT_TASK_METHODS = {
    "context_free",
    "hier_bilstm",
    "dlg_seq_ft",
    "dlg_seq_kd",
    "dlg_random_replay",
    "dlg_ours",
    "dlg_sa_cmd_no_replay",
    "dlg_task_sa_cmd",
    "dlg_text_task_sa_cmd",
    "text_task_sa_cmd",
    "dlg_text_task_sa_cmd_replay_kd",
    "text_task_sa_cmd_replay_kd",
    "dlg_text_task_sa_cmd_freeze_old_heads",
    "text_task_sa_cmd_freeze_old_heads",
    "dlg_text_task_sa_cmd_replay_kd_freeze_old_heads",
    "text_task_sa_cmd_replay_kd_freeze_old_heads",
    "dlg_task_pg_trd",
    "dlg_er",
    "dlg_icarl",
    "dlg_der",
    "dlg_derpp",
    "dlg_packnet",
    "dlg_ewc",
    "dlg_mas",
    "dlg_si",
}


class DialogueTextDataset(Dataset):
    def __init__(self, examples: list[DialogueExample], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pad_token_id": int(self.tokenizer.pad_token_id or 0),
            "labels": {
                "sentiment": torch.tensor(example.sentiment_labels, dtype=torch.long),
                "emotion": torch.tensor(example.emotion_labels, dtype=torch.long),
                "shift": torch.tensor(example.shift_labels, dtype=torch.long),
            },
            "length": torch.tensor(example.length, dtype=torch.long),
            "dialogue_id": example.dialogue_id,
            "texts": example.texts,
        }


class XLMRDialogueTaskModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        task_order: list[str],
        use_dialogue_encoder: bool,
        dialogue_hidden_dim: int = 256,
        dialogue_num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = int(self.encoder.config.hidden_size)
        self.use_dialogue_encoder = use_dialogue_encoder
        output_dim = hidden_size
        if use_dialogue_encoder:
            self.dialogue_encoder = nn.LSTM(
                hidden_size,
                dialogue_hidden_dim,
                num_layers=dialogue_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if dialogue_num_layers > 1 else 0.0,
            )
            output_dim = dialogue_hidden_dim * 2
        else:
            self.dialogue_encoder = None
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({task: nn.Linear(output_dim, TASK_NUM_LABELS[task]) for task in task_order})

    def forward(self, batch: dict[str, Any], task_name: str) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, max_len, token_len = input_ids.shape
        flat_output = self.encoder(
            input_ids=input_ids.reshape(batch_size * max_len, token_len),
            attention_mask=attention_mask.reshape(batch_size * max_len, token_len),
        )
        embeddings = flat_output.last_hidden_state[:, 0].reshape(batch_size, max_len, -1)
        sequence_mask = batch["sequence_mask"]
        if self.dialogue_encoder is not None:
            lengths = batch["lengths"].detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            encoded, _ = self.dialogue_encoder(packed)
            embeddings, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=max_len)
        embeddings = self.dropout(embeddings)
        logits = self.heads[task_name](embeddings)
        return {"logits": logits, "embedding": embeddings, "sequence_mask": sequence_mask}

    def classify_embedding(self, embedding: torch.Tensor, task_name: str) -> dict[str, torch.Tensor]:
        return {"logits": self.heads[task_name](embedding), "embedding": embedding}


def run_dialogue_text_task_experiment(
    config_path: str | Path,
    method: str,
    run_name: str | None = None,
    train_overrides: dict[str, Any] | None = None,
) -> Path:
    if method not in DIALOGUE_TEXT_TASK_METHODS:
        raise ValueError(f"Unknown dialogue text Task-STL method '{method}'. Expected {sorted(DIALOGUE_TEXT_TASK_METHODS)}")

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = load_config(config_path)
    if train_overrides:
        apply_train_overrides(config, train_overrides)
    config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "dialogue_text_task_stl"
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
        LOGGER.warning("No data.stl_task_split_root configured; dialogue text Task-STL uses full split data for every task.")
    log_stl_task_split_summary(dialogue_examples=dialogue_examples)

    model_path = _resolve_text_model_path(str(config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large")))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    device = _resolve_device(str(train_cfg.get("device", "auto")))
    batch_size = int(train_cfg.get("batch_size", 4))
    max_length = int(model_cfg.get("max_length", 128))
    loaders = {
        split: {
            task: _build_loader(
                examples,
                tokenizer,
                max_length,
                batch_size,
                shuffle=split == "train",
                sampler_task=task,
            )
            for task, examples in examples_by_task.items()
        }
        for split, examples_by_task in dialogue_examples.items()
    }

    model = XLMRDialogueTaskModel(
        model_path=model_path,
        task_order=task_order,
        use_dialogue_encoder=method != "context_free",
        dialogue_hidden_dim=int(model_cfg.get("dialogue_hidden_dim", 256)),
        dialogue_num_layers=int(model_cfg.get("dialogue_num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 5))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    total_steps = sum(len(loaders["train"][task]) for task in task_order) * epochs
    total_steps = max(1, total_steps // max(grad_accum_steps, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda"))

    rows = (
        _run_joint(
            model,
            loaders,
            task_order,
            optimizer,
            scheduler,
            scaler,
            train_cfg,
            device,
            method,
            str(data_cfg.get("eval_split", "test")),
        )
        if method in {"context_free", "hier_bilstm"}
        else _run_sequence(
            model,
            loaders,
            task_order,
            optimizer,
            scheduler,
            scaler,
            train_cfg,
            continual_cfg,
            device,
            method,
            output_dir,
            str(data_cfg.get("eval_split", "test")),
            dialogue_examples["train"],
            tokenizer,
            max_length,
            batch_size,
            int(config.get("seed", 13)),
        )
    )
    rows = decorate_final_metrics(rows, task_order)
    result_path = output_dir / "results" / "dialogue_text_task_stl_results.csv"
    _append_rows(result_path, rows)
    _save_checkpoint(model, output_dir, method, "final")
    LOGGER.info("Wrote dialogue text Task-STL results to %s", result_path)
    return result_path


def apply_train_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    train_cfg = config.setdefault("train", {})
    continual_cfg = config.setdefault("continual", {})
    feature_cfg = config.setdefault("feature_paths", {})
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "seed":
            config["seed"] = int(value)
        elif key == "text_model_path":
            feature_cfg["text_model_path"] = str(value)
        elif key in {
            "memory_per_class",
            "replay_strategy",
            "representative_ratio",
            "klmap_dim",
            "replay_batch_kd",
            "freeze_old_heads",
            "cl_reg_lambda",
            "importance_max_batches",
            "regularizer_scope",
            "si_xi",
            "packnet_prune_ratio",
        }:
            continual_cfg[key] = value
        else:
            train_cfg[key] = value


def _write_run_parameters(
    output_dir: Path,
    config: dict[str, Any],
    method: str,
    train_overrides: dict[str, Any],
) -> None:
    train_cfg = config.get("train", {})
    continual_cfg = config.get("continual", {})
    payload = {
        "method": method,
        "cli_train_overrides": {
            key: value for key, value in train_overrides.items() if value is not None
        },
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "effective_train": {
            "epochs": int(train_cfg.get("epochs", 5)),
            "batch_size": int(train_cfg.get("batch_size", 4)),
            "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
            "effective_batch_size": int(train_cfg.get("batch_size", 4)) * int(train_cfg.get("grad_accum_steps", 1)),
            "lr": float(train_cfg.get("lr", 2e-5)),
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "text_model_path": str(config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large")),
            "memory_per_class": int(continual_cfg.get("memory_per_class", 100)),
            "replay_strategy": str(continual_cfg.get("replay_strategy", "random")),
            "representative_ratio": float(continual_cfg.get("representative_ratio", 0.5)),
            "klmap_dim": int(continual_cfg.get("klmap_dim", 50)),
            "replay_batch_kd": bool(continual_cfg.get("replay_batch_kd", False)),
            "freeze_old_heads": bool(continual_cfg.get("freeze_old_heads", False)),
            "cl_reg_lambda": float(continual_cfg.get("cl_reg_lambda", 1.0)),
            "importance_max_batches": int(continual_cfg.get("importance_max_batches", 50)),
            "regularizer_scope": str(continual_cfg.get("regularizer_scope", "non_encoder")),
            "si_xi": float(continual_cfg.get("si_xi", 0.1)),
            "packnet_prune_ratio": float(continual_cfg.get("packnet_prune_ratio", 0.5)),
            "device": str(train_cfg.get("device", "auto")),
            "fp16": bool(train_cfg.get("fp16", False)),
        },
    }
    path = ensure_dir(output_dir / "logs") / "run_parameters.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_joint(
    model,
    loaders,
    task_order,
    optimizer,
    scheduler,
    scaler,
    train_cfg,
    device,
    method,
    eval_split,
) -> list[dict[str, object]]:
    epochs = int(train_cfg.get("epochs", 5))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for epoch in range(1, epochs + 1):
        iterators = {task: iter(loaders["train"][task]) for task in task_order}
        active = set(task_order)
        optimizer.zero_grad(set_to_none=True)
        step = 0
        model.train()
        while active:
            for task in list(task_order):
                if task not in active:
                    continue
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    active.remove(task)
                    continue
                step += 1
                batch = _move_batch(batch, device)
                with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
                    output = model(batch, task)
                    loss = _sequence_ce(criterion, output["logits"], batch["labels"][task]) / max(grad_accum_steps, 1)
                _backward_step(loss, model, optimizer, scheduler, scaler, train_cfg, step, grad_accum_steps)
        LOGGER.info("Epoch %d/%d method=%s finished", epoch, epochs, method)
    return _evaluate(model, loaders[eval_split], task_order, device, method, "joint")


def _resolve_text_model_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith(".") or path.exists() or (PROJECT_ROOT / path).exists():
        return str(resolve_path(value, PROJECT_ROOT))
    return value


def _run_sequence(
    model,
    loaders,
    task_order,
    optimizer,
    scheduler,
    scaler,
    train_cfg,
    continual_cfg,
    device,
    method,
    output_dir,
    eval_split,
    train_examples_by_task,
    tokenizer,
    max_length,
    batch_size,
    seed,
) -> list[dict[str, object]]:
    epochs = int(train_cfg.get("epochs", 5))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    teacher = None
    learned_tasks: list[str] = []
    rows: list[dict[str, object]] = []
    prototype_bank = TaskPrototypeBank() if method == "dlg_task_pg_trd" else None
    replay_memory: dict[str, list[DialogueExample]] = {}
    replay_logits: dict[str, dict[int, torch.Tensor]] = {}
    regularization_state = _init_regularization_state(model, continual_cfg) if _uses_parameter_regularizer(method) else None
    si_state = None
    packnet_state = _init_packnet_state(model, continual_cfg) if method == "dlg_packnet" else None
    for stage_index, task in enumerate(task_order, start=1):
        if _uses_freeze_old_heads(method, continual_cfg):
            _freeze_task_heads(model, learned_tasks)
        if method == "dlg_si":
            si_state = _init_si_stage_state(model, continual_cfg)
        replay_loaders = {
            old_task: _build_loader(
                examples,
                tokenizer,
                max_length,
                batch_size,
                shuffle=True,
                sampler_task=old_task,
            )
            for old_task, examples in replay_memory.items()
            if examples
        }
        for epoch in range(1, epochs + 1):
            loss = _train_epoch(
                model,
                teacher,
                loaders["train"][task],
                learned_tasks,
                task,
                optimizer,
                scheduler,
                scaler,
                criterion,
                device,
                train_cfg,
                continual_cfg,
                method,
                prototype_bank,
                grad_accum_steps,
                replay_loaders,
                regularization_state,
                si_state,
                packnet_state,
                replay_logits,
            )
            LOGGER.info("Epoch %d/%d task=%s method=%s loss=%.4f", epoch, epochs, task, method, loss)
            eval_interval = int(train_cfg.get("eval_interval", 0))
            if eval_interval > 0 and epoch % eval_interval == 0:
                rows.extend(_evaluate(model, loaders[eval_split], [*learned_tasks, task], device, method, f"stage_{stage_index}_{task}_ep{epoch}"))
        learned_tasks.append(task)
        if _uses_random_replay(method):
            replay_memory[task] = _select_replay_dialogues(
                train_examples_by_task[task],
                task,
                int(continual_cfg.get("memory_per_class", 100)),
                str(continual_cfg.get("replay_strategy", "random")),
                float(continual_cfg.get("representative_ratio", 0.5)),
                int(continual_cfg.get("klmap_dim", 50)),
                seed=seed + stage_index,
                model=model,
                loader=loaders["train"][task],
                device=device,
            )
            if method in {"dlg_der", "dlg_derpp"}:
                replay_logits[task] = _collect_replay_logits(
                    model,
                    replay_memory[task],
                    task,
                    tokenizer,
                    max_length,
                    batch_size,
                    device,
                )
            LOGGER.info(
                "Stage %s task=%s selected replay dialogues=%s memory_per_class=%s strategy=%s digest=%s",
                stage_index,
                task,
                len(replay_memory[task]),
                int(continual_cfg.get("memory_per_class", 100)),
                str(continual_cfg.get("replay_strategy", "random")),
                _replay_dialogue_digest(replay_memory[task]),
            )
            _write_replay_selection(
                output_dir,
                method,
                stage_index,
                task,
                str(continual_cfg.get("replay_strategy", "random")),
                replay_memory[task],
            )
        _update_prototype_bank(prototype_bank, method, task, loaders["train"][task], model, device)
        if method in {"dlg_ewc", "dlg_mas"}:
            _update_gradient_importance(
                regularization_state,
                model,
                loaders["train"][task],
                task,
                criterion,
                device,
                method,
                train_cfg,
                continual_cfg,
            )
        if method == "dlg_si":
            _consolidate_si_importance(regularization_state, si_state, model, continual_cfg)
        if method == "dlg_packnet":
            _update_packnet_masks(packnet_state, model, continual_cfg)
        if _uses_teacher(method) or (_uses_replay_batch_kd(method, continual_cfg) and _uses_random_replay(method)):
            teacher = _clone_frozen(model, device)
        _save_checkpoint(model, output_dir, method, f"stage{stage_index}_{task}")
        if method == "dlg_icarl":
            rows.extend(
                _evaluate_icarl_nearest_mean(
                    model,
                    loaders[eval_split],
                    learned_tasks,
                    device,
                    method,
                    f"stage_{stage_index}_{task}",
                    replay_memory,
                    tokenizer,
                    max_length,
                    batch_size,
                )
            )
        else:
            rows.extend(_evaluate(model, loaders[eval_split], learned_tasks, device, method, f"stage_{stage_index}_{task}"))
    return rows


def _train_epoch(
    model,
    teacher,
    loader,
    learned_tasks,
    task,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    train_cfg,
    continual_cfg,
    method,
    prototype_bank,
    grad_accum_steps,
    replay_loaders,
    regularization_state=None,
    si_state=None,
    packnet_state=None,
    replay_logits=None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    steps = 0
    replay_iters = {old_task: _infinite(loader) for old_task, loader in replay_loaders.items()}
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        replay_count = len(replay_iters)
        supervised_weight = 1.0 / (1 + replay_count)
        with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
            output = model(batch, task)
            ce_loss = _sequence_ce(criterion, output["logits"], batch["labels"][task])
            loss = ce_loss
            if teacher is not None and learned_tasks and _uses_teacher(method):
                kd_terms = []
                rel_terms = []
                student_logits = {}
                teacher_logits = {}
                masks = {}
                with torch.no_grad():
                    teacher_base_out = teacher(batch, learned_tasks[0])
                student_embedding = output["embedding"]
                teacher_embedding = teacher_base_out["embedding"]
                for old_task in learned_tasks:
                    teacher_out = teacher.classify_embedding(teacher_embedding, old_task)
                    student_out = model.classify_embedding(student_embedding, old_task)
                    valid_mask = batch["labels"][old_task] != IGNORE_INDEX
                    weights = (
                        confidence_weights(teacher_out["logits"], valid_mask)
                        if _uses_confidence_relation(method)
                        else None
                    )
                    kd_terms.append(masked_kd_loss(student_out["logits"], teacher_out["logits"], mask=valid_mask, temperature=float(continual_cfg.get("temperature", 2.0)), weights=weights))
                    student_logits[old_task] = student_out["logits"]
                    teacher_logits[old_task] = teacher_out["logits"]
                    masks[old_task] = valid_mask
                    if _uses_confidence_relation(method):
                        rel_terms.append(sample_relation_loss(student_out["embedding"], teacher_out["embedding"], mask=valid_mask, weights=weights))
                    if method == "dlg_task_pg_trd" and prototype_bank is not None:
                        rel_terms.append(prototype_alignment_loss(student_out["embedding"], teacher_out["embedding"], prototype_bank.prototypes_for(old_task, device), mask=valid_mask))
                if kd_terms:
                    loss = loss + float(continual_cfg.get("lambda_kd", 1.0)) * torch.stack(kd_terms).mean()
                if method == "dlg_task_pg_trd" and student_logits:
                    rel_terms.append(task_relation_distillation_loss(student_logits, teacher_logits, masks_by_task=masks, temperature=float(continual_cfg.get("temperature", 2.0))))
                if rel_terms:
                    loss = loss + float(continual_cfg.get("lambda_rel", 1.0)) * torch.stack(rel_terms).mean()
            loss = loss * supervised_weight
            if _uses_parameter_regularizer(method):
                reg_loss = _parameter_regularization_loss(model, regularization_state)
                loss = loss + float(continual_cfg.get("cl_reg_lambda", 1.0)) * reg_loss
            loss = loss / max(grad_accum_steps, 1)
        scaler.scale(loss).backward()
        total_loss_value = float(loss.detach().cpu()) * max(grad_accum_steps, 1)
        for old_task, replay_iter in replay_iters.items():
            replay_batch = _move_batch(next(replay_iter), device)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
                replay_output = model(replay_batch, old_task)
                replay_loss = _sequence_ce(criterion, replay_output["logits"], replay_batch["labels"][old_task])
                if method in {"dlg_der", "dlg_derpp"}:
                    der_loss = _der_replay_loss(replay_output["logits"], replay_batch, old_task, replay_logits or {})
                    if method == "dlg_der":
                        replay_loss = float(continual_cfg.get("lambda_kd", 1.0)) * der_loss
                    else:
                        replay_loss = replay_loss + float(continual_cfg.get("lambda_kd", 1.0)) * der_loss
                if teacher is not None and _uses_replay_batch_kd(method, continual_cfg):
                    with torch.no_grad():
                        teacher_replay_out = teacher(replay_batch, old_task)
                    replay_mask = replay_batch["labels"][old_task] != IGNORE_INDEX
                    replay_weights = confidence_weights(teacher_replay_out["logits"], replay_mask)
                    replay_loss = replay_loss + float(continual_cfg.get("lambda_kd", 1.0)) * masked_kd_loss(
                        replay_output["logits"],
                        teacher_replay_out["logits"],
                        mask=replay_mask,
                        temperature=float(continual_cfg.get("temperature", 2.0)),
                        weights=replay_weights,
                    )
                    replay_loss = replay_loss + float(continual_cfg.get("lambda_rel", 1.0)) * sample_relation_loss(
                        replay_output["embedding"],
                        teacher_replay_out["embedding"],
                        mask=replay_mask,
                        weights=replay_weights,
                    )
                replay_loss = replay_loss * supervised_weight / max(grad_accum_steps, 1)
            scaler.scale(replay_loss).backward()
            total_loss_value += float(replay_loss.detach().cpu()) * max(grad_accum_steps, 1)
        if step % max(grad_accum_steps, 1) == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            if method == "dlg_packnet":
                _apply_packnet_gradient_mask(model, packnet_state)
            grad_clip = float(train_cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            si_before_step = _capture_si_step(model, si_state) if method == "dlg_si" else None
            scaler.step(optimizer)
            scaler.update()
            if method == "dlg_packnet":
                _restore_packnet_frozen_weights(model, packnet_state)
            if method == "dlg_si":
                _accumulate_si_step(model, si_state, si_before_step)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += total_loss_value
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def _evaluate(model, loaders_by_task, learned_tasks, device, method, stage) -> list[dict[str, object]]:
    model.eval()
    rows = []
    for task in learned_tasks:
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loaders_by_task[task]:
            batch = _move_batch(batch, device)
            output = model(batch, task)
            labels = batch["labels"][task]
            mask = labels != IGNORE_INDEX
            y_true.extend(labels[mask].detach().cpu().tolist())
            y_pred.extend(output["logits"].argmax(dim=-1)[mask].detach().cpu().tolist())
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            TASK_NUM_LABELS[task],
            positive_label=1 if task == "shift" else None,
        )
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
                "forgetting": "",
                "retention": "",
            }
        )
    return rows


@torch.no_grad()
def _evaluate_icarl_nearest_mean(
    model,
    loaders_by_task,
    learned_tasks,
    device,
    method,
    stage,
    replay_memory: dict[str, list[DialogueExample]],
    tokenizer,
    max_length: int,
    batch_size: int,
) -> list[dict[str, object]]:
    model.eval()
    class_means_by_task = _build_icarl_class_means(
        model,
        replay_memory,
        learned_tasks,
        tokenizer,
        max_length,
        batch_size,
        device,
    )
    rows = []
    for task in learned_tasks:
        class_means = class_means_by_task.get(task)
        if class_means is None or class_means.numel() == 0:
            rows.extend(_evaluate(model, loaders_by_task, [task], device, method, stage))
            continue
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loaders_by_task[task]:
            batch = _move_batch(batch, device)
            output = model(batch, task)
            labels = batch["labels"][task]
            mask = labels != IGNORE_INDEX
            if not torch.any(mask):
                continue
            embeddings = torch.nn.functional.normalize(output["embedding"][mask], dim=-1)
            distances = torch.cdist(embeddings.float(), class_means.float())
            predictions = distances.argmin(dim=-1)
            y_true.extend(labels[mask].detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            TASK_NUM_LABELS[task],
            positive_label=1 if task == "shift" else None,
        )
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
                "forgetting": "",
                "retention": "",
            }
        )
    return rows


@torch.no_grad()
def _build_icarl_class_means(
    model,
    replay_memory: dict[str, list[DialogueExample]],
    learned_tasks,
    tokenizer,
    max_length: int,
    batch_size: int,
    device,
) -> dict[str, torch.Tensor]:
    means_by_task: dict[str, torch.Tensor] = {}
    was_training = model.training
    model.eval()
    for task in learned_tasks:
        examples = replay_memory.get(task, [])
        if not examples:
            continue
        embedding_dim = int(model.heads[task].in_features)
        loader = _build_loader(examples, tokenizer, max_length, batch_size, shuffle=False, sampler_task=task)
        embeddings_by_label: dict[int, list[torch.Tensor]] = {}
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch, task)
            labels = batch["labels"][task]
            mask = labels != IGNORE_INDEX
            if not torch.any(mask):
                continue
            normalized = torch.nn.functional.normalize(output["embedding"][mask], dim=-1)
            for label in range(TASK_NUM_LABELS[task]):
                label_mask = labels[mask] == label
                if torch.any(label_mask):
                    embeddings_by_label.setdefault(label, []).append(normalized[label_mask].detach())
        class_means = []
        for label in range(TASK_NUM_LABELS[task]):
            chunks = embeddings_by_label.get(label)
            if chunks:
                mean = torch.cat(chunks, dim=0).mean(dim=0)
                class_means.append(torch.nn.functional.normalize(mean, dim=0))
            else:
                class_means.append(torch.zeros(embedding_dim, device=device))
        means_by_task[task] = torch.stack(class_means, dim=0)
    if was_training:
        model.train()
    return means_by_task


def _uses_random_replay(method: str) -> bool:
    return method in {
        "dlg_random_replay",
        "dlg_er",
        "dlg_icarl",
        "dlg_der",
        "dlg_derpp",
        "dlg_text_task_sa_cmd",
        "text_task_sa_cmd",
        "dlg_task_sa_cmd",
        "dlg_text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_replay_kd",
        "dlg_text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_freeze_old_heads",
        "dlg_text_task_sa_cmd_replay_kd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _uses_teacher(method: str) -> bool:
    return method in {
        "dlg_seq_kd",
        "dlg_icarl",
        "dlg_ours",
        "dlg_sa_cmd_no_replay",
        "dlg_task_sa_cmd",
        "dlg_text_task_sa_cmd",
        "text_task_sa_cmd",
        "dlg_text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_replay_kd",
        "dlg_text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_freeze_old_heads",
        "dlg_text_task_sa_cmd_replay_kd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
        "dlg_task_pg_trd",
    }


def _uses_parameter_regularizer(method: str) -> bool:
    return method in {"dlg_ewc", "dlg_mas", "dlg_si"}


def _uses_confidence_relation(method: str) -> bool:
    return method in {
        "dlg_sa_cmd_no_replay",
        "dlg_task_sa_cmd",
        "dlg_text_task_sa_cmd",
        "text_task_sa_cmd",
        "dlg_text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_replay_kd",
        "dlg_text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_freeze_old_heads",
        "dlg_text_task_sa_cmd_replay_kd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
        "dlg_task_pg_trd",
    }


def _uses_replay_batch_kd(method: str, continual_cfg: dict[str, Any]) -> bool:
    return bool(continual_cfg.get("replay_batch_kd", False)) or method in {
        "dlg_text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_replay_kd",
        "dlg_text_task_sa_cmd_replay_kd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _uses_freeze_old_heads(method: str, continual_cfg: dict[str, Any]) -> bool:
    return bool(continual_cfg.get("freeze_old_heads", False)) or method in {
        "dlg_text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_freeze_old_heads",
        "dlg_text_task_sa_cmd_replay_kd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


@torch.no_grad()
def _collect_replay_logits(
    model,
    examples: list[DialogueExample],
    task: str,
    tokenizer,
    max_length: int,
    batch_size: int,
    device,
) -> dict[int, torch.Tensor]:
    if not examples:
        return {}
    was_training = model.training
    model.eval()
    loader = _build_loader(examples, tokenizer, max_length, batch_size, shuffle=False, sampler_task=task)
    storage: dict[int, torch.Tensor] = {}
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch, task)
        for index, dialogue_id in enumerate(batch["dialogue_id"]):
            length = int(batch["lengths"][index].detach().cpu().item())
            storage[int(dialogue_id)] = output["logits"][index, :length].detach().cpu()
    if was_training:
        model.train()
    return storage


def _der_replay_loss(logits: torch.Tensor, batch: dict[str, Any], task: str, replay_logits: dict[str, dict[int, torch.Tensor]]) -> torch.Tensor:
    task_storage = replay_logits.get(task, {})
    losses = []
    labels = batch["labels"][task]
    for index, dialogue_id in enumerate(batch["dialogue_id"]):
        target = task_storage.get(int(dialogue_id))
        if target is None:
            continue
        length = min(int(target.shape[0]), int(logits.shape[1]))
        if length <= 0:
            continue
        mask = labels[index, :length] != IGNORE_INDEX
        if not torch.any(mask):
            continue
        target = target[:length].to(device=logits.device, dtype=logits.dtype)
        losses.append(nn.functional.mse_loss(logits[index, :length][mask], target[mask]))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _freeze_task_heads(model: XLMRDialogueTaskModel, learned_tasks: list[str]) -> None:
    for task_name, head in model.heads.items():
        requires_grad = task_name not in learned_tasks
        for parameter in head.parameters():
            parameter.requires_grad_(requires_grad)


def _regularized_named_parameters(model: nn.Module, continual_cfg: dict[str, Any]):
    scope = str(continual_cfg.get("regularizer_scope", "non_encoder"))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if scope == "non_encoder" and name.startswith("encoder."):
            continue
        yield name, parameter


def _init_regularization_state(model: nn.Module, continual_cfg: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "anchor": {},
        "importance": {
            name: torch.zeros_like(parameter.detach())
            for name, parameter in _regularized_named_parameters(model, continual_cfg)
        },
    }


def _parameter_regularization_loss(model: nn.Module, state: dict[str, dict[str, torch.Tensor]] | None) -> torch.Tensor:
    if not state or not state.get("anchor"):
        return next(model.parameters()).new_zeros(())
    total = next(model.parameters()).new_zeros(())
    count = 0
    anchors = state["anchor"]
    importance = state["importance"]
    for name, parameter in model.named_parameters():
        if name not in anchors or name not in importance:
            continue
        diff = parameter - anchors[name].to(device=parameter.device, dtype=parameter.dtype)
        weight = importance[name].to(device=parameter.device, dtype=parameter.dtype)
        total = total + (weight * diff.pow(2)).sum()
        count += parameter.numel()
    return total / max(count, 1)


def _update_gradient_importance(
    state: dict[str, dict[str, torch.Tensor]] | None,
    model: nn.Module,
    loader: DataLoader,
    task: str,
    criterion,
    device,
    method: str,
    train_cfg: dict[str, Any],
    continual_cfg: dict[str, Any],
) -> None:
    if state is None:
        return
    importance = {
        name: torch.zeros_like(parameter.detach())
        for name, parameter in _regularized_named_parameters(model, continual_cfg)
    }
    max_batches = int(continual_cfg.get("importance_max_batches", 50))
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    steps = 0
    for steps, batch in enumerate(loader, start=1):
        if max_batches > 0 and steps > max_batches:
            break
        batch = _move_batch(batch, device)
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
            output = model(batch, task)
            if method == "dlg_ewc":
                loss = _sequence_ce(criterion, output["logits"], batch["labels"][task])
            else:
                valid_mask = batch["labels"][task] != IGNORE_INDEX
                logits = output["logits"][valid_mask]
                loss = logits.pow(2).mean() if logits.numel() else output["logits"].new_zeros(())
        loss.backward()
        for name, parameter in _regularized_named_parameters(model, continual_cfg):
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            importance[name] = importance[name] + (grad.pow(2) if method == "dlg_ewc" else grad.abs())
        del output, loss, batch
        model.zero_grad(set_to_none=True)
    steps = max(steps, 1)
    for name, parameter in _regularized_named_parameters(model, continual_cfg):
        state["importance"][name] = state["importance"].get(name, torch.zeros_like(parameter.detach())) + importance[name] / steps
        state["anchor"][name] = parameter.detach().clone()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()


def _init_si_stage_state(model: nn.Module, continual_cfg: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "start": {name: parameter.detach().clone() for name, parameter in _regularized_named_parameters(model, continual_cfg)},
        "prev": {name: parameter.detach().clone() for name, parameter in _regularized_named_parameters(model, continual_cfg)},
        "omega": {name: torch.zeros_like(parameter.detach()) for name, parameter in _regularized_named_parameters(model, continual_cfg)},
    }


def _capture_si_step(model: nn.Module, si_state: dict[str, dict[str, torch.Tensor]] | None):
    if si_state is None:
        return None
    captured = {}
    for name, parameter in model.named_parameters():
        if name not in si_state["prev"] or parameter.grad is None:
            continue
        captured[name] = (parameter.detach().clone(), parameter.grad.detach().clone())
    return captured


def _accumulate_si_step(model: nn.Module, si_state: dict[str, dict[str, torch.Tensor]] | None, captured) -> None:
    if si_state is None or captured is None:
        return
    for name, parameter in model.named_parameters():
        if name not in captured:
            continue
        before, grad = captured[name]
        delta = parameter.detach() - before
        si_state["omega"][name] = si_state["omega"][name] + (-grad * delta)
        si_state["prev"][name] = parameter.detach().clone()


def _consolidate_si_importance(
    state: dict[str, dict[str, torch.Tensor]] | None,
    si_state: dict[str, dict[str, torch.Tensor]] | None,
    model: nn.Module,
    continual_cfg: dict[str, Any],
) -> None:
    if state is None or si_state is None:
        return
    xi = float(continual_cfg.get("si_xi", 0.1))
    for name, parameter in _regularized_named_parameters(model, continual_cfg):
        if name not in si_state["start"]:
            continue
        delta = parameter.detach() - si_state["start"][name]
        contribution = si_state["omega"][name] / (delta.pow(2) + xi)
        contribution = torch.clamp(contribution, min=0.0)
        state["importance"][name] = state["importance"].get(name, torch.zeros_like(parameter.detach())) + contribution
        state["anchor"][name] = parameter.detach().clone()


def _packnet_named_parameters(model: nn.Module, continual_cfg: dict[str, Any]):
    yield from _regularized_named_parameters(model, continual_cfg)


def _init_packnet_state(model: nn.Module, continual_cfg: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "frozen": {
            name: torch.zeros_like(parameter.detach(), dtype=torch.bool)
            for name, parameter in _packnet_named_parameters(model, continual_cfg)
        },
        "values": {},
    }


def _apply_packnet_gradient_mask(model: nn.Module, state: dict[str, dict[str, torch.Tensor]] | None) -> None:
    if not state:
        return
    frozen = state.get("frozen", {})
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name not in frozen:
            continue
        mask = frozen[name].to(device=parameter.device)
        parameter.grad.masked_fill_(mask, 0)


@torch.no_grad()
def _restore_packnet_frozen_weights(model: nn.Module, state: dict[str, dict[str, torch.Tensor]] | None) -> None:
    if not state:
        return
    frozen = state.get("frozen", {})
    values = state.get("values", {})
    for name, parameter in model.named_parameters():
        if name not in frozen or name not in values:
            continue
        mask = frozen[name].to(device=parameter.device)
        stored = values[name].to(device=parameter.device, dtype=parameter.dtype)
        parameter.data[mask] = stored[mask]


@torch.no_grad()
def _update_packnet_masks(
    state: dict[str, dict[str, torch.Tensor]] | None,
    model: nn.Module,
    continual_cfg: dict[str, Any],
) -> None:
    if state is None:
        return
    prune_ratio = float(continual_cfg.get("packnet_prune_ratio", 0.5))
    keep_ratio = max(0.0, min(1.0, 1.0 - prune_ratio))
    if keep_ratio <= 0:
        return
    frozen = state["frozen"]
    values = state.setdefault("values", {})
    for name, parameter in _packnet_named_parameters(model, continual_cfg):
        if parameter.ndim <= 1:
            continue
        current_frozen = frozen.get(name)
        if current_frozen is None:
            current_frozen = torch.zeros_like(parameter.detach(), dtype=torch.bool)
        current_frozen = current_frozen.to(device=parameter.device)
        free_mask = ~current_frozen
        free_values = parameter.detach().abs()[free_mask]
        if free_values.numel() == 0:
            frozen[name] = current_frozen.detach().cpu()
            continue
        keep_count = max(1, int(round(float(free_values.numel()) * keep_ratio)))
        threshold = torch.topk(free_values.flatten(), k=keep_count, largest=True).values.min()
        newly_frozen = free_mask & (parameter.detach().abs() >= threshold)
        updated = current_frozen | newly_frozen
        frozen[name] = updated.detach().cpu()
        stored = values.get(name)
        if stored is None:
            stored = torch.zeros_like(parameter.detach().cpu())
        stored = stored.to(device=parameter.device, dtype=parameter.dtype)
        stored[updated] = parameter.detach()[updated]
        values[name] = stored.detach().cpu()


@torch.no_grad()
def _select_replay_dialogues(
    examples: list[DialogueExample],
    task_name: str,
    memory_per_class: int,
    replay_strategy: str,
    representative_ratio: float,
    klmap_dim: int,
    seed: int,
    model,
    loader,
    device,
) -> list[DialogueExample]:
    rng = random.Random(seed)
    by_label: dict[int, list[DialogueExample]] = {}
    for example in examples:
        labels = getattr(example, f"{task_name}_labels")
        valid_labels = [int(label) for label in labels if int(label) != IGNORE_INDEX]
        if not valid_labels:
            continue
        by_label.setdefault(valid_labels[0], []).append(example)

    if replay_strategy == "random":
        selected: list[DialogueExample] = []
        for label in sorted(by_label):
            label_examples = list(by_label[label])
            rng.shuffle(label_examples)
            selected.extend(label_examples[:memory_per_class])
        return selected

    was_training = model.training
    model.eval()
    embeddings: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[int] = []
    ordered_dialogue_ids: list[int] = []
    seen_dialogue_ids: set[int] = set()
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch, task_name)
        task_labels = batch["labels"][task_name]
        valid_mask = task_labels != IGNORE_INDEX
        for idx, dialogue_id in enumerate(batch["dialogue_id"]):
            dialogue_id = int(dialogue_id)
            if dialogue_id in seen_dialogue_ids:
                continue
            row_mask = valid_mask[idx]
            if not torch.any(row_mask):
                continue
            seen_dialogue_ids.add(dialogue_id)
            labels.append(int(task_labels[idx][row_mask][0].detach().cpu().item()))
            embeddings.append(output["embedding"][idx][row_mask].mean(dim=0).detach().cpu())
            logits.append(output["logits"][idx][row_mask].mean(dim=0).detach().cpu())
            ordered_dialogue_ids.append(dialogue_id)
    if was_training:
        model.train()
    if not embeddings:
        return []

    by_id = {example.dialogue_id: example for example in examples}
    all_embeddings = torch.stack(embeddings, dim=0)
    all_logits = torch.stack(logits, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    selection_embeddings = (
        _fit_klmap_selection_embeddings(all_embeddings, label_tensor, all_logits, klmap_dim, seed)
        if _uses_klmap_replay_strategy(replay_strategy)
        else all_embeddings
    )
    selected: list[DialogueExample] = []
    for label in sorted(set(labels)):
        indices = torch.nonzero(label_tensor == int(label), as_tuple=False).reshape(-1).tolist()
        if len(indices) <= memory_per_class:
            selected.extend(by_id[ordered_dialogue_ids[index]] for index in indices)
            continue
        label_embeddings = selection_embeddings[indices]
        local_indices = _select_replay_indices(label_embeddings, memory_per_class, replay_strategy, representative_ratio)
        selected.extend(by_id[ordered_dialogue_ids[indices[local_idx]]] for local_idx in local_indices)
    return selected


def _select_replay_indices(label_embeddings: torch.Tensor, count: int, replay_strategy: str, representative_ratio: float) -> list[int]:
    replay_strategy = _base_replay_strategy(replay_strategy)
    if replay_strategy == "prototype_nearest":
        return _select_nearest_indices(label_embeddings, count)
    if replay_strategy == "diverse":
        return _select_diverse_indices(label_embeddings, count, excluded=set())
    if replay_strategy == "hybrid":
        return _select_hybrid_indices(label_embeddings, count, representative_ratio)
    raise ValueError(f"Unknown replay_strategy: {replay_strategy}")


def _uses_klmap_replay_strategy(replay_strategy: str) -> bool:
    return replay_strategy.endswith("_klmap")


def _base_replay_strategy(replay_strategy: str) -> str:
    if replay_strategy.endswith("_klmap"):
        return replay_strategy[: -len("_klmap")]
    return replay_strategy


def _fit_klmap_selection_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    klmap_dim: int,
    seed: int,
) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    input_dim = int(embeddings.shape[1])
    reduced_dim = min(max(1, int(klmap_dim)), input_dim)
    num_classes = int(teacher_logits.shape[1])
    hidden_dim = min(256, max(reduced_dim * 2, input_dim // 2))
    mapper = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, reduced_dim),
    )
    classifier = nn.Linear(reduced_dim, num_classes)
    optimizer = torch.optim.AdamW([*mapper.parameters(), *classifier.parameters()], lr=1e-3, weight_decay=1e-4)
    batch_size = min(64, int(embeddings.shape[0]))
    temperature = 2.0
    lambda_kl = 1.0
    x = embeddings.float()
    y = labels.long()
    teacher_probs = torch.softmax(teacher_logits.float() / temperature, dim=-1)
    mapper.train()
    classifier.train()
    with torch.enable_grad():
        for _ in range(50):
            permutation = torch.randperm(int(x.shape[0]), generator=generator)
            for start in range(0, int(x.shape[0]), batch_size):
                index = permutation[start : start + batch_size]
                projected = mapper(x.index_select(0, index))
                student_logits = classifier(projected)
                ce_loss = nn.functional.cross_entropy(student_logits, y.index_select(0, index))
                kl_loss = nn.functional.kl_div(
                    torch.log_softmax(student_logits / temperature, dim=-1),
                    teacher_probs.index_select(0, index),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = ce_loss + lambda_kl * kl_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    mapper.eval()
    with torch.no_grad():
        return mapper(x).detach()


def _select_nearest_indices(label_embeddings: torch.Tensor, count: int) -> list[int]:
    prototype = label_embeddings.mean(dim=0, keepdim=True)
    distances = torch.cdist(label_embeddings, prototype).squeeze(1)
    return distances.argsort().tolist()[:count]


def _select_hybrid_indices(label_embeddings: torch.Tensor, count: int, representative_ratio: float) -> list[int]:
    representative_count = max(0, min(count, int(round(count * representative_ratio))))
    diverse_count = count - representative_count
    selected = []
    if representative_count > 0:
        selected.extend(_select_nearest_indices(label_embeddings, representative_count))
    if diverse_count > 0:
        selected.extend(_select_diverse_indices(label_embeddings, diverse_count, set(selected)))
    return _dedupe(selected)[:count]


def _select_diverse_indices(label_embeddings: torch.Tensor, count: int, excluded: set[int]) -> list[int]:
    candidates = [idx for idx in range(int(label_embeddings.shape[0])) if idx not in excluded]
    if len(candidates) <= count:
        return candidates
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    candidate_embeddings = label_embeddings.index_select(0, candidate_tensor)
    prototype = label_embeddings.mean(dim=0, keepdim=True)
    first = int(torch.cdist(candidate_embeddings, prototype).squeeze(1).argmax().item())
    chosen = [first]
    min_distances = torch.cdist(candidate_embeddings, candidate_embeddings[first : first + 1]).squeeze(1)
    while len(chosen) < count:
        next_idx = int(min_distances.argmax().item())
        if next_idx in chosen:
            break
        chosen.append(next_idx)
        next_distances = torch.cdist(candidate_embeddings, candidate_embeddings[next_idx : next_idx + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, next_distances)
    return [candidates[idx] for idx in chosen]


def _dedupe(indices: list[int]) -> list[int]:
    seen = set()
    result = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        result.append(index)
    return result


def _replay_dialogue_digest(examples: list[DialogueExample]) -> str:
    dialogue_ids = sorted(str(example.dialogue_id) for example in examples)
    payload = "\n".join(dialogue_ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _write_replay_selection(
    output_dir: Path,
    method: str,
    stage_index: int,
    task: str,
    replay_strategy: str,
    examples: list[DialogueExample],
) -> None:
    rows = [
        {
            "dialogue_id": example.dialogue_id,
            "length": example.length,
        }
        for example in sorted(examples, key=lambda item: int(item.dialogue_id))
    ]
    payload = {
        "method": method,
        "stage_index": stage_index,
        "task": task,
        "replay_strategy": replay_strategy,
        "num_dialogues": len(examples),
        "digest": _replay_dialogue_digest(examples),
        "dialogues": rows,
    }
    path = ensure_dir(output_dir / "replay_selections") / f"{method}_stage{stage_index}_{task}_{replay_strategy}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def _update_prototype_bank(prototype_bank, method, task, loader, model, device) -> None:
    if prototype_bank is None or method != "dlg_task_pg_trd":
        return
    was_training = model.training
    model.eval()
    embeddings = []
    labels = []
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch, task)
        task_labels = batch["labels"][task]
        mask = task_labels != IGNORE_INDEX
        if torch.any(mask):
            embeddings.append(output["embedding"][mask].detach().cpu())
            labels.append(task_labels[mask].detach().cpu())
    if was_training:
        model.train()
    if embeddings:
        prototype_bank.update_from_embeddings(task, torch.cat(embeddings, dim=0), torch.cat(labels, dim=0))


def _build_loader(examples, tokenizer, max_length, batch_size, shuffle, sampler_task) -> DataLoader:
    dataset = DialogueTextDataset(examples, tokenizer, max_length)
    sampler = _build_weighted_sampler(examples, sampler_task) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        collate_fn=_collate_dialogue_text_batch,
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


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _collate_dialogue_text_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["length"].item()) for item in items)
    token_len = int(items[0]["input_ids"].shape[-1])
    batch_size = len(items)
    pad_token_id = int(items[0].get("pad_token_id", 0))
    input_ids = torch.full((batch_size, max_len, token_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, token_len, dtype=torch.long)
    labels = {
        task: torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        for task in items[0]["labels"]
    }
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.stack([item["length"] for item in items])
    for idx, item in enumerate(items):
        length = int(item["length"].item())
        input_ids[idx, :length] = item["input_ids"]
        attention_mask[idx, :length] = item["attention_mask"]
        sequence_mask[idx, :length] = True
        for task in labels:
            labels[task][idx, :length] = item["labels"][task]
    padded_positions = attention_mask[~sequence_mask]
    if padded_positions.numel() > 0:
        padded_positions[:, 0] = 1
        attention_mask[~sequence_mask] = padded_positions
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,
        "sequence_mask": sequence_mask,
        "dialogue_id": [item["dialogue_id"] for item in items],
        "texts": [item["texts"] for item in items],
    }


def _sequence_ce(criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _backward_step(loss, model, optimizer, scheduler, scaler, train_cfg, step, grad_accum_steps, is_last=False) -> None:
    scaler.scale(loss).backward()
    if step % max(grad_accum_steps, 1) == 0 or is_last:
        scaler.unscale_(optimizer)
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


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
        "final_avg_weighted_f1",
        "final_avg_accuracy",
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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
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


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
