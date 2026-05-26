from __future__ import annotations

import copy
import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.datasets import build_speaker_vocab
from src.data.dialogue_dataset import IGNORE_INDEX, DialogueExample, build_dialogue_examples
from src.data.meld_csv import UtteranceRecord, read_all_splits
from src.data.stl_task_splits import (
    build_dialogue_examples_by_task_split,
    load_stl_task_split,
    log_stl_task_split_summary,
    resolve_stl_task_split_root,
)
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.models.stl_model import TASK_NUM_LABELS
from src.train.dialogue_modality_e2e_runner import RawAudioEncoder, _load_audio
from src.train.metrics import compute_classification_metrics, decorate_final_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
S5_E2E_TEXT_AUDIO_TASK_METHODS = {
    "s5_e2e_ta_seq_ft",
    "s5_e2e_ta_seq_kd",
    "s5_e2e_ta_sa_cmd",
}


class DialogueTextAudioTaskDataset(Dataset):
    def __init__(
        self,
        examples: list[DialogueExample],
        records_by_key: dict[str, UtteranceRecord],
        tokenizer,
        speaker_to_id: dict[str, int],
        max_text_length: int,
        max_audio_seconds: float,
        audio_sample_rate: int,
        audio_cache_root: str | Path | None,
    ) -> None:
        self.examples = examples
        self.records_by_key = records_by_key
        self.tokenizer = tokenizer
        self.speaker_to_id = speaker_to_id
        self.max_text_length = max_text_length
        self.max_audio_samples = int(round(max_audio_seconds * audio_sample_rate))
        self.audio_sample_rate = audio_sample_rate
        self.audio_cache_root = Path(audio_cache_root) if audio_cache_root else None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.texts,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        waveforms = []
        for key in example.utterance_keys:
            record = self.records_by_key[key]
            if record.video_exists:
                waveform = _load_audio(
                    record,
                    self.audio_sample_rate,
                    self.max_audio_samples,
                    self.audio_cache_root,
                )
            else:
                waveform = torch.zeros(self.max_audio_samples, dtype=torch.float32)
            waveforms.append(waveform)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "audio": torch.stack(waveforms),
            "audio_enabled": torch.tensor(1.0, dtype=torch.float32),
            "speaker_id": torch.tensor([self.speaker_to_id.get(s, 0) for s in example.speakers], dtype=torch.long),
            "labels": {
                "sentiment": torch.tensor(example.sentiment_labels, dtype=torch.long),
                "emotion": torch.tensor(example.emotion_labels, dtype=torch.long),
                "shift": torch.tensor(example.shift_labels, dtype=torch.long),
            },
            "length": torch.tensor(example.length, dtype=torch.long),
            "dialogue_id": example.dialogue_id,
        }


class DialogueTextAudioTaskE2EModel(nn.Module):
    def __init__(
        self,
        text_model_path: str,
        num_speakers: int,
        task_order: list[str],
        audio_encoder_type: str = "raw",
        audio_model_path: str | None = None,
        text_dim: int = 512,
        audio_dim: int = 256,
        fusion_dim: int = 256,
        speaker_dim: int = 32,
        dialogue_hidden_dim: int = 256,
        dialogue_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.text_encoder = AutoModel.from_pretrained(text_model_path, local_files_only=True)
        if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
            self.text_encoder.gradient_checkpointing_enable()
        hidden_size = int(self.text_encoder.config.hidden_size)
        self.text_proj = nn.Sequential(nn.Linear(hidden_size, text_dim), nn.GELU(), nn.Dropout(dropout))
        self.audio_dim = audio_dim
        self.audio_encoder_type = audio_encoder_type
        if audio_encoder_type == "raw":
            self.audio_encoder = RawAudioEncoder(audio_dim, dropout=dropout)
            self.audio_proj = nn.Identity()
        elif audio_encoder_type == "pretrained":
            if not audio_model_path:
                raise ValueError("audio_model_path is required when audio_encoder_type='pretrained'.")
            self.audio_encoder = AutoModel.from_pretrained(audio_model_path, local_files_only=True)
            if hasattr(self.audio_encoder, "gradient_checkpointing_enable"):
                self.audio_encoder.gradient_checkpointing_enable()
            audio_hidden_size = int(self.audio_encoder.config.hidden_size)
            self.audio_proj = nn.Sequential(nn.Linear(audio_hidden_size, audio_dim), nn.GELU(), nn.Dropout(dropout))
        else:
            raise ValueError(f"Unknown audio_encoder_type: {audio_encoder_type}. Expected raw or pretrained.")
        self.speaker_embedding = nn.Embedding(num_speakers + 1, speaker_dim, padding_idx=0)
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + audio_dim + speaker_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dialogue_encoder = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=dialogue_hidden_dim,
            num_layers=dialogue_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dialogue_num_layers > 1 else 0.0,
        )
        head_dim = dialogue_hidden_dim * 2
        self.heads = nn.ModuleDict({task: nn.Linear(head_dim, TASK_NUM_LABELS[task]) for task in task_order})

    def forward(self, batch: dict[str, Any], task_name: str) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, max_len, token_len = input_ids.shape
        text_out = self.text_encoder(
            input_ids=input_ids.reshape(batch_size * max_len, token_len),
            attention_mask=attention_mask.reshape(batch_size * max_len, token_len),
        )
        text_emb = self.text_proj(text_out.last_hidden_state[:, 0]).reshape(batch_size, max_len, -1)
        if torch.any(batch["audio_enabled"] > 0):
            audio = batch["audio"].reshape(batch_size * max_len, -1)
            if self.audio_encoder_type == "raw":
                audio_emb = self.audio_encoder(audio).reshape(batch_size, max_len, -1)
            else:
                audio_out = self.audio_encoder(input_values=audio)
                audio_emb = self.audio_proj(audio_out.last_hidden_state.mean(dim=1)).reshape(batch_size, max_len, -1)
            audio_emb = audio_emb * batch["audio_enabled"].view(batch_size, 1, 1)
        else:
            audio_emb = text_emb.new_zeros(batch_size, max_len, self.audio_dim)
        speaker_emb = self.speaker_embedding(batch["speaker_id"])
        utterance_emb = self.fusion(torch.cat([text_emb, audio_emb, speaker_emb], dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            utterance_emb,
            batch["lengths"].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.dialogue_encoder(packed)
        sequence_emb, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=max_len)
        return {"logits": self.heads[task_name](sequence_emb), "embedding": sequence_emb}

    def classify_embedding(self, embedding: torch.Tensor, task_name: str) -> dict[str, torch.Tensor]:
        return {"logits": self.heads[task_name](embedding), "embedding": embedding}


def run_s5_text_audio_task_e2e_experiment(
    config_path: str | Path,
    method: str,
    run_name: str | None = None,
    train_overrides: dict[str, Any] | None = None,
) -> Path:
    if method not in S5_E2E_TEXT_AUDIO_TASK_METHODS:
        raise ValueError(f"Unknown S5 e2e text/audio Task-STL method '{method}'.")

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = load_config(config_path)
    if train_overrides:
        _apply_overrides(config, train_overrides)
    config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "dialogue_text_audio_task_e2e_stl"
    if run_name:
        config.setdefault("run", {})["name"] = run_name
    output_dir = resolve_experiment_output_dir(config)
    setup_logging(output_dir / "logs" / f"{method}.log")
    _write_run_parameters(output_dir, config, method, train_overrides or {})
    seed_everything(int(config.get("seed", 13)))

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
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
        LOGGER.warning("No data.stl_task_split_root configured; S5 e2e uses full split data for every task.")
    log_stl_task_split_summary(dialogue_examples=dialogue_examples)

    records_by_split_key = {split: {r.utterance_key: r for r in records} for split, records in split_records.items()}
    speaker_to_id = build_speaker_vocab(split_records["train"])
    text_model_path = _resolve_text_model_path(str(config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large")))
    audio_model_path = _resolve_optional_model_path(config.get("feature_paths", {}).get("audio_model_path"))
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    model = DialogueTextAudioTaskE2EModel(
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
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    total_steps = max(1, sum(len(dialogue_examples["train"][task]) for task in task_order) * epochs // max(grad_accum_steps, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda"))

    teacher = None
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
            sampler=str(config.get("continual", {}).get("sampler", "")),
            sampler_task=task,
            model_cfg=model_cfg,
        )
        for epoch in range(1, epochs + 1):
            loss = _train_epoch(
                model,
                teacher,
                loader,
                learned_tasks,
                task,
                optimizer,
                scheduler,
                scaler,
                device,
                train_cfg,
                config.get("continual", {}),
                method,
                grad_accum_steps,
            )
            LOGGER.info("Epoch %d/%d task=%s method=%s loss=%.4f", epoch, epochs, task, method, loss)
        learned_tasks.append(task)
        if method in {"s5_e2e_ta_seq_kd", "s5_e2e_ta_sa_cmd"}:
            teacher = _clone_frozen(model, device)
        _save_checkpoint(model, output_dir, method, f"stage{stage_index}_{task}")
        rows.extend(
            _evaluate(
                model,
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
    result_path = output_dir / "results" / "dialogue_text_audio_task_e2e_results.csv"
    _write_rows(result_path, rows)
    LOGGER.info("Wrote S5 e2e text/audio Task-STL results to %s", result_path)
    return result_path


def _train_epoch(
    model,
    teacher,
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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
            output = model(batch, task)
            loss = _sequence_ce(criterion, output["logits"], batch["labels"][task])
            if teacher is not None and learned_tasks and method in {"s5_e2e_ta_seq_kd", "s5_e2e_ta_sa_cmd"}:
                kd_terms = []
                rel_terms = []
                with torch.no_grad():
                    teacher_base = teacher(batch, learned_tasks[0])
                for old_task in learned_tasks:
                    teacher_out = teacher.classify_embedding(teacher_base["embedding"], old_task)
                    student_out = model.classify_embedding(output["embedding"], old_task)
                    valid_mask = batch["labels"][old_task] != IGNORE_INDEX
                    weights = confidence_weights(teacher_out["logits"], valid_mask) if method == "s5_e2e_ta_sa_cmd" else None
                    kd_terms.append(
                        masked_kd_loss(
                            student_out["logits"],
                            teacher_out["logits"],
                            mask=valid_mask,
                            temperature=float(continual_cfg.get("temperature", 2.0)),
                            weights=weights,
                        )
                    )
                    if method == "s5_e2e_ta_sa_cmd":
                        rel_terms.append(sample_relation_loss(student_out["embedding"], teacher_out["embedding"], mask=valid_mask, weights=weights))
                if kd_terms:
                    loss = loss + float(continual_cfg.get("lambda_kd", 1.0)) * torch.stack(kd_terms).mean()
                if rel_terms:
                    loss = loss + float(continual_cfg.get("lambda_rel", 1.0)) * torch.stack(rel_terms).mean()
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
    return total / max(steps, 1)


@torch.no_grad()
def _evaluate(model, examples_by_task, records_by_key, tokenizer, speaker_to_id, learned_tasks, device, train_cfg, model_cfg, method, stage):
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


def _build_loader(examples, records_by_key, tokenizer, speaker_to_id, batch_size, shuffle, sampler, sampler_task, model_cfg) -> DataLoader:
    dataset = DialogueTextAudioTaskDataset(
        examples,
        records_by_key=records_by_key,
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        max_text_length=int(model_cfg.get("max_length", 64)),
        max_audio_seconds=float(model_cfg.get("max_audio_seconds", 4.0)),
        audio_sample_rate=int(model_cfg.get("audio_sample_rate", 16000)),
        audio_cache_root=_resolve_audio_cache_root(model_cfg),
    )
    weighted_sampler = _build_weighted_sampler(examples, sampler_task) if shuffle and sampler == "weighted_random" else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if weighted_sampler is None else False, sampler=weighted_sampler, num_workers=0, collate_fn=_collate)


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.stack([item["length"] for item in items])
    batch_size = len(items)
    max_len = int(lengths.max().item())
    token_len = int(items[0]["input_ids"].shape[-1])
    audio_len = int(items[0]["audio"].shape[-1])
    input_ids = torch.zeros(batch_size, max_len, token_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, token_len, dtype=torch.long)
    audio = torch.zeros(batch_size, max_len, audio_len, dtype=torch.float32)
    speaker_id = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = {task: torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long) for task in items[0]["labels"]}
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        length = int(item["length"].item())
        input_ids[idx, :length] = item["input_ids"]
        attention_mask[idx, :length] = item["attention_mask"]
        audio[idx, :length] = item["audio"]
        speaker_id[idx, :length] = item["speaker_id"]
        for task in labels:
            labels[task][idx, :length] = item["labels"][task]
        sequence_mask[idx, :length] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio": audio,
        "audio_enabled": torch.stack([item["audio_enabled"] for item in items]),
        "speaker_id": speaker_id,
        "labels": labels,
        "length": lengths,
        "lengths": lengths,
        "sequence_mask": sequence_mask,
        "dialogue_id": [item["dialogue_id"] for item in items],
    }


def _build_weighted_sampler(examples: list[DialogueExample], task_name: str) -> WeightedRandomSampler | None:
    if not examples:
        return None
    labels = []
    for example in examples:
        valid = [label for label in getattr(example, f"{task_name}_labels") if label != IGNORE_INDEX]
        labels.append(valid[0] if valid else 0)
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long)).float()
    counts[counts == 0] = 1.0
    weights = torch.tensor([1.0 / counts[label].item() for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _sequence_ce(criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {k: v.to(device) if hasattr(v, "to") else v for k, v in value.items()}
        else:
            moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _clone_frozen(model: nn.Module, device: torch.device) -> nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _save_checkpoint(model: nn.Module, output_dir: Path, method: str, suffix: str) -> None:
    torch.save(model.state_dict(), ensure_dir(output_dir / "checkpoints") / f"{method}_{suffix}.pt")


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
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "seed":
            config["seed"] = int(value)
        elif key == "text_model_path":
            feature_cfg["text_model_path"] = str(value)
        elif key in {"max_length", "max_audio_seconds", "audio_sample_rate", "audio_cache_root"}:
            model_cfg[key] = value
        else:
            train_cfg[key] = value


def _write_run_parameters(output_dir: Path, config: dict[str, Any], method: str, train_overrides: dict[str, Any]) -> None:
    payload = {
        "method": method,
        "cli_train_overrides": {k: v for k, v in train_overrides.items() if v is not None},
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    path = ensure_dir(output_dir / "logs") / "run_parameters.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_audio_cache_root(model_cfg: dict[str, Any]) -> Path | None:
    value = model_cfg.get("audio_cache_root", "outputs/audio_waveforms_16k_s5_task_e2e")
    if value in {None, ""}:
        return None
    return resolve_path(value, PROJECT_ROOT)


def _resolve_text_model_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith(".") or path.exists() or (PROJECT_ROOT / path).exists():
        return str(resolve_path(value, PROJECT_ROOT))
    return value


def _resolve_optional_model_path(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return _resolve_text_model_path(str(value))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
