from __future__ import annotations

import copy
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.datasets import build_speaker_vocab
from src.data.dialogue_dataset import IGNORE_INDEX, DialogueExample, build_dialogue_examples
from src.data.meld_csv import UtteranceRecord, read_all_splits
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.train.metrics import compute_classification_metrics
from src.utils.logging import setup_logging
from src.utils.paths import PROJECT_ROOT, ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)
E2E_DIALOGUE_MODALITY_METHODS = {
    "dlg_e2e_mod_seq_ft",
    "dlg_e2e_mod_seq_kd",
    "dlg_e2e_modality_sa_cmd",
    "dlg_e2e_modality_sa_cmd_view_heads",
    "dlg_e2e_modality_sa_cmd_view_heads_freeze",
}


class DialogueRawTextAudioDataset(Dataset):
    def __init__(
        self,
        examples: list[DialogueExample],
        records_by_key: dict[str, UtteranceRecord],
        tokenizer,
        speaker_to_id: dict[str, int],
        active_modalities: list[str],
        max_text_length: int,
        max_audio_seconds: float,
        audio_sample_rate: int,
        audio_cache_root: str | Path | None,
    ) -> None:
        self.examples = examples
        self.records_by_key = records_by_key
        self.tokenizer = tokenizer
        self.speaker_to_id = speaker_to_id
        self.active_modalities = list(active_modalities)
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
        use_audio = "audio" in set(self.active_modalities)
        waveforms = []
        zero_audio_samples = self.max_audio_samples if use_audio else 1
        for key in example.utterance_keys:
            record = self.records_by_key[key]
            if use_audio and record.video_exists:
                waveform = _load_audio(
                    record,
                    self.audio_sample_rate,
                    self.max_audio_samples,
                    self.audio_cache_root,
                )
            else:
                waveform = torch.zeros(zero_audio_samples, dtype=torch.float32)
            waveforms.append(waveform)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "audio": torch.stack(waveforms),
            "audio_enabled": torch.tensor(1.0 if use_audio else 0.0, dtype=torch.float32),
            "speaker_id": torch.tensor(
                [self.speaker_to_id.get(speaker, 0) for speaker in example.speakers],
                dtype=torch.long,
            ),
            "labels": {
                "emotion": torch.tensor(example.emotion_labels, dtype=torch.long),
            },
            "length": torch.tensor(example.length, dtype=torch.long),
            "dialogue_id": example.dialogue_id,
            "utterance_keys": example.utterance_keys,
            "texts": example.texts,
        }


class RawAudioEncoder(nn.Module):
    def __init__(self, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=160, padding=120),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        hidden = self.net(waveform.unsqueeze(1)).squeeze(-1)
        return self.proj(hidden)


class DialogueTextAudioE2EModel(nn.Module):
    def __init__(
        self,
        text_model_path: str,
        num_speakers: int,
        stage_order: list[str],
        method: str,
        text_dim: int = 1024,
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
        self.audio_encoder = RawAudioEncoder(audio_dim, dropout=dropout)
        # build_speaker_vocab assigns speaker ids from 1..N and reserves 0 for padding/unknown.
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
        self.use_view_heads = _uses_view_heads(method)
        heads = {"emotion": nn.Linear(head_dim, 7)}
        if self.use_view_heads:
            heads.update({_stage_head_name(stage): nn.Linear(head_dim, 7) for stage in stage_order})
        self.heads = nn.ModuleDict(heads)

    def forward(self, batch: dict[str, Any], active_modalities: list[str], head_name: str = "emotion") -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, max_len, token_len = input_ids.shape
        text_out = self.text_encoder(
            input_ids=input_ids.reshape(batch_size * max_len, token_len),
            attention_mask=attention_mask.reshape(batch_size * max_len, token_len),
        )
        text_emb = self.text_proj(text_out.last_hidden_state[:, 0]).reshape(batch_size, max_len, -1)

        active_modality_set = set(active_modalities)
        if "audio" in active_modality_set:
            audio = batch["audio"].reshape(batch_size * max_len, -1)
            audio_emb = self.audio_encoder(audio).reshape(batch_size, max_len, -1)
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
        logits = self.heads[head_name](sequence_emb)
        return {"logits": logits, "embedding": sequence_emb, "utterance_embedding": utterance_emb}


def run_dialogue_modality_e2e_experiment(
    config_path: str | Path,
    method: str,
    run_name: str | None = None,
    train_overrides: dict[str, Any] | None = None,
) -> Path:
    if method not in E2E_DIALOGUE_MODALITY_METHODS:
        raise ValueError(f"Unknown e2e dialogue modality method '{method}'. Expected {sorted(E2E_DIALOGUE_MODALITY_METHODS)}")

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = load_config(config_path)
    if train_overrides:
        _apply_train_overrides(config, train_overrides)
    config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "dialogue_modality_e2e_stl"
    if run_name:
        config.setdefault("run", {})["name"] = run_name
    output_dir = resolve_experiment_output_dir(config)
    setup_logging(output_dir / "logs" / f"{method}.log")
    _write_run_parameters(output_dir, config, method, train_overrides or {})
    seed_everything(int(config.get("seed", 13)))

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    modality_cfg = config.get("modalities", {})
    data_root = resolve_data_root(config)
    split_records = read_all_splits(data_root, warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)))
    dialogue_examples = {split: build_dialogue_examples(records) for split, records in split_records.items()}
    records_by_split_key = {
        split: {record.utterance_key: record for record in records}
        for split, records in split_records.items()
    }
    speaker_to_id = build_speaker_vocab(split_records["train"])

    stage_order = list(modality_cfg.get("e2e_stage_order", ["text", "text_audio"]))
    stages = _parse_e2e_stages(modality_cfg, stage_order)
    unsupported = sorted({modality for mods in stages.values() for modality in mods} - {"text", "audio"})
    if unsupported:
        raise ValueError(f"E2E dialogue modality runner currently supports text/audio only, got unsupported modalities: {unsupported}")

    text_model_path = str(config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large"))
    text_model_path = str(resolve_path(text_model_path, PROJECT_ROOT)) if not text_model_path.startswith("xlm-") else text_model_path
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    model = DialogueTextAudioE2EModel(
        text_model_path=text_model_path,
        num_speakers=len(speaker_to_id),
        stage_order=stage_order,
        method=method,
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
    total_steps = max(1, sum(len(dialogue_examples["train"]) for _ in stage_order) * epochs // max(grad_accum_steps, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda"))

    teacher_by_stage: dict[str, nn.Module] = {}
    learned_stages: list[str] = []
    rows: list[dict[str, object]] = []
    for stage_name in stage_order:
        if _uses_freeze_view_heads(method):
            _freeze_view_heads(model, learned_stages)
        loader = _build_loader(
            dialogue_examples["train"],
            records_by_split_key["train"],
            tokenizer,
            speaker_to_id,
            stages[stage_name],
            int(train_cfg.get("batch_size", 1)),
            shuffle=True,
            sampler=str(config.get("continual", {}).get("sampler", "")),
            max_text_length=int(model_cfg.get("max_length", 128)),
            max_audio_seconds=float(model_cfg.get("max_audio_seconds", 6.0)),
            audio_sample_rate=int(model_cfg.get("audio_sample_rate", 16000)),
            audio_cache_root=_resolve_audio_cache_root(model_cfg),
        )
        loss = _train_stage(
            model=model,
            teacher_by_stage=teacher_by_stage,
            learned_stages=learned_stages,
            stages=stages,
            stage_name=stage_name,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            method=method,
            train_cfg=train_cfg,
            continual_cfg=config.get("continual", {}),
            grad_accum_steps=grad_accum_steps,
        )
        LOGGER.info("Finished e2e dialogue modality stage=%s loss=%.4f", stage_name, loss)
        learned_stages.append(stage_name)
        if _uses_teacher(method):
            teacher_by_stage[stage_name] = _clone_frozen(model, device)
        _save_checkpoint(model, output_dir, method, stage_name)
        rows.extend(
            _evaluate_stage(
                model,
                dialogue_examples[str(data_cfg.get("eval_split", "test"))],
                records_by_split_key[str(data_cfg.get("eval_split", "test"))],
                tokenizer,
                speaker_to_id,
                learned_stages,
                stages,
                train_cfg,
                model_cfg,
                device,
                method,
                current_stage=stage_name,
            )
        )

    rows = _decorate_modality_metrics(rows, stage_order)
    result_path = output_dir / "results" / "dialogue_modality_e2e_results.csv"
    _write_rows(result_path, rows)
    LOGGER.info("Wrote e2e dialogue modality results to %s", result_path)
    return result_path


def _train_stage(
    model: nn.Module,
    teacher_by_stage: dict[str, nn.Module],
    learned_stages: list[str],
    stages: dict[str, list[str]],
    stage_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    method: str,
    train_cfg: dict,
    continual_cfg: dict,
    grad_accum_steps: int,
) -> float:
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(train_cfg.get("epochs", 30)) + 1):
        epoch_total = 0.0
        epoch_steps = 0
        model.train()
        for step, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("fp16", False) and device.type == "cuda")):
                output = model(batch, stages[stage_name], head_name=_head_name_for_method(method, stage_name))
                supervised_terms = [_sequence_ce(criterion, output["logits"], batch["labels"]["emotion"])]
                if _uses_sa_cmd(method):
                    for old_stage in learned_stages:
                        old_out = model(batch, stages[old_stage], head_name=_head_name_for_method(method, old_stage))
                        supervised_terms.append(_sequence_ce(criterion, old_out["logits"], batch["labels"]["emotion"]))
                loss = torch.stack(supervised_terms).mean()

                kd_terms = []
                rel_terms = []
                if _uses_teacher(method):
                    for old_stage in learned_stages:
                        teacher = teacher_by_stage.get(old_stage)
                        if teacher is None:
                            continue
                        with torch.no_grad():
                            teacher_out = teacher(batch, stages[old_stage], head_name=_head_name_for_method(method, old_stage))
                        student_out = model(batch, stages[old_stage], head_name=_head_name_for_method(method, old_stage))
                        mask = batch["labels"]["emotion"] != IGNORE_INDEX
                        weights = confidence_weights(teacher_out["logits"], mask) if _uses_sa_cmd(method) else None
                        kd_terms.append(
                            masked_kd_loss(
                                student_out["logits"],
                                teacher_out["logits"],
                                mask=mask,
                                temperature=float(continual_cfg.get("temperature", 2.0)),
                                weights=weights,
                            )
                        )
                        if _uses_sa_cmd(method):
                            rel_terms.append(sample_relation_loss(student_out["embedding"], teacher_out["embedding"], mask=mask, weights=weights))
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
            loss_float = float(loss.detach().cpu()) * max(grad_accum_steps, 1)
            total += loss_float
            epoch_total += loss_float
            steps += 1
            epoch_steps += 1
        LOGGER.info("Epoch %d/%d stage=%s method=%s loss=%.4f", epoch, int(train_cfg.get("epochs", 30)), stage_name, method, epoch_total / max(epoch_steps, 1))
    return total / max(steps, 1)


@torch.no_grad()
def _evaluate_stage(
    model: nn.Module,
    examples: list[DialogueExample],
    records_by_key: dict[str, UtteranceRecord],
    tokenizer,
    speaker_to_id: dict[str, int],
    learned_stages: list[str],
    stages: dict[str, list[str]],
    train_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    method: str,
    current_stage: str,
) -> list[dict[str, object]]:
    model.eval()
    rows = []
    for eval_stage in learned_stages:
        loader = _build_loader(
            examples,
            records_by_key,
            tokenizer,
            speaker_to_id,
            stages[eval_stage],
            int(train_cfg.get("batch_size", 1)),
            shuffle=False,
            sampler="",
            max_text_length=int(model_cfg.get("max_length", 128)),
            max_audio_seconds=float(model_cfg.get("max_audio_seconds", 6.0)),
            audio_sample_rate=int(model_cfg.get("audio_sample_rate", 16000)),
            audio_cache_root=_resolve_audio_cache_root(model_cfg),
        )
        y_true, y_pred = [], []
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch, stages[eval_stage], head_name=_head_name_for_method(method, eval_stage))
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
                "eval_modalities": "+".join(stages[eval_stage]),
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "final_avg": "",
                "modality_forgetting": "",
                "modality_retention": "",
                "num_eval_dialogues": len(examples),
                "num_eval_utterances": len(y_true),
            }
        )
    return rows


def _build_loader(
    examples: list[DialogueExample],
    records_by_key: dict[str, UtteranceRecord],
    tokenizer,
    speaker_to_id: dict[str, int],
    active_modalities: list[str],
    batch_size: int,
    shuffle: bool,
    sampler: str,
    max_text_length: int,
    max_audio_seconds: float,
    audio_sample_rate: int,
    audio_cache_root: Path | None,
) -> DataLoader:
    dataset = DialogueRawTextAudioDataset(
        examples,
        records_by_key=records_by_key,
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        active_modalities=active_modalities,
        max_text_length=max_text_length,
        max_audio_seconds=max_audio_seconds,
        audio_sample_rate=audio_sample_rate,
        audio_cache_root=audio_cache_root,
    )
    weighted_sampler = _build_weighted_sampler(examples) if shuffle and sampler == "weighted_random" else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if weighted_sampler is None else False,
        sampler=weighted_sampler,
        num_workers=0,
        collate_fn=_collate_raw_dialogue_batch,
    )


def _collate_raw_dialogue_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.stack([item["length"] for item in items])
    batch_size = len(items)
    max_len = int(lengths.max().item())
    token_len = int(items[0]["input_ids"].shape[-1])
    audio_len = int(items[0]["audio"].shape[-1])
    input_ids = torch.zeros(batch_size, max_len, token_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, token_len, dtype=torch.long)
    audio = torch.zeros(batch_size, max_len, audio_len, dtype=torch.float32)
    speaker_id = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = {"emotion": torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)}
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        length = int(item["length"].item())
        input_ids[idx, :length] = item["input_ids"]
        attention_mask[idx, :length] = item["attention_mask"]
        audio[idx, :length] = item["audio"]
        speaker_id[idx, :length] = item["speaker_id"]
        labels["emotion"][idx, :length] = item["labels"]["emotion"]
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
        "utterance_keys": [item["utterance_keys"] for item in items],
        "texts": [item["texts"] for item in items],
    }


def _load_audio(
    record: UtteranceRecord,
    sample_rate: int,
    max_samples: int,
    audio_cache_root: Path | None,
) -> torch.Tensor:
    cache_path = None
    if audio_cache_root is not None:
        cache_path = audio_cache_root / str(sample_rate) / record.split / f"{record.utterance_key}_{max_samples}.pt"
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu")
                if isinstance(cached, torch.Tensor) and cached.numel() == max_samples:
                    return cached.to(dtype=torch.float32).contiguous()
            except Exception:
                pass

    audio = _load_audio_from_video(record.video_path, sample_rate, max_samples)
    if cache_path is not None:
        try:
            ensure_dir(cache_path.parent)
            torch.save(audio.cpu(), cache_path)
        except Exception as exc:
            LOGGER.warning("Failed to write audio cache %s: %s", cache_path, exc)
    return audio


def _load_audio_from_video(path: Path, sample_rate: int, max_samples: int) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        raw = subprocess.check_output(command, timeout=10)
        audio = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.float32) / 32768.0
    except Exception:
        audio = torch.zeros(0, dtype=torch.float32)
    if audio.numel() >= max_samples:
        return audio[:max_samples].contiguous()
    padded = torch.zeros(max_samples, dtype=torch.float32)
    padded[: audio.numel()] = audio
    return padded


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


def _parse_e2e_stages(modality_cfg: dict, stage_order: list[str]) -> dict[str, list[str]]:
    configured = modality_cfg.get("e2e_stages")
    if not configured:
        configured = {
            "text": ["text"],
            "text_audio": ["text", "audio"],
        }
    return {stage: list(configured[stage]) for stage in stage_order}


def _uses_sa_cmd(method: str) -> bool:
    return method in {
        "dlg_e2e_modality_sa_cmd",
        "dlg_e2e_modality_sa_cmd_view_heads",
        "dlg_e2e_modality_sa_cmd_view_heads_freeze",
    }


def _uses_teacher(method: str) -> bool:
    return method in {
        "dlg_e2e_mod_seq_kd",
        "dlg_e2e_modality_sa_cmd",
        "dlg_e2e_modality_sa_cmd_view_heads",
        "dlg_e2e_modality_sa_cmd_view_heads_freeze",
    }


def _uses_view_heads(method: str) -> bool:
    return method in {
        "dlg_e2e_modality_sa_cmd_view_heads",
        "dlg_e2e_modality_sa_cmd_view_heads_freeze",
    }


def _uses_freeze_view_heads(method: str) -> bool:
    return method == "dlg_e2e_modality_sa_cmd_view_heads_freeze"


def _stage_head_name(stage_name: str) -> str:
    return f"emotion_{stage_name}"


def _head_name_for_method(method: str, stage_name: str) -> str:
    return _stage_head_name(stage_name) if _uses_view_heads(method) else "emotion"


def _freeze_view_heads(model: nn.Module, learned_stages: list[str]) -> None:
    learned_head_names = {_stage_head_name(stage_name) for stage_name in learned_stages}
    for head_name, head in model.heads.items():
        if not head_name.startswith("emotion_"):
            continue
        requires_grad = head_name not in learned_head_names
        for parameter in head.parameters():
            parameter.requires_grad_(requires_grad)


def _decorate_modality_metrics(rows: list[dict[str, object]], stage_order: list[str]) -> list[dict[str, object]]:
    final_stage = stage_order[-1]
    by_eval: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_eval.setdefault(str(row["eval_modalities"]), []).append(row)
    final_rows = [row for row in rows if row["stage"] == final_stage]
    final_avg = sum(float(row["weighted_f1"]) for row in final_rows) / len(final_rows) if final_rows else 0.0
    for row in rows:
        if row["stage"] != final_stage:
            continue
        history = by_eval.get(str(row["eval_modalities"]), [])
        best = max(float(item["weighted_f1"]) for item in history) if history else 0.0
        final = float(row["weighted_f1"])
        row["final_avg"] = final_avg
        row["modality_forgetting"] = max(0.0, best - final)
        row["modality_retention"] = final / best if best > 0 else 0.0
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
        "num_eval_dialogues",
        "num_eval_utterances",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _clone_frozen(model: nn.Module, device: torch.device) -> nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _save_checkpoint(model: nn.Module, output_dir: Path, method: str, suffix: str) -> None:
    torch.save(model.state_dict(), ensure_dir(output_dir / "checkpoints") / f"{method}_{suffix}.pt")


def _resolve_audio_cache_root(model_cfg: dict[str, Any]) -> Path | None:
    value = model_cfg.get("audio_cache_root", "outputs/audio_waveforms_16k_s4_e2e")
    if value in {None, ""}:
        return None
    return resolve_path(str(value), PROJECT_ROOT)


def _apply_train_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    train_cfg = config.setdefault("train", {})
    model_cfg = config.setdefault("model", {})
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "seed":
            config["seed"] = int(value)
        elif key in {"max_audio_seconds", "audio_sample_rate", "max_length", "audio_cache_root"}:
            model_cfg[key] = value
        else:
            train_cfg[key] = value


def _write_run_parameters(output_dir: Path, config: dict[str, Any], method: str, train_overrides: dict[str, Any]) -> None:
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    payload = {
        "method": method,
        "cli_train_overrides": {k: v for k, v in train_overrides.items() if v is not None},
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "effective_train": {
            "epochs": int(train_cfg.get("epochs", 30)),
            "batch_size": int(train_cfg.get("batch_size", 1)),
            "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
            "effective_batch_size": int(train_cfg.get("batch_size", 1)) * int(train_cfg.get("grad_accum_steps", 1)),
            "lr": float(train_cfg.get("lr", 2e-5)),
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "device": str(train_cfg.get("device", "auto")),
            "fp16": bool(train_cfg.get("fp16", False)),
            "max_audio_seconds": float(model_cfg.get("max_audio_seconds", 6.0)),
            "audio_sample_rate": int(model_cfg.get("audio_sample_rate", 16000)),
            "audio_cache_root": str(_resolve_audio_cache_root(model_cfg)) if _resolve_audio_cache_root(model_cfg) else "",
            "view_heads": _uses_view_heads(method),
            "freeze_old_view_heads": _uses_freeze_view_heads(method),
        },
    }
    path = ensure_dir(output_dir / "logs") / "run_parameters.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
