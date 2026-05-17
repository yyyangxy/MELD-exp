from __future__ import annotations

import hashlib
from pathlib import Path

from src.data.meld_csv import UtteranceRecord
from src.features.feature_store import feature_path, save_feature


def hashing_text_feature(text: str, dim: int = 256) -> list[float]:
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def extract_hash_text_features(
    split_records: dict[str, list[UtteranceRecord]],
    output_root: str | Path,
    dim: int = 256,
    skip_existing: bool = True,
) -> int:
    count = 0
    for split, records in split_records.items():
        for record in records:
            if skip_existing and feature_path(output_root, split, "text", record.utterance_key).exists():
                count += 1
                continue
            feature = hashing_text_feature(record.utterance, dim=dim)
            save_feature(output_root, split, "text", record.utterance_key, feature)
            count += 1
    return count


def extract_xlmr_text_features(
    split_records: dict[str, list[UtteranceRecord]],
    output_root: str | Path,
    model_path: str | Path,
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 128,
    skip_existing: bool = True,
) -> int:
    """Extract frozen XLM-R utterance embeddings using the first-token representation."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "XLM-R text feature extraction requires torch and transformers. "
            "Install requirements.txt in the server environment."
        ) from exc

    resolved_device = _resolve_device(device, torch)
    model_name_or_path = str(model_path) or "xlm-roberta-large"
    model_ref = Path(model_name_or_path).expanduser() if _looks_like_path(model_name_or_path) else model_name_or_path
    local_only = isinstance(model_ref, Path) and model_ref.exists()

    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_only)
    model = AutoModel.from_pretrained(model_ref, local_files_only=local_only).to(resolved_device)
    model.eval()

    count = 0
    for split, records in split_records.items():
        pending: list[UtteranceRecord] = []
        for record in records:
            out_path = feature_path(output_root, split, "text", record.utterance_key)
            if skip_existing and out_path.exists():
                count += 1
                continue
            pending.append(record)
            if len(pending) >= batch_size:
                count += _flush_xlmr_batch(
                    pending,
                    output_root,
                    tokenizer,
                    model,
                    resolved_device,
                    max_length,
                    torch,
                )
                pending = []
        if pending:
            count += _flush_xlmr_batch(
                pending,
                output_root,
                tokenizer,
                model,
                resolved_device,
                max_length,
                torch,
            )
    return count


def _flush_xlmr_batch(
    records: list[UtteranceRecord],
    output_root: str | Path,
    tokenizer,
    model,
    device: str,
    max_length: int,
    torch_module,
) -> int:
    texts = [record.utterance for record in records]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch_module.no_grad():
        hidden = model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
    for record, feature in zip(records, hidden):
        save_feature(output_root, record.split, "text", record.utterance_key, feature)
    return len(records)


def _resolve_device(device: str, torch_module) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~"))
