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
