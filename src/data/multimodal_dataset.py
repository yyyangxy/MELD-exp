from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .task_builder import TaskExample


DEFAULT_MODALITIES = ["text", "audio", "visual"]


def feature_cache_root(feature_root: str | Path) -> Path:
    root = Path(feature_root)
    if root.name.startswith("features"):
        return root
    return root / "features"


def feature_path_for(
    feature_root: str | Path,
    split: str,
    modality: str,
    utterance_key: str,
) -> Path:
    return feature_cache_root(feature_root) / split / modality / f"{utterance_key}.npy"


def summarize_feature_coverage(
    examples: list[TaskExample],
    feature_root: str | Path,
    modalities: list[str],
) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for modality in modalities:
        coverage[modality] = sum(
            int(feature_path_for(feature_root, example.split, modality, example.utterance_key).exists())
            for example in examples
        )
    return coverage


def filter_examples_by_modalities(
    examples: list[TaskExample],
    feature_root: str | Path,
    required_modalities: list[str],
) -> tuple[list[TaskExample], dict[str, int]]:
    kept: list[TaskExample] = []
    missing_counts: Counter[str] = Counter()
    for example in examples:
        missing = [
            modality
            for modality in required_modalities
            if not feature_path_for(feature_root, example.split, modality, example.utterance_key).exists()
        ]
        if missing:
            missing_counts.update(missing)
            continue
        kept.append(example)
    return kept, dict(missing_counts)


class MeldMultimodalFeatureDataset(Dataset):
    def __init__(
        self,
        examples: list[TaskExample],
        feature_root: str | Path,
        feature_dims: dict[str, int],
        speaker_to_id: dict[str, int],
        active_modalities: list[str],
        all_modalities: list[str] | None = None,
        allow_missing: bool = False,
    ) -> None:
        self.examples = examples
        self.feature_root = Path(feature_root)
        self.feature_dims = feature_dims
        self.speaker_to_id = speaker_to_id
        self.active_modalities = list(active_modalities)
        self.all_modalities = list(all_modalities or DEFAULT_MODALITIES)
        self.allow_missing = allow_missing

        unknown = sorted(set(self.active_modalities) - set(self.all_modalities))
        if unknown:
            raise ValueError(f"Unknown active modalities: {unknown}")
        for modality in self.all_modalities:
            if modality not in self.feature_dims:
                raise ValueError(f"Missing feature dimension for modality '{modality}'")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        features: dict[str, torch.Tensor] = {}
        mask_values: list[float] = []
        active = set(self.active_modalities)

        for modality in self.all_modalities:
            dim = int(self.feature_dims[modality])
            path = feature_path_for(self.feature_root, example.split, modality, example.utterance_key)
            should_load = modality in active
            if should_load and path.exists():
                features[modality] = _load_feature(path, dim)
                mask_values.append(1.0)
            elif should_load and not self.allow_missing:
                raise FileNotFoundError(f"Missing {modality} feature for {example.utterance_key}: {path}")
            else:
                features[modality] = torch.zeros(dim, dtype=torch.float32)
                mask_values.append(0.0)

        return {
            "features": features,
            "modality_mask": torch.tensor(mask_values, dtype=torch.float32),
            "speaker_id": torch.tensor(self.speaker_to_id.get(example.speaker, 0), dtype=torch.long),
            "label": torch.tensor(example.label, dtype=torch.long),
            "task_name": example.task_name,
            "utterance_key": example.utterance_key,
            "active_modalities": list(self.active_modalities),
            "text": example.text,
        }


def collate_multimodal_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty multimodal batch")
    modalities = list(items[0]["features"])
    return {
        "features": {
            modality: torch.stack([item["features"][modality] for item in items])
            for modality in modalities
        },
        "modality_mask": torch.stack([item["modality_mask"] for item in items]),
        "speaker_id": torch.stack([item["speaker_id"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "task_name": items[0]["task_name"],
        "utterance_key": [item["utterance_key"] for item in items],
        "active_modalities": items[0]["active_modalities"],
        "text": [item["text"] for item in items],
    }


def _load_feature(path: Path, expected_dim: int) -> torch.Tensor:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to load multimodal .npy features") from exc

    array = np.load(path).astype("float32").reshape(-1)
    if int(array.shape[0]) != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for {path}: expected {expected_dim}, got {array.shape[0]}"
        )
    return torch.from_numpy(array)
