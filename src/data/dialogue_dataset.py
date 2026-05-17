from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.meld_csv import EMOTION_LABELS, SENTIMENT_LABELS, SHIFT_LABELS, UtteranceRecord
from src.data.multimodal_dataset import DEFAULT_MODALITIES, _load_feature, feature_path_for


IGNORE_INDEX = -100


@dataclass(frozen=True)
class DialogueExample:
    split: str
    dialogue_id: int
    utterance_keys: list[str]
    texts: list[str]
    speakers: list[str]
    sentiment_labels: list[int]
    emotion_labels: list[int]
    shift_labels: list[int]
    shift_mask: list[int]

    @property
    def length(self) -> int:
        return len(self.utterance_keys)


def build_dialogue_examples(records: list[UtteranceRecord]) -> list[DialogueExample]:
    by_dialogue: dict[int, list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        by_dialogue[record.dialogue_id].append(record)

    examples: list[DialogueExample] = []
    for dialogue_id in sorted(by_dialogue):
        utterances = sorted(by_dialogue[dialogue_id], key=lambda item: item.utterance_id)
        last_emotion_by_speaker: dict[str, str] = {}
        shift_labels: list[int] = []
        shift_mask: list[int] = []
        for record in utterances:
            previous_emotion = last_emotion_by_speaker.get(record.speaker)
            if previous_emotion is None:
                shift_labels.append(IGNORE_INDEX)
                shift_mask.append(0)
            else:
                label_name = "shift" if previous_emotion != record.emotion else "no_shift"
                shift_labels.append(SHIFT_LABELS[label_name])
                shift_mask.append(1)
            last_emotion_by_speaker[record.speaker] = record.emotion

        examples.append(
            DialogueExample(
                split=utterances[0].split,
                dialogue_id=dialogue_id,
                utterance_keys=[record.utterance_key for record in utterances],
                texts=[record.utterance for record in utterances],
                speakers=[record.speaker for record in utterances],
                sentiment_labels=[SENTIMENT_LABELS[record.sentiment] for record in utterances],
                emotion_labels=[EMOTION_LABELS[record.emotion] for record in utterances],
                shift_labels=shift_labels,
                shift_mask=shift_mask,
            )
        )
    return examples


def filter_dialogues_by_modalities(
    examples: list[DialogueExample],
    feature_root: str | Path,
    required_modalities: list[str],
) -> tuple[list[DialogueExample], dict[str, int]]:
    kept: list[DialogueExample] = []
    missing_counts: Counter[str] = Counter()
    for example in examples:
        missing_for_dialogue: set[str] = set()
        for key in example.utterance_keys:
            for modality in required_modalities:
                if not feature_path_for(feature_root, example.split, modality, key).exists():
                    missing_for_dialogue.add(modality)
        if missing_for_dialogue:
            missing_counts.update(missing_for_dialogue)
            continue
        kept.append(example)
    return kept, dict(missing_counts)


class MeldDialogueFeatureDataset(Dataset):
    def __init__(
        self,
        examples: list[DialogueExample],
        feature_root: str | Path,
        feature_dims: dict[str, int],
        speaker_to_id: dict[str, int],
        active_modalities: list[str],
        all_modalities: list[str] | None = None,
    ) -> None:
        self.examples = examples
        self.feature_root = Path(feature_root)
        self.feature_dims = feature_dims
        self.speaker_to_id = speaker_to_id
        self.active_modalities = list(active_modalities)
        self.all_modalities = list(all_modalities or DEFAULT_MODALITIES)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        active = set(self.active_modalities)
        features: dict[str, torch.Tensor] = {}
        modality_mask: list[float] = []
        for modality in self.all_modalities:
            dim = int(self.feature_dims[modality])
            values: list[torch.Tensor] = []
            for key in example.utterance_keys:
                path = feature_path_for(self.feature_root, example.split, modality, key)
                if modality in active:
                    values.append(_load_feature(path, dim))
                else:
                    values.append(torch.zeros(dim, dtype=torch.float32))
            features[modality] = torch.stack(values)
            modality_mask.append(1.0 if modality in active else 0.0)

        return {
            "features": features,
            "modality_mask": torch.tensor(modality_mask, dtype=torch.float32),
            "speaker_id": torch.tensor(
                [self.speaker_to_id.get(speaker, 0) for speaker in example.speakers],
                dtype=torch.long,
            ),
            "labels": {
                "sentiment": torch.tensor(example.sentiment_labels, dtype=torch.long),
                "emotion": torch.tensor(example.emotion_labels, dtype=torch.long),
                "shift": torch.tensor(example.shift_labels, dtype=torch.long),
            },
            "length": torch.tensor(example.length, dtype=torch.long),
            "dialogue_id": example.dialogue_id,
            "utterance_keys": example.utterance_keys,
            "active_modalities": list(self.active_modalities),
            "texts": example.texts,
        }


def collate_dialogue_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty dialogue batch")
    modalities = list(items[0]["features"])
    tasks = list(items[0]["labels"])
    lengths = torch.stack([item["length"] for item in items])
    batch_size = len(items)
    max_len = int(lengths.max().item())

    features: dict[str, torch.Tensor] = {}
    for modality in modalities:
        dim = int(items[0]["features"][modality].shape[-1])
        tensor = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
        for idx, item in enumerate(items):
            length = int(item["length"].item())
            tensor[idx, :length] = item["features"][modality]
        features[modality] = tensor

    speaker_id = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = {
        task: torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        for task in tasks
    }
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        length = int(item["length"].item())
        speaker_id[idx, :length] = item["speaker_id"]
        sequence_mask[idx, :length] = True
        for task in tasks:
            labels[task][idx, :length] = item["labels"][task]

    return {
        "features": features,
        "modality_mask": torch.stack([item["modality_mask"] for item in items]),
        "speaker_id": speaker_id,
        "labels": labels,
        "lengths": lengths,
        "sequence_mask": sequence_mask,
        "dialogue_id": [item["dialogue_id"] for item in items],
        "utterance_keys": [item["utterance_keys"] for item in items],
        "active_modalities": items[0]["active_modalities"],
        "texts": [item["texts"] for item in items],
    }
