from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .task_builder import TaskExample


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[^\w\s]", re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_size: int | None = 30000,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        tokens = [token for token, freq in counter.items() if freq >= min_freq]
        tokens.sort(key=lambda item: (-counter[item], item))
        if max_size is not None:
            tokens = tokens[: max(0, max_size - 2)]
        mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        mapping.update({token: idx + 2 for idx, token in enumerate(tokens)})
        return cls(mapping)

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        tokens = tokenize(text)[:max_length]
        ids = [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in tokens]
        mask = [1] * len(ids)
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids.extend([self.token_to_id[PAD_TOKEN]] * pad_len)
            mask.extend([0] * pad_len)
        return ids, mask

    def __len__(self) -> int:
        return len(self.token_to_id)


def build_speaker_vocab(records_or_examples: Iterable[object]) -> dict[str, int]:
    speakers = sorted({getattr(item, "speaker") for item in records_or_examples})
    return {speaker: idx + 1 for idx, speaker in enumerate(speakers)}


class MeldTextDataset(Dataset):
    def __init__(
        self,
        examples: list[TaskExample],
        vocabulary: Vocabulary,
        speaker_to_id: dict[str, int],
        max_length: int = 96,
        use_context: bool = True,
    ) -> None:
        self.examples = examples
        self.vocabulary = vocabulary
        self.speaker_to_id = speaker_to_id
        self.max_length = max_length
        self.use_context = use_context

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        text = example.context_text if self.use_context else example.text
        input_ids, attention_mask = self.vocabulary.encode(text, self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "speaker_id": torch.tensor(self.speaker_to_id.get(example.speaker, 0), dtype=torch.long),
            "label": torch.tensor(example.label, dtype=torch.long),
            "task_name": example.task_name,
            "utterance_key": example.utterance_key,
            "text": example.text,
        }

