from __future__ import annotations

from typing import Any

import torch


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    return {
        "input_ids": torch.stack([item["input_ids"] for item in items]),
        "attention_mask": torch.stack([item["attention_mask"] for item in items]),
        "speaker_id": torch.stack([item["speaker_id"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "task_name": items[0]["task_name"],
        "utterance_key": [item["utterance_key"] for item in items],
        "text": [item["text"] for item in items],
    }

