from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .meld_csv import EMOTION_LABELS, SENTIMENT_LABELS, SHIFT_LABELS, TASK_LABELS, UtteranceRecord


@dataclass(frozen=True)
class TaskExample:
    task_name: str
    split: str
    utterance_key: str
    text: str
    speaker: str
    label: int
    label_name: str
    dialogue_id: int
    utterance_id: int
    video_path: str
    context_text: str
    meta: dict[str, Any]


def build_task_examples(
    records: list[UtteranceRecord],
    task_name: str,
    context_window: int = 3,
) -> list[TaskExample]:
    if task_name == "sentiment":
        return _build_classification_examples(records, task_name, "sentiment", SENTIMENT_LABELS, context_window)
    if task_name == "emotion":
        return _build_classification_examples(records, task_name, "emotion", EMOTION_LABELS, context_window)
    if task_name == "shift":
        return _build_shift_examples(records, context_window)
    raise ValueError(f"Unknown task '{task_name}'. Expected one of {sorted(TASK_LABELS)}")


def build_all_tasks(
    split_records: dict[str, list[UtteranceRecord]],
    task_order: list[str],
    context_window: int = 3,
) -> dict[str, dict[str, list[TaskExample]]]:
    return {
        split: {
            task_name: build_task_examples(records, task_name, context_window=context_window)
            for task_name in task_order
        }
        for split, records in split_records.items()
    }


def _build_classification_examples(
    records: list[UtteranceRecord],
    task_name: str,
    label_attr: str,
    label_map: dict[str, int],
    context_window: int,
) -> list[TaskExample]:
    context_map = _context_text_by_key(records, context_window)
    examples: list[TaskExample] = []
    for record in records:
        label_name = getattr(record, label_attr)
        examples.append(
            TaskExample(
                task_name=task_name,
                split=record.split,
                utterance_key=record.utterance_key,
                text=record.utterance,
                speaker=record.speaker,
                label=label_map[label_name],
                label_name=label_name,
                dialogue_id=record.dialogue_id,
                utterance_id=record.utterance_id,
                video_path=str(record.video_path),
                context_text=context_map[record.utterance_key],
                meta={"emotion": record.emotion, "sentiment": record.sentiment},
            )
        )
    return examples


def _build_shift_examples(records: list[UtteranceRecord], context_window: int) -> list[TaskExample]:
    context_map = _context_text_by_key(records, context_window)
    by_dialogue: dict[int, list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        by_dialogue[record.dialogue_id].append(record)

    examples: list[TaskExample] = []
    for dialogue_id in sorted(by_dialogue):
        last_by_speaker: dict[str, UtteranceRecord] = {}
        utterances = sorted(by_dialogue[dialogue_id], key=lambda item: item.utterance_id)
        for record in utterances:
            previous = last_by_speaker.get(record.speaker)
            if previous is not None:
                shifted = int(previous.emotion != record.emotion)
                label_name = "shift" if shifted else "no_shift"
                examples.append(
                    TaskExample(
                        task_name="shift",
                        split=record.split,
                        utterance_key=record.utterance_key,
                        text=record.utterance,
                        speaker=record.speaker,
                        label=SHIFT_LABELS[label_name],
                        label_name=label_name,
                        dialogue_id=record.dialogue_id,
                        utterance_id=record.utterance_id,
                        video_path=str(record.video_path),
                        context_text=context_map[record.utterance_key],
                        meta={
                            "emotion": record.emotion,
                            "previous_same_speaker_emotion": previous.emotion,
                            "previous_same_speaker_key": previous.utterance_key,
                        },
                    )
                )
            last_by_speaker[record.speaker] = record
    return examples


def _context_text_by_key(records: list[UtteranceRecord], context_window: int) -> dict[str, str]:
    if context_window <= 0:
        return {record.utterance_key: record.utterance for record in records}

    by_dialogue: dict[int, list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        by_dialogue[record.dialogue_id].append(record)

    context: dict[str, str] = {}
    for dialogue_id in sorted(by_dialogue):
        utterances = sorted(by_dialogue[dialogue_id], key=lambda item: item.utterance_id)
        for idx, record in enumerate(utterances):
            start = max(0, idx - context_window + 1)
            window = utterances[start : idx + 1]
            context[record.utterance_key] = " [SEP] ".join(item.utterance for item in window)
    return context

