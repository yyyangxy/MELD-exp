from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypeVar

from src.data.dialogue_dataset import DialogueExample
from src.data.meld_csv import UtteranceRecord
from src.data.task_builder import TaskExample
from src.utils.paths import ensure_dir, resolve_path


LOGGER = logging.getLogger(__name__)
T = TypeVar("T", TaskExample, DialogueExample)


@dataclass(frozen=True)
class StlTaskSplit:
    root: Path
    task_order: list[str]
    ids: dict[str, dict[str, set[int]]]

    def ids_for(self, split: str, task_name: str) -> set[int]:
        return self.ids[split][task_name]


def resolve_stl_task_split_root(data_cfg: dict, data_root: Path) -> Path | None:
    value = data_cfg.get("stl_task_split_root")
    if value in {None, ""}:
        return None
    return resolve_path(value, data_root)


def load_stl_task_split(root: str | Path, task_order: list[str], splits: Iterable[str]) -> StlTaskSplit:
    split_root = Path(root)
    ids: dict[str, dict[str, set[int]]] = {}
    for split in splits:
        ids[split] = {}
        seen_by_task: dict[str, set[int]] = {}
        for task_index, task_name in enumerate(task_order, start=1):
            path = split_root / split / f"task{task_index}" / "dialogue_ids.txt"
            if not path.exists():
                raise FileNotFoundError(f"STL task split file not found for {split}/{task_name}: {path}")
            task_ids = _read_dialogue_ids(path)
            ids[split][task_name] = task_ids
            seen_by_task[task_name] = task_ids
        _validate_disjoint(split, seen_by_task)
    return StlTaskSplit(root=split_root, task_order=list(task_order), ids=ids)


def filter_task_examples_by_stl_split(
    task_examples: dict[str, dict[str, list[TaskExample]]],
    task_split: StlTaskSplit | None,
) -> dict[str, dict[str, list[TaskExample]]]:
    if task_split is None:
        return task_examples
    return {
        split: {
            task_name: _filter_examples(examples, task_split.ids_for(split, task_name))
            for task_name, examples in examples_by_task.items()
        }
        for split, examples_by_task in task_examples.items()
    }


def build_dialogue_examples_by_task_split(
    dialogue_examples: dict[str, list[DialogueExample]],
    task_order: list[str],
    task_split: StlTaskSplit | None,
) -> dict[str, dict[str, list[DialogueExample]]]:
    if task_split is None:
        return {
            split: {task_name: list(examples) for task_name in task_order}
            for split, examples in dialogue_examples.items()
        }
    return {
        split: {
            task_name: _filter_examples(examples, task_split.ids_for(split, task_name))
            for task_name in task_order
        }
        for split, examples in dialogue_examples.items()
    }


def log_stl_task_split_summary(
    task_examples: dict[str, dict[str, list[TaskExample]]] | None = None,
    dialogue_examples: dict[str, dict[str, list[DialogueExample]]] | None = None,
) -> None:
    source = task_examples if task_examples is not None else dialogue_examples
    if source is None:
        return
    for split, examples_by_task in source.items():
        summary = {
            task: {
                "dialogues": len({example.dialogue_id for example in examples}),
                "examples": len(examples),
            }
            for task, examples in examples_by_task.items()
        }
        LOGGER.info("STL task split summary for %s: %s", split, summary)


def write_stl_task_splits(
    split_records: dict[str, list[UtteranceRecord]],
    output_root: str | Path,
    task_order: list[str],
    seed: int = 13,
) -> Path:
    root = ensure_dir(output_root)
    metadata: dict[str, object] = {
        "seed": seed,
        "task_order": list(task_order),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_counts": {},
    }

    for split, records in split_records.items():
        ids_by_task = _partition_dialogues_for_tasks(records, task_order, seed)
        split_counts = {}
        for task_index, task_name in enumerate(task_order, start=1):
            dialogue_ids = ids_by_task[task_name]
            path = ensure_dir(root / split / f"task{task_index}") / "dialogue_ids.txt"
            path.write_text("".join(f"{dialogue_id}\n" for dialogue_id in dialogue_ids), encoding="utf-8")
            split_counts[task_name] = {
                "task_index": task_index,
                "dialogues": len(dialogue_ids),
                "dialogue_utterances": _count_utterances(records, dialogue_ids),
                "supervised_examples": _count_supervised_examples(records, dialogue_ids, task_name),
            }
        metadata["split_counts"][split] = split_counts

    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _partition_dialogues_for_tasks(
    records: list[UtteranceRecord],
    task_order: list[str],
    seed: int,
) -> dict[str, list[int]]:
    all_ids = sorted({record.dialogue_id for record in records})
    valid_ids_by_task = {
        task_name: set(_valid_dialogue_ids_for_task(records, task_name))
        for task_name in task_order
    }
    candidates = sorted(all_ids)
    random.Random(seed).shuffle(candidates)

    ids_by_task = {task_name: [] for task_name in task_order}
    for dialogue_id in candidates:
        eligible_tasks = [task for task in task_order if dialogue_id in valid_ids_by_task[task]]
        if not eligible_tasks:
            continue
        task_name = min(eligible_tasks, key=lambda task: (len(ids_by_task[task]), task_order.index(task)))
        ids_by_task[task_name].append(dialogue_id)

    for task_name in task_order:
        ids_by_task[task_name].sort()
        if not ids_by_task[task_name]:
            raise ValueError(f"No valid dialogues assigned to task '{task_name}'")
    _validate_disjoint("generated", {task: set(ids) for task, ids in ids_by_task.items()})
    return ids_by_task


def _valid_dialogue_ids_for_task(records: list[UtteranceRecord], task_name: str) -> list[int]:
    if task_name != "shift":
        return sorted({record.dialogue_id for record in records})

    by_dialogue: dict[int, list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        by_dialogue[record.dialogue_id].append(record)

    valid = []
    for dialogue_id, utterances in by_dialogue.items():
        last_by_speaker: set[str] = set()
        for record in sorted(utterances, key=lambda item: item.utterance_id):
            if record.speaker in last_by_speaker:
                valid.append(dialogue_id)
                break
            last_by_speaker.add(record.speaker)
    return sorted(valid)


def _read_dialogue_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            ids.add(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid dialogue id at {path}:{line_number}: {line!r}") from exc
    if not ids:
        raise ValueError(f"STL task split file is empty: {path}")
    return ids


def _validate_disjoint(split: str, ids_by_task: dict[str, set[int]]) -> None:
    tasks = list(ids_by_task)
    for left_index, left_task in enumerate(tasks):
        for right_task in tasks[left_index + 1 :]:
            overlap = ids_by_task[left_task] & ids_by_task[right_task]
            if overlap:
                sample = sorted(overlap)[:10]
                raise ValueError(
                    f"STL task split '{split}' is not disjoint: "
                    f"{left_task} and {right_task} overlap on dialogue ids {sample}"
                )


def _filter_examples(examples: list[T], dialogue_ids: set[int]) -> list[T]:
    return [example for example in examples if example.dialogue_id in dialogue_ids]


def _count_utterances(records: list[UtteranceRecord], dialogue_ids: list[int]) -> int:
    selected = set(dialogue_ids)
    return sum(1 for record in records if record.dialogue_id in selected)


def _count_supervised_examples(records: list[UtteranceRecord], dialogue_ids: list[int], task_name: str) -> int:
    selected = set(dialogue_ids)
    selected_records = [record for record in records if record.dialogue_id in selected]
    if task_name != "shift":
        return len(selected_records)

    count = 0
    by_dialogue: dict[int, list[UtteranceRecord]] = defaultdict(list)
    for record in selected_records:
        by_dialogue[record.dialogue_id].append(record)
    for utterances in by_dialogue.values():
        seen_speakers: set[str] = set()
        for record in sorted(utterances, key=lambda item: item.utterance_id):
            if record.speaker in seen_speakers:
                count += 1
            seen_speakers.add(record.speaker)
    return count
