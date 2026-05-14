from __future__ import annotations

import random
from collections import defaultdict

from src.data.task_builder import TaskExample


class RandomReplayBuffer:
    def __init__(self, memory_per_class: int = 20, seed: int = 13) -> None:
        self.memory_per_class = memory_per_class
        self.rng = random.Random(seed)
        self._storage: dict[str, dict[int, list[TaskExample]]] = defaultdict(dict)

    def update(self, task_name: str, examples: list[TaskExample]) -> None:
        by_label: dict[int, list[TaskExample]] = defaultdict(list)
        for example in examples:
            by_label[example.label].append(example)

        task_storage: dict[int, list[TaskExample]] = {}
        for label, label_examples in by_label.items():
            shuffled = list(label_examples)
            self.rng.shuffle(shuffled)
            task_storage[label] = shuffled[: self.memory_per_class]
        self._storage[task_name] = task_storage

    def examples_for(self, task_name: str) -> list[TaskExample]:
        examples: list[TaskExample] = []
        for label_examples in self._storage.get(task_name, {}).values():
            examples.extend(label_examples)
        return examples

    def task_names(self) -> list[str]:
        return sorted(self._storage)

    def __len__(self) -> int:
        return sum(len(items) for task in self._storage.values() for items in task.values())

