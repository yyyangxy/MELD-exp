from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_batch
from src.data.datasets import MeldTextDataset, Vocabulary
from src.data.task_builder import TaskExample


class PrototypeMemory:
    def __init__(
        self,
        memory_per_class: int = 20,
        batch_size: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.memory_per_class = memory_per_class
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._storage: dict[str, dict[int, list[TaskExample]]] = {}

    @torch.no_grad()
    def update(
        self,
        task_name: str,
        examples: list[TaskExample],
        model: torch.nn.Module,
        vocabulary: Vocabulary,
        speaker_to_id: dict[str, int],
        max_length: int,
        use_context: bool = True,
    ) -> None:
        if not examples:
            self._storage[task_name] = {}
            return

        dataset = MeldTextDataset(
            examples,
            vocabulary=vocabulary,
            speaker_to_id=speaker_to_id,
            max_length=max_length,
            use_context=use_context,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

        model_was_training = model.training
        model.eval()
        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch in loader:
            batch = _move_batch(batch, self.device)
            output = model(batch, task_name=task_name)
            embeddings.append(output["embedding"].detach().cpu())
            labels.append(batch["label"].detach().cpu())
        if model_was_training:
            model.train()

        all_embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(labels, dim=0)
        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(all_labels.tolist()):
            by_label[int(label)].append(index)

        selected: dict[int, list[TaskExample]] = {}
        for label, indices in by_label.items():
            label_embeddings = all_embeddings[indices]
            prototype = label_embeddings.mean(dim=0, keepdim=True)
            distances = torch.cdist(label_embeddings, prototype).squeeze(1)
            sorted_local = distances.argsort().tolist()
            chosen_indices = [indices[local_idx] for local_idx in sorted_local[: self.memory_per_class]]
            selected[label] = [examples[idx] for idx in chosen_indices]
        self._storage[task_name] = selected

    def examples_for(self, task_name: str) -> list[TaskExample]:
        examples: list[TaskExample] = []
        for label_examples in self._storage.get(task_name, {}).values():
            examples.extend(label_examples)
        return examples

    def task_names(self) -> list[str]:
        return sorted(self._storage)

    def __len__(self) -> int:
        return sum(len(items) for task in self._storage.values() for items in task.values())


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved

