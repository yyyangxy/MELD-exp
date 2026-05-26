from __future__ import annotations

from collections import defaultdict

import torch


class TaskPrototypeBank:
    """Stores class prototypes for each learned task."""

    def __init__(self) -> None:
        self._storage: dict[str, dict[int, torch.Tensor]] = {}

    def update_from_embeddings(
        self,
        task_name: str,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if embeddings.numel() == 0:
            self._storage[task_name] = {}
            return

        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu().long()
        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels.tolist()):
            by_label[int(label)].append(index)

        prototypes: dict[int, torch.Tensor] = {}
        for label, indices in by_label.items():
            index_tensor = torch.tensor(indices, dtype=torch.long)
            prototypes[label] = embeddings.index_select(0, index_tensor).mean(dim=0)
        self._storage[task_name] = prototypes

    def prototypes_for(self, task_name: str, device: torch.device | str) -> torch.Tensor | None:
        prototypes = self._storage.get(task_name)
        if not prototypes:
            return None
        ordered = [prototypes[label] for label in sorted(prototypes)]
        return torch.stack(ordered, dim=0).to(device)

    def task_names(self) -> list[str]:
        return sorted(self._storage)
