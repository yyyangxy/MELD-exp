from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MeldMultimodalFeatureDataset, collate_multimodal_batch
from src.data.task_builder import TaskExample


class MultimodalPrototypeMemory:
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
        self.stage_modalities: dict[str, list[str]] = {}

    @torch.no_grad()
    def update(
        self,
        stage_name: str,
        examples: list[TaskExample],
        active_modalities: list[str],
        model: torch.nn.Module,
        feature_root: str,
        feature_dims: dict[str, int],
        speaker_to_id: dict[str, int],
        all_modalities: list[str],
    ) -> None:
        self.stage_modalities[stage_name] = list(active_modalities)
        if not examples:
            self._storage[stage_name] = {}
            return

        dataset = MeldMultimodalFeatureDataset(
            examples,
            feature_root=feature_root,
            feature_dims=feature_dims,
            speaker_to_id=speaker_to_id,
            active_modalities=active_modalities,
            all_modalities=all_modalities,
            allow_missing=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_multimodal_batch,
        )

        was_training = model.training
        model.eval()
        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch in loader:
            batch = _move_batch(batch, self.device)
            output = model(batch, task_name="emotion", active_modalities=active_modalities)
            embeddings.append(output["embedding"].detach().cpu())
            labels.append(batch["label"].detach().cpu())
        if was_training:
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
            chosen_local = distances.argsort().tolist()[: self.memory_per_class]
            selected[label] = [examples[indices[local_idx]] for local_idx in chosen_local]
        self._storage[stage_name] = selected

    def examples_for(self, stage_name: str) -> list[TaskExample]:
        examples: list[TaskExample] = []
        for label_examples in self._storage.get(stage_name, {}).values():
            examples.extend(label_examples)
        return examples

    def stage_names(self) -> list[str]:
        return sorted(self._storage)

    def __len__(self) -> int:
        return sum(len(items) for stage in self._storage.values() for items in stage.values())


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                nested_key: nested_value.to(device) if hasattr(nested_value, "to") else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved

