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
        memory_strategy: str = "prototype_nearest",
        representative_ratio: float = 0.5,
        kmeans_iters: int = 10,
        seed: int = 13,
    ) -> None:
        self.memory_per_class = memory_per_class
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.memory_strategy = memory_strategy
        self.representative_ratio = float(representative_ratio)
        self.kmeans_iters = int(kmeans_iters)
        self.seed = int(seed)
        self._storage: dict[str, dict[int, list[TaskExample]]] = {}
        self.stage_modalities: dict[str, list[str]] = {}

        if self.memory_strategy not in {"prototype_nearest", "diverse", "hybrid", "kmeans_centroid"}:
            raise ValueError(
                "Unknown memory_strategy "
                f"'{self.memory_strategy}'. Expected prototype_nearest, diverse, hybrid, or kmeans_centroid."
            )
        if not 0.0 <= self.representative_ratio <= 1.0:
            raise ValueError("representative_ratio must be in [0, 1]")

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
            chosen_local = self._select_local_indices(label_embeddings)
            selected[label] = [examples[indices[local_idx]] for local_idx in chosen_local]
        self._storage[stage_name] = selected

    def _select_local_indices(self, label_embeddings: torch.Tensor) -> list[int]:
        num_items = int(label_embeddings.shape[0])
        if num_items <= self.memory_per_class:
            return list(range(num_items))
        if self.memory_strategy == "prototype_nearest":
            return self._select_nearest(label_embeddings, self.memory_per_class)
        if self.memory_strategy == "diverse":
            return self._select_diverse_k_center(label_embeddings, self.memory_per_class, excluded=set())
        if self.memory_strategy == "kmeans_centroid":
            return self._select_kmeans_centroid(label_embeddings, self.memory_per_class)
        return self._select_hybrid(label_embeddings)

    def _select_hybrid(self, label_embeddings: torch.Tensor) -> list[int]:
        representative_count = int(round(self.memory_per_class * self.representative_ratio))
        representative_count = max(0, min(self.memory_per_class, representative_count))
        diverse_count = self.memory_per_class - representative_count

        selected: list[int] = []
        if representative_count > 0:
            selected.extend(self._select_nearest(label_embeddings, representative_count))
        if diverse_count > 0:
            selected_set = set(selected)
            selected.extend(
                self._select_diverse_k_center(
                    label_embeddings,
                    diverse_count,
                    excluded=selected_set,
                )
            )
        return _dedupe_preserve_order(selected)[: self.memory_per_class]

    @staticmethod
    def _select_nearest(label_embeddings: torch.Tensor, count: int) -> list[int]:
        if count <= 0:
            return []
        prototype = label_embeddings.mean(dim=0, keepdim=True)
        distances = torch.cdist(label_embeddings, prototype).squeeze(1)
        return distances.argsort().tolist()[:count]

    @staticmethod
    def _select_diverse_k_center(
        label_embeddings: torch.Tensor,
        count: int,
        excluded: set[int],
    ) -> list[int]:
        if count <= 0:
            return []
        candidate_indices = [idx for idx in range(int(label_embeddings.shape[0])) if idx not in excluded]
        if not candidate_indices:
            return []
        if len(candidate_indices) <= count:
            return candidate_indices

        candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
        candidate_embeddings = label_embeddings.index_select(0, candidate_tensor)
        prototype = label_embeddings.mean(dim=0, keepdim=True)
        distances_to_prototype = torch.cdist(candidate_embeddings, prototype).squeeze(1)

        first_local = int(distances_to_prototype.argmax().item())
        chosen_local = [first_local]
        min_distances = torch.cdist(
            candidate_embeddings,
            candidate_embeddings[first_local : first_local + 1],
        ).squeeze(1)

        while len(chosen_local) < count:
            next_local = int(min_distances.argmax().item())
            if next_local in chosen_local:
                break
            chosen_local.append(next_local)
            next_distances = torch.cdist(
                candidate_embeddings,
                candidate_embeddings[next_local : next_local + 1],
            ).squeeze(1)
            min_distances = torch.minimum(min_distances, next_distances)

        return [candidate_indices[local_idx] for local_idx in chosen_local]

    def _select_kmeans_centroid(self, label_embeddings: torch.Tensor, count: int) -> list[int]:
        if count <= 0:
            return []
        num_items = int(label_embeddings.shape[0])
        if num_items <= count:
            return list(range(num_items))

        try:
            from sklearn.cluster import KMeans  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "memory_strategy='kmeans_centroid' requires scikit-learn. "
                "Install it with `pip install scikit-learn` in the active environment."
            ) from exc

        embeddings_np = label_embeddings.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=count,
            random_state=self.seed,
            n_init=10,
            max_iter=self.kmeans_iters,
        )
        assignments = kmeans.fit_predict(embeddings_np)
        centers = torch.from_numpy(kmeans.cluster_centers_).to(label_embeddings)

        selected: list[int] = []
        for cluster_idx in range(count):
            member_indices = torch.nonzero(
                torch.tensor(assignments == cluster_idx, dtype=torch.bool),
                as_tuple=False,
            ).reshape(-1)
            if member_indices.numel() == 0:
                continue
            member_embeddings = label_embeddings.index_select(0, member_indices.to(label_embeddings.device))
            center = centers[cluster_idx : cluster_idx + 1]
            distances = torch.cdist(member_embeddings, center).squeeze(1)
            chosen_member = int(member_indices[int(distances.argmin().item())].item())
            selected.append(chosen_member)

        selected = _dedupe_preserve_order(selected)
        if len(selected) < count:
            selected.extend(
                _nearest_unselected_to_prototype(
                    label_embeddings,
                    count=count - len(selected),
                    excluded=set(selected),
                )
            )
        return _dedupe_preserve_order(selected)[:count]

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


def _dedupe_preserve_order(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        deduped.append(index)
    return deduped


def _nearest_unselected_to_prototype(
    label_embeddings: torch.Tensor,
    count: int,
    excluded: set[int],
) -> list[int]:
    if count <= 0:
        return []
    prototype = label_embeddings.mean(dim=0, keepdim=True)
    distances = torch.cdist(label_embeddings, prototype).squeeze(1)
    selected: list[int] = []
    for index in distances.argsort().tolist():
        if index in excluded:
            continue
        selected.append(index)
        if len(selected) >= count:
            break
    return selected
