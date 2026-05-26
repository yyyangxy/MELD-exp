from __future__ import annotations

import torch
from torch.nn import functional as F

from src.losses.sa_cmd import flatten_valid_sequence


def prototype_alignment_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    prototypes: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if prototypes is None or prototypes.numel() == 0:
        return student_embedding.sum() * 0.0

    student_flat, _ = flatten_valid_sequence(student_embedding, mask)
    teacher_flat, _ = flatten_valid_sequence(teacher_embedding, mask)
    if student_flat.numel() == 0:
        return student_embedding.sum() * 0.0

    proto = F.normalize(prototypes.detach(), dim=-1)
    student_sim = F.normalize(student_flat, dim=-1) @ proto.transpose(0, 1)
    teacher_sim = F.normalize(teacher_flat.detach(), dim=-1) @ proto.transpose(0, 1)
    return F.mse_loss(student_sim, teacher_sim)


def task_relation_distillation_loss(
    student_logits_by_task: dict[str, torch.Tensor],
    teacher_logits_by_task: dict[str, torch.Tensor],
    masks_by_task: dict[str, torch.Tensor] | None = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    task_names = [task for task in teacher_logits_by_task if task in student_logits_by_task]
    if len(task_names) < 2:
        sample = next(iter(student_logits_by_task.values()), None)
        if sample is None:
            return torch.tensor(0.0)
        return sample.sum() * 0.0

    max_dim = max(teacher_logits_by_task[task].shape[-1] for task in task_names)
    student_vectors = []
    teacher_vectors = []
    for task in task_names:
        mask = masks_by_task.get(task) if masks_by_task is not None else None
        student_flat, _ = flatten_valid_sequence(student_logits_by_task[task], mask)
        teacher_flat, _ = flatten_valid_sequence(teacher_logits_by_task[task], mask)
        if student_flat.numel() == 0:
            continue

        student_prob = F.softmax(student_flat / temperature, dim=-1).mean(dim=0)
        teacher_prob = F.softmax(teacher_flat / temperature, dim=-1).mean(dim=0)
        student_vectors.append(_pad_to_dim(student_prob, max_dim))
        teacher_vectors.append(_pad_to_dim(teacher_prob, max_dim))

    if len(student_vectors) < 2:
        sample = next(iter(student_logits_by_task.values()))
        return sample.sum() * 0.0

    student_matrix = _relation_matrix(torch.stack(student_vectors, dim=0))
    teacher_matrix = _relation_matrix(torch.stack(teacher_vectors, dim=0).detach())
    return F.mse_loss(student_matrix, teacher_matrix)


def _pad_to_dim(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[0] == dim:
        return values
    return F.pad(values, (0, dim - values.shape[0]))


def _relation_matrix(values: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(values, dim=-1)
    return normalized @ normalized.transpose(0, 1)
