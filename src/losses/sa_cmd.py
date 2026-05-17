from __future__ import annotations

import torch
from torch.nn import functional as F


def flatten_valid_sequence(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if values.dim() < 3:
        flat = values
        if mask is None:
            return flat, None
        return flat[mask.reshape(-1).bool()], mask.reshape(-1).bool()

    flat = values.reshape(-1, values.shape[-1])
    if mask is None:
        return flat, None
    flat_mask = mask.reshape(-1).bool()
    return flat[flat_mask], flat_mask


def confidence_weights(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    mode: str = "max_prob",
) -> torch.Tensor:
    flat_logits, _ = flatten_valid_sequence(logits, mask)
    if flat_logits.numel() == 0:
        return torch.ones(0, device=logits.device, dtype=logits.dtype)

    probabilities = F.softmax(flat_logits, dim=-1)
    if mode == "entropy":
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(probabilities.shape[-1]), device=logits.device))
        weights = 1.0 - entropy / max_entropy.clamp_min(1e-8)
    else:
        weights = probabilities.max(dim=-1).values
    return weights.detach().clamp(min=0.05)


def masked_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 2.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    student_flat, _ = flatten_valid_sequence(student_logits, mask)
    teacher_flat, _ = flatten_valid_sequence(teacher_logits, mask)
    if student_flat.numel() == 0:
        return student_logits.sum() * 0.0

    per_sample = F.kl_div(
        F.log_softmax(student_flat / temperature, dim=-1),
        F.softmax(teacher_flat / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temperature**2)

    if weights is None:
        return per_sample.mean()
    if weights.numel() != per_sample.numel():
        raise ValueError(f"KD weights length mismatch: got {weights.numel()}, expected {per_sample.numel()}")
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)


def sample_relation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    max_items: int = 128,
) -> torch.Tensor:
    student_flat, _ = flatten_valid_sequence(student_embedding, mask)
    teacher_flat, _ = flatten_valid_sequence(teacher_embedding, mask)
    if student_flat.shape[0] < 2:
        return student_embedding.sum() * 0.0

    if student_flat.shape[0] > max_items:
        indices = torch.linspace(0, student_flat.shape[0] - 1, steps=max_items, device=student_flat.device).long()
        student_flat = student_flat.index_select(0, indices)
        teacher_flat = teacher_flat.index_select(0, indices)
        if weights is not None:
            weights = weights.index_select(0, indices)

    student_norm = F.normalize(student_flat, dim=-1)
    teacher_norm = F.normalize(teacher_flat.detach(), dim=-1)
    student_relation = student_norm @ student_norm.transpose(0, 1)
    teacher_relation = teacher_norm @ teacher_norm.transpose(0, 1)
    loss_matrix = (student_relation - teacher_relation).pow(2)

    if weights is None:
        return loss_matrix.mean()
    pair_weights = weights.detach().clamp(min=0.05)
    pair_weights = pair_weights[:, None] * pair_weights[None, :]
    return (loss_matrix * pair_weights).sum() / pair_weights.sum().clamp(min=1e-6)
