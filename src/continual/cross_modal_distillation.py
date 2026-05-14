from __future__ import annotations

import torch
from torch.nn import functional as F


def cross_modal_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    confidence_weighted: bool = True,
) -> torch.Tensor:
    per_sample = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temperature**2)

    if not confidence_weighted:
        return per_sample.mean()

    with torch.no_grad():
        weights = F.softmax(teacher_logits, dim=-1).max(dim=-1).values
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)

