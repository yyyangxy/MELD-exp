from __future__ import annotations

import torch

from src.models.stl_model import TASK_NUM_LABELS
from src.train.metrics import compute_classification_metrics


@torch.no_grad()
def evaluate(model, loader, task_name: str, device: torch.device) -> dict[str, float | None]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch, task_name=task_name)
        predictions = output["logits"].argmax(dim=-1)
        y_true.extend(batch["label"].detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
    positive_label = 1 if task_name == "shift" else None
    return compute_classification_metrics(
        y_true,
        y_pred,
        num_labels=TASK_NUM_LABELS[task_name],
        positive_label=positive_label,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved

