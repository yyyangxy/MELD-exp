from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.continual.distillation import kd_loss
from src.data.collate import collate_batch
from src.data.datasets import MeldTextDataset, Vocabulary
from src.data.task_builder import TaskExample
from src.train.evaluator import move_batch


@dataclass
class TrainStats:
    loss: float
    steps: int


def build_loader(
    examples: list[TaskExample],
    vocabulary: Vocabulary,
    speaker_to_id: dict[str, int],
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int = 0,
    use_context: bool = True,
) -> DataLoader:
    dataset = MeldTextDataset(
        examples,
        vocabulary=vocabulary,
        speaker_to_id=speaker_to_id,
        max_length=max_length,
        use_context=use_context,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )


def train_one_task(
    model: nn.Module,
    train_loader: DataLoader,
    task_name: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    grad_clip: float,
    method: str,
    old_task_names: list[str],
    teacher: nn.Module | None,
    replay_loaders: dict[str, DataLoader] | None,
    lambda_replay: float,
    lambda_kd: float,
    temperature: float,
) -> TrainStats:
    criterion = nn.CrossEntropyLoss()
    replay_loaders = replay_loaders or {}
    replay_iters = {task: _infinite(loader) for task, loader in replay_loaders.items()}
    total_loss = 0.0
    total_steps = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            output = model(batch, task_name=task_name)
            loss = criterion(output["logits"], batch["label"])

            if method == "lwf" and teacher is not None and old_task_names:
                kd_terms = []
                for old_task in old_task_names:
                    with torch.no_grad():
                        teacher_logits = teacher(batch, task_name=old_task)["logits"]
                    student_logits = model(batch, task_name=old_task)["logits"]
                    kd_terms.append(kd_loss(student_logits, teacher_logits, temperature))
                if kd_terms:
                    loss = loss + lambda_kd * torch.stack(kd_terms).mean()

            replay_terms = []
            replay_kd_terms = []
            for replay_task, iterator in replay_iters.items():
                replay_batch = move_batch(next(iterator), device)
                replay_output = model(replay_batch, task_name=replay_task)
                replay_terms.append(criterion(replay_output["logits"], replay_batch["label"]))

                if method == "proto_replay_kd" and teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(replay_batch, task_name=replay_task)["logits"]
                    replay_kd_terms.append(kd_loss(replay_output["logits"], teacher_logits, temperature))

            if replay_terms:
                loss = loss + lambda_replay * torch.stack(replay_terms).mean()
            if replay_kd_terms:
                loss = loss + lambda_kd * torch.stack(replay_kd_terms).mean()

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_steps += 1

    return TrainStats(loss=total_loss / max(total_steps, 1), steps=total_steps)


def train_joint(
    model: nn.Module,
    train_loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    grad_clip: float,
) -> TrainStats:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = 0
    task_names = list(train_loaders)

    for _ in range(epochs):
        model.train()
        iterators = {task: iter(loader) for task, loader in train_loaders.items()}
        active = set(task_names)
        while active:
            for task in list(task_names):
                if task not in active:
                    continue
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    active.remove(task)
                    continue

                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, task_name=task)
                loss = criterion(output["logits"], batch["label"])
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_steps += 1

    return TrainStats(loss=total_loss / max(total_steps, 1), steps=total_steps)


def _infinite(loader: DataLoader):
    while True:
        yield from loader
