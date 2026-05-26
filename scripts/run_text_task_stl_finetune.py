from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.data.meld_csv import TASK_LABELS, read_all_splits
from src.data.stl_task_splits import (
    filter_task_examples_by_stl_split,
    load_stl_task_split,
    resolve_stl_task_split_root,
)
from src.data.task_builder import build_all_tasks
from src.losses.sa_cmd import confidence_weights, masked_kd_loss, sample_relation_loss
from src.train.metrics import compute_classification_metrics
from src.utils.paths import ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


TEXT_TASK_METHODS = [
    "seq_ft",
    "lwf",
    "text_random_replay",
    "text_sa_cmd_no_replay",
    "text_task_sa_cmd",
    "text_task_sa_cmd_replay_kd",
    "text_task_sa_cmd_freeze_old_heads",
    "text_task_sa_cmd_replay_kd_freeze_old_heads",
]


class TextTaskDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(example.label, dtype=torch.long),
        }


class XLMRTaskSTLModel(nn.Module):
    def __init__(self, model_path: str, task_order: list[str]) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = int(self.encoder.config.hidden_size)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, len(TASK_LABELS[task])) for task in task_order}
        )

    def forward(self, batch: dict[str, torch.Tensor], task_name: str) -> dict[str, torch.Tensor]:
        output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        embedding = output.last_hidden_state[:, 0]
        return {"logits": self.heads[task_name](embedding), "embedding": embedding}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R for text-only Task-STL.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl_v2.yaml"))
    parser.add_argument(
        "--method",
        choices=TEXT_TASK_METHODS,
        default=None,
        help="Single method to run. Kept for backward compatibility.",
    )
    parser.add_argument("--methods", nargs="*", choices=TEXT_TASK_METHODS, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-rel", type=float, default=1.0)
    parser.add_argument("--memory-per-class", type=int, default=100)
    parser.add_argument("--replay-strategy", choices=["random", "prototype_nearest", "diverse", "hybrid"], default="random")
    parser.add_argument("--representative-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=None, help="Physical GPU id to expose, e.g. 8. Overrides --device to cuda.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--replay-batch-kd", action="store_true", help="Apply teacher KD/relation on replay batches.")
    parser.add_argument("--freeze-old-heads", action="store_true", help="Freeze task heads after their stage is learned.")
    parser.add_argument("--log-steps", type=int, default=100)
    args = parser.parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda"
    methods = args.methods or ([args.method] if args.method else TEXT_TASK_METHODS)
    for method in methods:
        method_args = copy.copy(args)
        method_args.method = method
        if args.run_name and len(methods) > 1:
            method_args.run_name = f"{args.run_name}_{method}"
        result_path = run(method_args)
        print(f"{method}: {result_path}")


def run(args: argparse.Namespace) -> Path:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = load_config(args.config)
    config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "text_task_stl_finetune"
    config.setdefault("run", {})["name"] = args.run_name or args.method
    output_dir = resolve_experiment_output_dir(config)
    _write_run_parameters(output_dir, config, args)
    seed_everything(int(config.get("seed", 13)))

    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    model_path = args.model_path or config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large")
    model_path = str(resolve_path(model_path, PROJECT_ROOT)) if not str(model_path).startswith("xlm-") else str(model_path)
    device = _resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    data_root = resolve_data_root(config)
    split_records = read_all_splits(
        data_root,
        warn_missing_videos=bool(config.get("data", {}).get("warn_missing_videos", True)),
    )
    task_examples = build_all_tasks(split_records, task_order, context_window=0)
    split_root = resolve_stl_task_split_root(config.get("data", {}), data_root)
    if split_root is not None:
        task_split = load_stl_task_split(split_root, task_order, split_records.keys())
        task_examples = filter_task_examples_by_stl_split(task_examples, task_split)
    loaders = {
        split: {
            task: _loader(task_examples[split][task], tokenizer, args.max_length, args.batch_size, split == "train")
            for task in task_order
        }
        for split in ["train", "dev", "test"]
    }

    model = XLMRTaskSTLModel(model_path, task_order).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = sum(len(loaders["train"][task]) for task in task_order) * args.epochs
    total_steps = max(1, total_steps // max(args.grad_accum_steps, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    teacher: nn.Module | None = None
    learned_tasks: list[str] = []
    replay_memory: dict[str, list] = {}
    rows: list[dict[str, object]] = []
    for stage_idx, task in enumerate(task_order, start=1):
        if _uses_freeze_old_heads(args):
            _freeze_task_heads(model, learned_tasks)
        replay_loaders = {
            old_task: _loader(examples, tokenizer, args.max_length, args.batch_size, True)
            for old_task, examples in replay_memory.items()
            if examples
        }
        for epoch in range(1, args.epochs + 1):
            loss = _train_epoch(
                model,
                teacher,
                loaders["train"][task],
                replay_loaders,
                learned_tasks,
                task,
                optimizer,
                scheduler,
                scaler,
                criterion,
                device,
                args,
                epoch,
            )
            print(f"method={args.method} stage={task} epoch={epoch}/{args.epochs} loss={loss:.4f}", flush=True)
        learned_tasks.append(task)
        if _uses_replay(args.method):
            replay_memory[task] = _select_replay_examples(
                model,
                task_examples["train"][task],
                tokenizer,
                args.max_length,
                args.batch_size,
                device,
                task,
                args.memory_per_class,
                args.replay_strategy,
                args.representative_ratio,
                seed=int(config.get("seed", 13)) + stage_idx,
            )
            print(
                f"method={args.method} stage={task} selected_replay={len(replay_memory[task])} "
                f"memory_per_class={args.memory_per_class}",
                flush=True,
            )
        teacher = _clone_frozen(model, device)
        torch.save(model.state_dict(), ensure_dir(output_dir / "checkpoints") / f"{args.method}_stage{stage_idx}_{task}.pt")
        for split in ["dev", "test"]:
            rows.extend(_evaluate(model, loaders[split], learned_tasks, device, args.method, split, f"stage_{stage_idx}_{task}"))

    result_path = output_dir / "results" / "text_task_stl_finetune.csv"
    _write_rows(result_path, rows)
    print(f"results={result_path}")
    return result_path


def _train_epoch(model, teacher, loader, replay_loaders, learned_tasks, task, optimizer, scheduler, scaler, criterion, device, args, epoch) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    steps = 0
    replay_iters = {old_task: _infinite(replay_loader) for old_task, replay_loader in replay_loaders.items()}
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=bool(args.fp16 and device.type == "cuda")):
            output = model(batch, task)
            loss = criterion(output["logits"], batch["labels"])
            if _uses_replay(args.method):
                replay_terms = []
                for old_task, iterator in replay_iters.items():
                    replay_batch = _move_batch(next(iterator), device)
                    replay_output = model(replay_batch, old_task)
                    replay_loss = criterion(replay_output["logits"], replay_batch["labels"])
                    if _uses_replay_batch_kd(args) and teacher is not None:
                        with torch.no_grad():
                            teacher_replay_out = teacher(replay_batch, old_task)
                        replay_weights = confidence_weights(teacher_replay_out["logits"])
                        replay_loss = replay_loss + args.lambda_kd * masked_kd_loss(
                            replay_output["logits"],
                            teacher_replay_out["logits"],
                            temperature=args.temperature,
                            weights=replay_weights,
                        )
                        replay_loss = replay_loss + args.lambda_rel * sample_relation_loss(
                            replay_output["embedding"],
                            teacher_replay_out["embedding"],
                            weights=replay_weights,
                        )
                    replay_terms.append(replay_loss)
                if replay_terms:
                    loss = torch.stack([loss, *replay_terms]).mean()
            if teacher is not None and learned_tasks and _uses_kd(args.method):
                kd_terms = []
                rel_terms = []
                for old_task in learned_tasks:
                    with torch.no_grad():
                        teacher_out = teacher(batch, old_task)
                    student_out = model(batch, old_task)
                    weights = (
                        confidence_weights(teacher_out["logits"])
                        if _uses_confidence_relation(args.method)
                        else None
                    )
                    kd_terms.append(masked_kd_loss(student_out["logits"], teacher_out["logits"], temperature=args.temperature, weights=weights))
                    if _uses_confidence_relation(args.method):
                        rel_terms.append(sample_relation_loss(student_out["embedding"], teacher_out["embedding"], weights=weights))
                loss = loss + args.lambda_kd * torch.stack(kd_terms).mean()
                if rel_terms:
                    loss = loss + args.lambda_rel * torch.stack(rel_terms).mean()
            loss = loss / max(args.grad_accum_steps, 1)
        scaler.scale(loss).backward()
        if step % max(args.grad_accum_steps, 1) == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * max(args.grad_accum_steps, 1)
        steps += 1
        if args.log_steps > 0 and step % args.log_steps == 0:
            print(
                f"method={args.method} stage={task} epoch={epoch}/{args.epochs} "
                f"step={step}/{len(loader)} loss={total_loss / max(steps, 1):.4f}",
                flush=True,
            )
    return total_loss / max(steps, 1)


def _uses_replay(method: str) -> bool:
    return method in {
        "text_random_replay",
        "text_task_sa_cmd",
        "text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _uses_kd(method: str) -> bool:
    return method in {
        "lwf",
        "text_sa_cmd_no_replay",
        "text_task_sa_cmd",
        "text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _uses_confidence_relation(method: str) -> bool:
    return method in {
        "text_sa_cmd_no_replay",
        "text_task_sa_cmd",
        "text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _uses_replay_batch_kd(args: argparse.Namespace) -> bool:
    return bool(args.replay_batch_kd) or args.method in {
        "text_task_sa_cmd_replay_kd",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _uses_freeze_old_heads(args: argparse.Namespace) -> bool:
    return bool(args.freeze_old_heads) or args.method in {
        "text_task_sa_cmd_freeze_old_heads",
        "text_task_sa_cmd_replay_kd_freeze_old_heads",
    }


def _freeze_task_heads(model: XLMRTaskSTLModel, learned_tasks: list[str]) -> None:
    for task_name, head in model.heads.items():
        requires_grad = task_name not in learned_tasks
        for parameter in head.parameters():
            parameter.requires_grad_(requires_grad)


@torch.no_grad()
def _select_replay_examples(
    model,
    examples,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    task: str,
    memory_per_class: int,
    replay_strategy: str,
    representative_ratio: float,
    seed: int,
) -> list:
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for index, example in enumerate(examples):
        by_label.setdefault(int(example.label), []).append(example)
    if replay_strategy == "random":
        selected = []
        for label in sorted(by_label):
            label_examples = list(by_label[label])
            rng.shuffle(label_examples)
            selected.extend(label_examples[:memory_per_class])
        return selected

    loader = _loader(examples, tokenizer, max_length, batch_size, False)
    was_training = model.training
    model.eval()
    embeddings = []
    labels = []
    for batch in loader:
        labels.append(batch["labels"].detach().cpu())
        batch = _move_batch(batch, device)
        output = model(batch, task)
        embeddings.append(output["embedding"].detach().cpu())
    if was_training:
        model.train()
    if not embeddings:
        return []

    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.cat(labels, dim=0)
    selected = []
    for label in sorted(set(all_labels.tolist())):
        indices = torch.nonzero(all_labels == int(label), as_tuple=False).reshape(-1).tolist()
        if len(indices) <= memory_per_class:
            selected.extend(examples[index] for index in indices)
            continue
        label_embeddings = all_embeddings[indices]
        local_indices = _select_replay_indices(label_embeddings, memory_per_class, replay_strategy, representative_ratio)
        selected.extend(examples[indices[local_idx]] for local_idx in local_indices)
    return selected


def _select_replay_indices(label_embeddings: torch.Tensor, count: int, replay_strategy: str, representative_ratio: float) -> list[int]:
    if replay_strategy == "prototype_nearest":
        return _select_nearest_indices(label_embeddings, count)
    if replay_strategy == "diverse":
        return _select_diverse_indices(label_embeddings, count, excluded=set())
    if replay_strategy == "hybrid":
        return _select_hybrid_indices(label_embeddings, count, representative_ratio)
    raise ValueError(f"Unknown replay_strategy: {replay_strategy}")


def _select_nearest_indices(label_embeddings: torch.Tensor, count: int) -> list[int]:
    prototype = label_embeddings.mean(dim=0, keepdim=True)
    distances = torch.cdist(label_embeddings, prototype).squeeze(1)
    return distances.argsort().tolist()[:count]


def _select_hybrid_indices(label_embeddings: torch.Tensor, count: int, representative_ratio: float) -> list[int]:
    representative_count = max(0, min(count, int(round(count * representative_ratio))))
    diverse_count = count - representative_count
    selected = []
    if representative_count > 0:
        selected.extend(_select_nearest_indices(label_embeddings, representative_count))
    if diverse_count > 0:
        selected.extend(_select_diverse_indices(label_embeddings, diverse_count, set(selected)))
    return _dedupe(selected)[:count]


def _select_diverse_indices(label_embeddings: torch.Tensor, count: int, excluded: set[int]) -> list[int]:
    candidates = [idx for idx in range(int(label_embeddings.shape[0])) if idx not in excluded]
    if len(candidates) <= count:
        return candidates
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    candidate_embeddings = label_embeddings.index_select(0, candidate_tensor)
    prototype = label_embeddings.mean(dim=0, keepdim=True)
    first = int(torch.cdist(candidate_embeddings, prototype).squeeze(1).argmax().item())
    chosen = [first]
    min_distances = torch.cdist(candidate_embeddings, candidate_embeddings[first : first + 1]).squeeze(1)
    while len(chosen) < count:
        next_idx = int(min_distances.argmax().item())
        if next_idx in chosen:
            break
        chosen.append(next_idx)
        next_distances = torch.cdist(candidate_embeddings, candidate_embeddings[next_idx : next_idx + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, next_distances)
    return [candidates[idx] for idx in chosen]


def _dedupe(indices: list[int]) -> list[int]:
    seen = set()
    result = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        result.append(index)
    return result


@torch.no_grad()
def _evaluate(model, loaders_by_task, learned_tasks, device, method, split, stage) -> list[dict[str, object]]:
    model.eval()
    rows = []
    for task in learned_tasks:
        y_true, y_pred = [], []
        for batch in loaders_by_task[task]:
            labels = batch["labels"]
            batch = _move_batch(batch, device)
            logits = model(batch, task)["logits"]
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred, len(TASK_LABELS[task]), positive_label=1 if task == "shift" else None)
        rows.append({"method": method, "split": split, "stage": stage, "task": task, **metrics})
    return rows


def _loader(examples, tokenizer, max_length, batch_size, shuffle) -> DataLoader:
    dataset = TextTaskDataset(examples, tokenizer, max_length)
    sampler = None
    if shuffle:
        labels = [example.label for example in examples]
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long)).float()
        counts[counts == 0] = 1.0
        weights = torch.tensor([1.0 / counts[label].item() for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=0)


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["method", "split", "stage", "task", "accuracy", "weighted_f1", "macro_f1", "positive_f1_for_shift"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_parameters(output_dir: Path, config: dict, args: argparse.Namespace) -> None:
    payload = {
        "method": args.method,
        "cli_args": vars(args),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "effective_train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "lambda_kd": args.lambda_kd,
            "lambda_rel": args.lambda_rel,
            "memory_per_class": args.memory_per_class,
            "replay_strategy": args.replay_strategy,
            "representative_ratio": args.representative_ratio,
            "replay_batch_kd": _uses_replay_batch_kd(args),
            "freeze_old_heads": _uses_freeze_old_heads(args),
            "device": args.device,
            "gpu_id": args.gpu_id,
            "fp16": bool(args.fp16),
        },
    }
    path = ensure_dir(output_dir / "logs") / "run_parameters.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clone_frozen(model: nn.Module, device: torch.device) -> nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
