from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.data.meld_csv import EMOTION_LABELS, read_all_splits
from src.train.metrics import compute_classification_metrics
from src.utils.paths import ensure_dir, load_config, resolve_data_root, resolve_experiment_output_dir, resolve_path
from src.utils.seed import seed_everything


class MeldTextEmotionDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.utterance,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(EMOTION_LABELS[record.emotion], dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R on MELD emotion-only text classification.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl_v2.yaml"))
    parser.add_argument("--run-name", default="xlmr_emotion_only_finetune")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--log-steps", type=int, default=100)
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> Path:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    config = load_config(args.config)
    config.setdefault("run", {})["enabled"] = True
    config.setdefault("run", {})["group"] = "text_finetune"
    config.setdefault("run", {})["name"] = args.run_name
    output_dir = resolve_experiment_output_dir(config)
    seed_everything(int(config.get("seed", 13)))

    model_path = args.model_path or config.get("feature_paths", {}).get("text_model_path", "xlm-roberta-large")
    model_path = str(resolve_path(model_path, PROJECT_ROOT)) if not str(model_path).startswith("xlm-") else str(model_path)
    device = _resolve_device(args.device)

    split_records = read_all_splits(
        resolve_data_root(config),
        warn_missing_videos=bool(config.get("data", {}).get("warn_missing_videos", True)),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(EMOTION_LABELS),
        local_files_only=True,
    ).to(device)

    train_dataset = MeldTextEmotionDataset(split_records["train"], tokenizer, args.max_length)
    dev_dataset = MeldTextEmotionDataset(split_records["dev"], tokenizer, args.max_length)
    test_dataset = MeldTextEmotionDataset(split_records["test"], tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=_weighted_sampler(split_records["train"]),
        num_workers=0,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.epochs) // max(args.grad_accum_steps, 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))

    rows = []
    best_dev = -1.0
    best_epoch = 0
    best_path = ensure_dir(output_dir / "checkpoints") / "best_dev.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            args.grad_accum_steps,
            args.fp16,
            args.log_steps,
            epoch,
            args.epochs,
        )
        dev_metrics = _evaluate(model, dev_loader, device)
        test_metrics = _evaluate(model, test_loader, device)
        rows.extend(_metric_rows(epoch, "dev", dev_metrics))
        rows.extend(_metric_rows(epoch, "test", test_metrics))
        print(
            f"epoch={epoch}/{args.epochs} loss={train_loss:.4f} "
            f"dev_wf1={dev_metrics['weighted_f1']:.4f} test_wf1={test_metrics['weighted_f1']:.4f}"
        )
        if float(dev_metrics["weighted_f1"]) > best_dev:
            best_dev = float(dev_metrics["weighted_f1"])
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    best_test = _evaluate(model, test_loader, device)
    rows.extend(_metric_rows(best_epoch, "best_dev_test", best_test))
    result_path = output_dir / "results" / "text_emotion_finetune.csv"
    _write_rows(result_path, rows)
    print(f"best_epoch={best_epoch} best_dev_wf1={best_dev:.4f} best_test_wf1={best_test['weighted_f1']:.4f}")
    print(f"results={result_path}")
    return result_path


def _train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    grad_accum_steps: int,
    fp16: bool,
    log_steps: int,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=bool(fp16 and device.type == "cuda")):
            output = model(**batch)
            loss = output.loss / max(grad_accum_steps, 1)
        scaler.scale(loss).backward()
        if step % max(grad_accum_steps, 1) == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * max(grad_accum_steps, 1)
        steps += 1
        if log_steps > 0 and step % log_steps == 0:
            print(
                f"epoch={epoch}/{epochs} step={step}/{len(loader)} "
                f"train_loss={total_loss / max(steps, 1):.4f}",
                flush=True,
            )
    return total_loss / max(steps, 1)


@torch.no_grad()
def _evaluate(model, loader, device) -> dict[str, float | None]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        labels = batch["labels"]
        batch = _move_batch(batch, device)
        logits = model(**batch).logits
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return compute_classification_metrics(y_true, y_pred, num_labels=len(EMOTION_LABELS))


def _weighted_sampler(records) -> WeightedRandomSampler:
    labels = [EMOTION_LABELS[record.emotion] for record in records]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long)).float()
    counts[counts == 0] = 1.0
    weights = torch.tensor([1.0 / counts[label].item() for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _metric_rows(epoch: int, split: str, metrics: dict[str, float | None]) -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "split": split,
            "task": "emotion",
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
        }
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "split", "task", "accuracy", "weighted_f1", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
