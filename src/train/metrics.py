from __future__ import annotations


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_labels: int,
    positive_label: int | None = None,
) -> dict[str, float | None]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {
            "accuracy": 0.0,
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "positive_f1_for_shift": None,
        }

    correct = sum(int(true == pred) for true, pred in zip(y_true, y_pred))
    f1s: list[float] = []
    supports: list[int] = []
    for label in range(num_labels):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        support = sum(1 for true in y_true if true == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        supports.append(support)

    total = sum(supports)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1s, supports)) / total if total else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    positive_f1 = f1s[positive_label] if positive_label is not None else None

    return {
        "accuracy": correct / len(y_true),
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "positive_f1_for_shift": positive_f1,
    }


def decorate_final_metrics(rows: list[dict[str, object]], task_order: list[str]) -> list[dict[str, object]]:
    if not rows:
        return rows
    final_stage = rows[-1]["stage"]
    best_by_task: dict[str, float] = {}
    final_by_task: dict[str, float] = {}
    for row in rows:
        task = str(row["task"])
        score = float(row["weighted_f1"])
        best_by_task[task] = max(best_by_task.get(task, 0.0), score)
        if row["stage"] == final_stage:
            final_by_task[task] = score

    final_scores = [final_by_task[task] for task in task_order if task in final_by_task]
    final_avg = sum(final_scores) / len(final_scores) if final_scores else 0.0

    for row in rows:
        task = str(row["task"])
        if row["stage"] == final_stage and task in final_by_task:
            best = best_by_task.get(task, 0.0)
            final = final_by_task[task]
            row["final_avg"] = final_avg
            row["forgetting"] = max(0.0, best - final)
            row["retention"] = final / best if best > 0 else 0.0
        else:
            row["final_avg"] = ""
            row["forgetting"] = ""
            row["retention"] = ""
    return rows

