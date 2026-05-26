from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.dialogue_text_task_runner import DIALOGUE_TEXT_TASK_METHODS, run_dialogue_text_task_experiment


DEFAULT_METHODS = [
    "context_free",
    "hier_bilstm",
    "dlg_seq_ft",
    "dlg_seq_kd",
    "dlg_random_replay",
    "dlg_sa_cmd_no_replay",
    "text_task_sa_cmd",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end text dialogue-level Task-STL experiments.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dialogue_task_stl_v2.yaml"))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Run each method for each seed in order.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--backbone", dest="text_model_path", default=None, help="Text backbone name or local path, e.g. bert-base-uncased.")
    parser.add_argument("--text-model-path", dest="text_model_path", default=None, help="Alias for --backbone.")
    parser.add_argument("--memory-per-class", type=int, default=None)
    parser.add_argument(
        "--replay-strategy",
        choices=[
            "random",
            "prototype_nearest",
            "diverse",
            "hybrid",
            "prototype_nearest_klmap",
            "diverse_klmap",
            "hybrid_klmap",
        ],
        default=None,
    )
    parser.add_argument("--representative-ratio", type=float, default=None)
    parser.add_argument("--klmap-dim", type=int, default=None, help="Low-dimensional KLMap replay-selection feature size.")
    parser.add_argument("--cl-reg-lambda", type=float, default=None)
    parser.add_argument("--importance-max-batches", type=int, default=None)
    parser.add_argument("--regularizer-scope", choices=["non_encoder", "all"], default=None)
    parser.add_argument("--si-xi", type=float, default=None)
    parser.add_argument("--packnet-prune-ratio", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-id", default=None, help="Physical GPU id to expose, e.g. 8. Overrides --device to cuda.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--replay-batch-kd", action="store_true", help="Apply teacher KD/relation on replay batches.")
    parser.add_argument("--freeze-old-heads", action="store_true", help="Freeze task heads after their stage is learned.")
    args = parser.parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda"

    unknown = sorted(set(args.methods) - DIALOGUE_TEXT_TASK_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Expected subset of {sorted(DIALOGUE_TEXT_TASK_METHODS)}")

    base_train_overrides = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "text_model_path": args.text_model_path,
        "memory_per_class": args.memory_per_class,
        "replay_strategy": args.replay_strategy,
        "representative_ratio": args.representative_ratio,
        "klmap_dim": args.klmap_dim,
        "cl_reg_lambda": args.cl_reg_lambda,
        "importance_max_batches": args.importance_max_batches,
        "regularizer_scope": args.regularizer_scope,
        "si_xi": args.si_xi,
        "packnet_prune_ratio": args.packnet_prune_ratio,
        "device": args.device,
        "replay_batch_kd": True if args.replay_batch_kd else None,
        "freeze_old_heads": True if args.freeze_old_heads else None,
    }
    if args.fp16:
        base_train_overrides["fp16"] = True
    if args.no_fp16:
        base_train_overrides["fp16"] = False

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        train_overrides = dict(base_train_overrides)
        train_overrides["seed"] = seed
        for method in args.methods:
            run_name_parts = [args.run_name] if args.run_name else []
            if seed is not None:
                run_name_parts.append(f"seed{seed}")
            if len(args.methods) > 1:
                run_name_parts.append(method)
            method_run_name = "_".join(run_name_parts) if run_name_parts else None
            result_path = run_dialogue_text_task_experiment(
                args.config,
                method,
                run_name=method_run_name,
                train_overrides=train_overrides,
            )
            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"{method}{seed_label}: {result_path}")


if __name__ == "__main__":
    main()
