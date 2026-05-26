from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.dialogue_task_runner import DIALOGUE_TASK_METHODS, run_dialogue_task_experiment


DEFAULT_METHODS = [
    "dlg_seq_ft",
    "dlg_seq_kd",
    "dlg_task_sa_cmd",
    "dlg_task_pg_trd",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-feature multimodal dialogue-level Task-STL experiments (S5)."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dialogue_multimodal_task_stl_v2.yaml"))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--lambda-kd", type=float, default=None)
    parser.add_argument("--lambda-rel", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--active-modalities",
        nargs="*",
        choices=["text", "audio", "visual"],
        default=None,
        help="Modalities to use for all task stages. Default comes from config.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-id", default=None, help="Physical GPU id to expose, e.g. 2. Overrides --device to cuda.")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda"

    unknown = sorted(set(args.methods) - DIALOGUE_TASK_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Expected subset of {sorted(DIALOGUE_TASK_METHODS)}")

    base_overrides = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "lambda_kd": args.lambda_kd,
        "lambda_rel": args.lambda_rel,
        "temperature": args.temperature,
        "active_modalities": args.active_modalities,
        "device": args.device,
    }

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        overrides = dict(base_overrides)
        overrides["seed"] = seed
        for method in args.methods:
            run_name_parts = [args.run_name] if args.run_name else []
            if seed is not None:
                run_name_parts.append(f"seed{seed}")
            if len(args.methods) > 1:
                run_name_parts.append(method)
            result_path = run_dialogue_task_experiment(
                args.config,
                method,
                run_name="_".join(run_name_parts) if run_name_parts else None,
                train_overrides=overrides,
            )
            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"{method}{seed_label}: {result_path}")


if __name__ == "__main__":
    main()
