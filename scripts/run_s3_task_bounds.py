from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.dialogue_text_task_runner import run_dialogue_text_task_experiment


BOUND_METHODS = {
    "lower_seq_ft": "dlg_seq_ft",
    "upper_joint_bilstm": "hier_bilstm",
    "upper_joint_context_free": "context_free",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run S3 Task-STL lower/upper bounds. "
            "lower_seq_ft is sequential fine-tuning without CL; "
            "upper_joint_bilstm is joint multi-task training with all task data available."
        )
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dialogue_task_stl_v2.yaml"))
    parser.add_argument(
        "--bounds",
        nargs="*",
        choices=sorted(BOUND_METHODS),
        default=["lower_seq_ft", "upper_joint_bilstm"],
    )
    parser.add_argument("--run-name", default="s3_task_bounds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-id", default=None, help="Physical GPU id to expose, e.g. 8. Overrides --device to cuda.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda"

    base_overrides = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": args.device,
    }
    if args.fp16:
        base_overrides["fp16"] = True
    if args.no_fp16:
        base_overrides["fp16"] = False

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        overrides = dict(base_overrides)
        overrides["seed"] = seed
        for bound_name in args.bounds:
            method = BOUND_METHODS[bound_name]
            run_name_parts = [args.run_name, bound_name]
            if seed is not None:
                run_name_parts.append(f"seed{seed}")
            result_path = run_dialogue_text_task_experiment(
                args.config,
                method,
                run_name="_".join(run_name_parts),
                train_overrides=overrides,
            )
            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"{bound_name} ({method}){seed_label}: {result_path}")


if __name__ == "__main__":
    main()
