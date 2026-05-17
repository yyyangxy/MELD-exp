from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.joint_runner import run_joint_experiment
from src.train.sequential_runner import SEQUENTIAL_METHODS, run_sequential_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MELD sequence task learning baselines.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl.yaml"))
    parser.add_argument(
        "--method",
        required=True,
        choices=["joint", *sorted(SEQUENTIAL_METHODS)],
        help="Baseline method to run.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. When provided, results go under outputs/runs/<group>/<timestamp>_<run-name>.",
    )
    args = parser.parse_args()

    if args.method == "joint":
        result_path = run_joint_experiment(args.config, run_name=args.run_name)
    else:
        result_path = run_sequential_experiment(args.config, args.method, run_name=args.run_name)
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
