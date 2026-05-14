from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.joint_runner import run_joint_experiment
from src.train.sequential_runner import run_sequential_experiment


DEFAULT_METHODS = [
    "joint",
    "seq_ft",
    "lwf",
    "random_replay",
    "prototype_replay",
    "proto_replay_kd",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MELD main baselines.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl.yaml"))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    args = parser.parse_args()

    for method in args.methods:
        if method == "joint":
            result_path = run_joint_experiment(args.config)
        else:
            result_path = run_sequential_experiment(args.config, method)
        print(f"{method}: {result_path}")


if __name__ == "__main__":
    main()

