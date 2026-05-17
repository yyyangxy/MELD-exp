from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.modality_runner import MODALITY_METHODS, run_modality_experiment


DEFAULT_METHODS = [
    "mod_seq_ft",
    "mod_seq_kd",
    "prototype_replay",
    "cmcrd_ours",
    "utt_modality_sa_cmd",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full Modality-STL comparison suite.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "modality_stl_v2.yaml"))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - MODALITY_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Expected subset of {sorted(MODALITY_METHODS)}")

    for method in args.methods:
        result_path = run_modality_experiment(args.config, method, run_name=args.run_name)
        print(f"{method}: {result_path}")


if __name__ == "__main__":
    main()
