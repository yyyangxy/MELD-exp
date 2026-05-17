from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.modality_runner import MODALITY_METHODS, run_modality_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MELD modality-sequential baselines.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "modality_stl.yaml"))
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(MODALITY_METHODS),
        help="Modality-sequential method to run.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. When provided, results go under outputs/runs/<group>/<timestamp>_<run-name>.",
    )
    args = parser.parse_args()

    result_path = run_modality_experiment(args.config, args.method, run_name=args.run_name)
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
