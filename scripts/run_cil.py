from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()


def main() -> None:
    print("Class-incremental MELD experiments are not part of the baseline implementation yet.")


if __name__ == "__main__":
    main()
