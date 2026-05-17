from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.data.meld_csv import read_all_splits
from src.data.task_builder import build_task_examples
from src.utils.paths import load_config, resolve_data_root


EXPECTED_COUNTS = {"train": 9989, "dev": 1109, "test": 2610}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dependency-light MELD data checks.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    split_records = read_all_splits(data_root, warn_missing_videos=False)

    for split, records in split_records.items():
        expected = EXPECTED_COUNTS[split]
        actual = len(records)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{split}: rows={actual} expected={expected} {status}")
        if actual != expected:
            raise SystemExit(1)

        sentiment = build_task_examples(records, "sentiment", context_window=3)
        emotion = build_task_examples(records, "emotion", context_window=3)
        shift = build_task_examples(records, "shift", context_window=3)
        shift_counts = Counter(example.label_name for example in shift)
        print(
            f"{split}: sentiment={len(sentiment)} emotion={len(emotion)} "
            f"shift={len(shift)} shift_counts={dict(shift_counts)}"
        )


if __name__ == "__main__":
    main()
