from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.data.meld_csv import read_all_splits
from src.data.task_builder import build_task_examples
from src.utils.paths import ensure_dir, load_config, resolve_data_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write same-speaker emotion shift labels.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "shift_labels"))
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    output_dir = ensure_dir(resolve_path(args.output_dir, PROJECT_ROOT))
    split_records = read_all_splits(data_root, warn_missing_videos=False)

    for split, records in split_records.items():
        examples = build_task_examples(records, "shift", context_window=0)
        path = output_dir / f"{split}_shift_labels.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "utterance_key",
                    "dialogue_id",
                    "utterance_id",
                    "speaker",
                    "emotion",
                    "previous_same_speaker_emotion",
                    "label",
                    "label_name",
                ],
            )
            writer.writeheader()
            for example in examples:
                writer.writerow(
                    {
                        "utterance_key": example.utterance_key,
                        "dialogue_id": example.dialogue_id,
                        "utterance_id": example.utterance_id,
                        "speaker": example.speaker,
                        "emotion": example.meta["emotion"],
                        "previous_same_speaker_emotion": example.meta[
                            "previous_same_speaker_emotion"
                        ],
                        "label": example.label,
                        "label_name": example.label_name,
                    }
                )
        print(f"{split}: {len(examples)} labels -> {path}")


if __name__ == "__main__":
    main()
