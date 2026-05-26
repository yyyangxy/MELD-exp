from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.meld_csv import read_all_splits
from src.data.stl_task_splits import resolve_stl_task_split_root, write_stl_task_splits
from src.utils.paths import load_config, resolve_data_root, resolve_path


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed, disjoint dialogue-id splits for MELD Task-STL.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl_v2.yaml"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    data_root = resolve_data_root(config)
    task_order = list(config.get("tasks", {}).get("order", ["sentiment", "emotion", "shift"]))
    seed = int(args.seed if args.seed is not None else config.get("seed", 13))

    if args.output_root:
        output_root = resolve_path(args.output_root, data_root)
    else:
        output_root = resolve_stl_task_split_root(data_cfg, data_root) or data_root / "stl_task_splits"

    split_records = read_all_splits(
        data_root,
        warn_missing_videos=bool(data_cfg.get("warn_missing_videos", True)),
    )
    written_root = write_stl_task_splits(split_records, output_root, task_order=task_order, seed=seed)
    LOGGER.info("Wrote STL task splits to %s", written_root)


if __name__ == "__main__":
    main()
