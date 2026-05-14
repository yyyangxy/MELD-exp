from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.meld_csv import read_all_splits
from src.features.extract_audio import extract_audio_features
from src.features.extract_text import extract_hash_text_features
from src.features.extract_visual import extract_visual_features
from src.utils.paths import load_config, resolve_data_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MELD baseline feature caches.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "main_stl.yaml"))
    parser.add_argument("--modalities", nargs="+", default=["text"])
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--device", default="auto", help="Device for neural feature extractors.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--visual-num-frames", type=int, default=16)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Debug mode: keep only the first N utterances from each split; 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even when the output .npy already exists.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    output_dir = resolve_path(config.get("train", {}).get("output_dir", "outputs"), PROJECT_ROOT)
    split_records = read_all_splits(data_root, warn_missing_videos=False)
    split_records = _limit_records(split_records, args.limit_per_split)
    feature_paths = config.get("feature_paths", {})
    skip_existing = not args.overwrite

    for modality in args.modalities:
        if modality == "text":
            count = extract_hash_text_features(
                split_records,
                output_dir,
                dim=args.text_dim,
                skip_existing=skip_existing,
            )
            print(f"text: wrote or reused {count} .npy feature files under {output_dir / 'features'}")
        elif modality == "audio":
            count = extract_audio_features(
                split_records,
                output_dir,
                model_path=feature_paths.get("audio_model_path", ""),
                device=args.device,
                sample_rate=args.audio_sample_rate,
                skip_existing=skip_existing,
            )
            print(f"audio: wrote or reused {count} .npy feature files under {output_dir / 'features'}")
        elif modality == "visual":
            count = extract_visual_features(
                split_records,
                output_dir,
                model_path=feature_paths.get("visual_model_path", ""),
                device=args.device,
                num_frames=args.visual_num_frames,
                skip_existing=skip_existing,
            )
            print(f"visual: wrote or reused {count} .npy feature files under {output_dir / 'features'}")
        else:
            raise ValueError(f"Unknown modality: {modality}")


def _limit_records(records_by_split, limit_per_split: int):
    if limit_per_split <= 0:
        return records_by_split
    return {
        split: records[:limit_per_split]
        for split, records in records_by_split.items()
    }


if __name__ == "__main__":
    main()
