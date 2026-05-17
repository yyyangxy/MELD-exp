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
from src.features.feature_audit import audit_features
from src.features.feature_store import feature_cache_root
from src.utils.paths import ensure_dir, load_config, resolve_data_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MELD .npy feature coverage and dimensions.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "modality_stl_v2.yaml"))
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    modalities_cfg = config.get("modalities", {})
    modalities = args.modalities or list(modalities_cfg.get("order", ["text", "audio", "visual"]))
    feature_dims = {key: int(value) for key, value in modalities_cfg.get("feature_dims", {}).items()}
    feature_root = _resolve_feature_root(config)

    missing_dims = [modality for modality in modalities if modality not in feature_dims]
    if missing_dims:
        raise ValueError(f"Missing feature dims in config for modalities: {missing_dims}")

    split_records = read_all_splits(data_root, warn_missing_videos=False)
    rows = audit_features(split_records, feature_root, feature_dims, modalities)

    fieldnames = [
        "split",
        "modality",
        "expected_count",
        "found_count",
        "missing_count",
        "expected_dim",
        "bad_shape_count",
    ]
    for row in rows:
        print(row.as_dict())

    output_path = Path(args.output) if args.output else _default_output_path(config)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    print(f"wrote audit CSV to {output_path}")
    print(f"audited feature cache under {feature_cache_root(feature_root)}")


def _resolve_feature_root(config: dict) -> Path:
    feature_paths = config.get("feature_paths", {})
    if feature_paths.get("feature_root"):
        return resolve_path(feature_paths["feature_root"], PROJECT_ROOT)
    modalities = config.get("modalities", {})
    if modalities.get("feature_root"):
        return resolve_path(modalities["feature_root"], PROJECT_ROOT)
    return resolve_path(config.get("train", {}).get("output_dir", "outputs"), PROJECT_ROOT)


def _default_output_path(config: dict) -> Path:
    output_root = resolve_path(config.get("train", {}).get("output_dir", "outputs"), PROJECT_ROOT)
    return output_root / "results" / "feature_audit.csv"


if __name__ == "__main__":
    main()
