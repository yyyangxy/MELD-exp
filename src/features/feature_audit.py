from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.meld_csv import UtteranceRecord
from src.features.feature_store import feature_path


@dataclass
class FeatureAuditRow:
    split: str
    modality: str
    expected_count: int
    found_count: int
    missing_count: int
    expected_dim: int
    bad_shape_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "modality": self.modality,
            "expected_count": self.expected_count,
            "found_count": self.found_count,
            "missing_count": self.missing_count,
            "expected_dim": self.expected_dim,
            "bad_shape_count": self.bad_shape_count,
        }


def audit_features(
    split_records: dict[str, list[UtteranceRecord]],
    feature_root: str | Path,
    feature_dims: dict[str, int],
    modalities: list[str],
) -> list[FeatureAuditRow]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to audit .npy feature caches") from exc

    rows: list[FeatureAuditRow] = []
    for split, records in split_records.items():
        for modality in modalities:
            expected_dim = int(feature_dims[modality])
            found_count = 0
            bad_shape_count = 0
            for record in records:
                path = feature_path(feature_root, split, modality, record.utterance_key)
                if not path.exists():
                    continue
                found_count += 1
                try:
                    array = np.load(path).reshape(-1)
                except Exception:
                    bad_shape_count += 1
                    continue
                if int(array.shape[0]) != expected_dim:
                    bad_shape_count += 1
            expected_count = len(records)
            rows.append(
                FeatureAuditRow(
                    split=split,
                    modality=modality,
                    expected_count=expected_count,
                    found_count=found_count,
                    missing_count=expected_count - found_count,
                    expected_dim=expected_dim,
                    bad_shape_count=bad_shape_count,
                )
            )
    return rows
