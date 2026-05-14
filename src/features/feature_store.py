from __future__ import annotations

from pathlib import Path


def feature_path(output_root: str | Path, split: str, modality: str, utterance_key: str) -> Path:
    return Path(output_root) / "features" / split / modality / f"{utterance_key}.npy"


def save_feature(output_root: str | Path, split: str, modality: str, utterance_key: str, values) -> Path:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to write .npy feature caches") from exc

    path = feature_path(output_root, split, modality, utterance_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype="float32"))
    return path


def load_feature(path: str | Path):
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to read .npy feature caches") from exc
    return np.load(Path(path))

