from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.data.meld_csv import UtteranceRecord
from src.features.feature_store import feature_path, save_feature


LOGGER = logging.getLogger(__name__)


def extract_opensmile_audio_features(
    split_records: dict[str, list[UtteranceRecord]],
    output_root: str | Path,
    sample_rate: int = 16000,
    skip_existing: bool = True,
) -> int:
    """Extract openSMILE ComParE 2016 functionals, 6373 dim per utterance."""
    try:
        import opensmile
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openSMILE audio feature extraction requires the Python package 'opensmile'. "
            "Install it with `pip install opensmile` in the experiment environment."
        ) from exc

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("openSMILE extraction requires ffmpeg on PATH to decode MP4 audio.")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    count = 0
    skipped_missing = 0
    with tempfile.TemporaryDirectory(prefix="meld_opensmile_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for split, records in split_records.items():
            for record in records:
                out_path = feature_path(output_root, split, "audio", record.utterance_key)
                if skip_existing and out_path.exists():
                    count += 1
                    continue
                if not record.video_exists:
                    skipped_missing += 1
                    continue

                wav_path = tmp_root / f"{record.utterance_key}.wav"
                if not _extract_wav(record.video_path, wav_path, sample_rate):
                    continue

                try:
                    frame = smile.process_file(str(wav_path))
                except Exception as exc:  # pragma: no cover - depends on server codecs.
                    LOGGER.warning("Skipping openSMILE for %s: %s", record.utterance_key, exc)
                    continue
                if frame.empty:
                    LOGGER.warning("Skipping openSMILE for %s: empty feature frame", record.utterance_key)
                    continue

                feature = frame.iloc[0].to_numpy(dtype="float32")
                if int(feature.shape[0]) != 6373:
                    raise ValueError(
                        f"openSMILE feature dimension mismatch for {record.utterance_key}: "
                        f"expected 6373, got {feature.shape[0]}"
                    )
                save_feature(output_root, split, "audio", record.utterance_key, feature)
                count += 1

    if skipped_missing:
        LOGGER.warning("Skipped %s openSMILE features because source videos were missing.", skipped_missing)
    return count


def _extract_wav(video_path: Path, wav_path: Path, sample_rate: int) -> bool:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        LOGGER.warning(
            "ffmpeg failed for %s: %s",
            video_path,
            result.stderr.decode("utf-8", errors="replace").strip(),
        )
        return False
    return True
