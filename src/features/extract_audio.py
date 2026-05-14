from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from src.data.meld_csv import UtteranceRecord
from src.features.feature_store import feature_path, save_feature


LOGGER = logging.getLogger(__name__)
DEFAULT_AUDIO_MODEL = "facebook/wav2vec2-base"


def extract_audio_features(
    split_records: dict[str, list[UtteranceRecord]],
    output_root: str | Path,
    model_path: str | Path,
    device: str = "auto",
    sample_rate: int = 16000,
    skip_existing: bool = True,
) -> int:
    """Extract one Wav2Vec2 mean-pooled feature per MELD utterance video."""
    try:
        import numpy as np
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Audio feature extraction requires numpy, transformers, and torch. "
            "Install requirements.txt in the server environment."
        ) from exc

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Audio feature extraction requires ffmpeg on PATH to decode MP4 audio.")

    resolved_device = _resolve_device(device, torch)
    model_name_or_path = str(model_path) or DEFAULT_AUDIO_MODEL
    model_ref = Path(model_name_or_path).expanduser() if _looks_like_path(model_name_or_path) else model_name_or_path
    local_only = isinstance(model_ref, Path) and model_ref.exists()

    processor = AutoFeatureExtractor.from_pretrained(
        model_ref,
        local_files_only=local_only,
    )
    model = Wav2Vec2Model.from_pretrained(
        model_ref,
        local_files_only=local_only,
    ).to(resolved_device)
    model.eval()

    count = 0
    skipped_missing = 0
    for split, records in split_records.items():
        for record in records:
            out_path = feature_path(output_root, split, "audio", record.utterance_key)
            if skip_existing and out_path.exists():
                count += 1
                continue
            if not record.video_exists:
                skipped_missing += 1
                continue

            waveform = _load_audio_with_ffmpeg(record.video_path, sample_rate, np)
            if waveform is None:
                LOGGER.warning("Skipping audio for %s: failed to decode %s", record.utterance_key, record.video_path)
                continue

            if waveform.size == 0:
                LOGGER.warning("Skipping audio for %s: empty waveform in %s", record.utterance_key, record.video_path)
                continue

            inputs = processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=False,
            )
            inputs = {
                key: value.to(resolved_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
                feature = hidden.mean(dim=1).squeeze(0).cpu().numpy()

            save_feature(output_root, split, "audio", record.utterance_key, feature)
            count += 1

    if skipped_missing:
        LOGGER.warning("Skipped %s audio features because source videos were missing.", skipped_missing)
    return count


def _resolve_device(device: str, torch_module) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~"))


def _load_audio_with_ffmpeg(video_path: Path, sample_rate: int, np_module):
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        LOGGER.warning(
            "ffmpeg failed for %s: %s",
            video_path,
            result.stderr.decode("utf-8", errors="replace").strip(),
        )
        return None
    return np_module.frombuffer(result.stdout, dtype=np_module.float32).copy()
