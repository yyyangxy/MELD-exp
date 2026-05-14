from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)

SPLIT_TO_CSV = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

SPLIT_TO_VIDEO_DIR = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}

SENTIMENT_LABELS = {"negative": 0, "neutral": 1, "positive": 2}
EMOTION_LABELS = {
    "neutral": 0,
    "joy": 1,
    "surprise": 2,
    "anger": 3,
    "sadness": 4,
    "disgust": 5,
    "fear": 6,
}
SHIFT_LABELS = {"no_shift": 0, "shift": 1}

TASK_LABELS = {
    "sentiment": SENTIMENT_LABELS,
    "emotion": EMOTION_LABELS,
    "shift": SHIFT_LABELS,
}


@dataclass(frozen=True)
class UtteranceRecord:
    split: str
    sr_no: int
    utterance: str
    speaker: str
    emotion: str
    sentiment: str
    dialogue_id: int
    utterance_id: int
    season: str
    episode: str
    start_time: str
    end_time: str
    utterance_key: str
    video_path: Path
    video_exists: bool


def utterance_key(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}"


def video_path_for(data_root: Path, split: str, dialogue_id: int, utterance_id: int) -> Path:
    return data_root / SPLIT_TO_VIDEO_DIR[split] / f"{utterance_key(dialogue_id, utterance_id)}.mp4"


def read_split_csv(
    data_root: str | Path,
    split: str,
    warn_missing_videos: bool = False,
) -> list[UtteranceRecord]:
    if split not in SPLIT_TO_CSV:
        raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(SPLIT_TO_CSV)}")

    root = Path(data_root)
    csv_path = root / SPLIT_TO_CSV[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    records: list[UtteranceRecord] = []
    missing_videos = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])
            key = utterance_key(dialogue_id, utterance_id)
            video_path = video_path_for(root, split, dialogue_id, utterance_id)
            video_exists = video_path.exists()
            if not video_exists:
                missing_videos += 1
            records.append(
                UtteranceRecord(
                    split=split,
                    sr_no=int(row["Sr No."]),
                    utterance=row["Utterance"].strip(),
                    speaker=row["Speaker"].strip(),
                    emotion=row["Emotion"].strip().lower(),
                    sentiment=row["Sentiment"].strip().lower(),
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                    season=row.get("Season", "").strip(),
                    episode=row.get("Episode", "").strip(),
                    start_time=row.get("StartTime", "").strip(),
                    end_time=row.get("EndTime", "").strip(),
                    utterance_key=key,
                    video_path=video_path,
                    video_exists=video_exists,
                )
            )

    if warn_missing_videos and missing_videos:
        LOGGER.warning(
            "%s split has %s missing video files under %s; annotation-only runs continue.",
            split,
            missing_videos,
            root / SPLIT_TO_VIDEO_DIR[split],
        )
    return records


def read_all_splits(
    data_root: str | Path,
    splits: Iterable[str] = ("train", "dev", "test"),
    warn_missing_videos: bool = False,
) -> dict[str, list[UtteranceRecord]]:
    return {
        split: read_split_csv(data_root, split, warn_missing_videos=warn_missing_videos)
        for split in splits
    }

