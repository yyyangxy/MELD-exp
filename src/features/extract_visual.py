from __future__ import annotations

import logging
from pathlib import Path

from src.data.meld_csv import UtteranceRecord
from src.features.feature_store import feature_path, save_feature


LOGGER = logging.getLogger(__name__)
DEFAULT_VISUAL_MODEL = "torchvision:resnet50"


def extract_visual_features(
    split_records: dict[str, list[UtteranceRecord]],
    output_root: str | Path,
    model_path: str | Path,
    device: str = "auto",
    num_frames: int = 16,
    resize_size: int = 256,
    crop_size: int = 224,
    skip_existing: bool = True,
) -> int:
    """Extract one ResNet50 mean-pooled frame feature per MELD utterance video."""
    try:
        import torch
        import torch.nn.functional as F
        from torchvision.io import read_video
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Visual feature extraction requires torchvision and PyAV. "
            "Install requirements.txt in the server environment."
        ) from exc

    resolved_device = _resolve_device(device, torch)
    model = _load_resnet50_feature_extractor(model_path, resolved_device, torch)

    count = 0
    skipped_missing = 0
    for split, records in split_records.items():
        for record in records:
            out_path = feature_path(output_root, split, "visual", record.utterance_key)
            if skip_existing and out_path.exists():
                count += 1
                continue
            if not record.video_exists:
                skipped_missing += 1
                continue

            try:
                frames, _, _ = read_video(str(record.video_path), pts_unit="sec", output_format="TCHW")
            except Exception as exc:  # pragma: no cover - depends on server codecs.
                LOGGER.warning("Skipping visual for %s: failed to load %s (%s)", record.utterance_key, record.video_path, exc)
                continue

            if frames.numel() == 0 or int(frames.shape[0]) == 0:
                LOGGER.warning("Skipping visual for %s: no frames in %s", record.utterance_key, record.video_path)
                continue

            frames = _sample_frames(frames, num_frames, torch).float().div(255.0)
            frames = F.interpolate(
                frames,
                size=(resize_size, resize_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            frames = _center_crop(frames, crop_size)
            frames = _normalize_imagenet(frames, torch)

            with torch.no_grad():
                feature = model(frames.to(resolved_device)).flatten(1).mean(dim=0).cpu().numpy()

            save_feature(output_root, split, "visual", record.utterance_key, feature)
            count += 1

    if skipped_missing:
        LOGGER.warning("Skipped %s visual features because source videos were missing.", skipped_missing)
    return count


def _load_resnet50_feature_extractor(model_path: str | Path, device: str, torch_module):
    try:
        from torch import nn
        from torchvision.models import ResNet50_Weights, resnet50
    except ModuleNotFoundError as exc:
        raise RuntimeError("Visual feature extraction requires torchvision.") from exc

    model_name_or_path = str(model_path) or DEFAULT_VISUAL_MODEL
    if model_name_or_path in {"torchvision:resnet50", "resnet50", "ResNet50_Weights.IMAGENET1K_V1"}:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    checkpoint_path = _find_checkpoint_path(Path(model_name_or_path).expanduser())
    model = resnet50(weights=None)
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported ResNet50 checkpoint format: {checkpoint_path}")
    state_dict = {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return nn.Sequential(*list(model.children())[:-1]).to(device).eval()


def _find_checkpoint_path(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    if not model_path.exists():
        raise FileNotFoundError(f"ResNet50 model path not found: {model_path}")
    for pattern in ("*.pth", "*.pt", "*.bin"):
        matches = sorted(model_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No ResNet50 checkpoint found under {model_path}. Expected .pth, .pt, or .bin."
    )


def _sample_frames(frames, num_frames: int, torch_module):
    if int(frames.shape[0]) <= num_frames:
        return frames
    indices = torch_module.linspace(0, int(frames.shape[0]) - 1, steps=num_frames).round().long()
    return frames.index_select(0, indices)


def _center_crop(frames, crop_size: int):
    _, _, height, width = frames.shape
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return frames[:, :, top : top + crop_size, left : left + crop_size]


def _normalize_imagenet(frames, torch_module):
    mean = torch_module.tensor([0.485, 0.456, 0.406], dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch_module.tensor([0.229, 0.224, 0.225], dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean) / std


def _resolve_device(device: str, torch_module) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device
