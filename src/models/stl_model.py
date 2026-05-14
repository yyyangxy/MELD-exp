from __future__ import annotations

import torch
from torch import nn

from .encoders import SpeakerEncoder, TextEncoder
from .fusion import ConcatMLPFusion


TASK_NUM_LABELS = {
    "sentiment": 3,
    "emotion": 7,
    "shift": 2,
}


class MultiTaskSTLModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_speakers: int,
        config: dict,
        task_num_labels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        task_num_labels = task_num_labels or TASK_NUM_LABELS

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            use_lstm=bool(model_cfg.get("use_lstm", True)),
        )

        self.use_speaker_embedding = bool(model_cfg.get("speaker_embedding", True))
        if self.use_speaker_embedding:
            self.speaker_encoder = SpeakerEncoder(
                num_speakers=num_speakers,
                embedding_dim=int(model_cfg.get("speaker_embedding_dim", 32)),
            )
            fusion_input_dim = self.text_encoder.output_dim + self.speaker_encoder.output_dim
        else:
            self.speaker_encoder = None
            fusion_input_dim = self.text_encoder.output_dim

        self.fusion = ConcatMLPFusion(
            input_dim=fusion_input_dim,
            hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
            output_dim=int(model_cfg.get("fusion_output_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Linear(self.fusion.output_dim, num_labels)
                for task_name, num_labels in task_num_labels.items()
            }
        )

    def forward(self, batch: dict, task_name: str | None = None) -> dict[str, torch.Tensor]:
        task_name = task_name or batch["task_name"]
        text_embedding = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        features = [text_embedding]
        if self.speaker_encoder is not None:
            features.append(self.speaker_encoder(batch["speaker_id"]))
        embedding = self.fusion(torch.cat(features, dim=-1))
        logits = self.heads[task_name](embedding)
        return {"logits": logits, "embedding": embedding}

