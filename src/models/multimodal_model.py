from __future__ import annotations

import torch
from torch import nn

from .encoders import SpeakerEncoder
from .fusion import ConcatMLPFusion
from .stl_model import TASK_NUM_LABELS


class MultimodalSTLModel(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        num_speakers: int,
        config: dict,
        task_num_labels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        modality_cfg = config.get("modalities", {})
        task_num_labels = task_num_labels or TASK_NUM_LABELS

        self.modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
        self.modality_to_index = {modality: idx for idx, modality in enumerate(self.modalities)}
        projection_dim = int(model_cfg.get("modality_projection_dim", 128))
        dropout = float(model_cfg.get("dropout", 0.2))
        negative_slope = float(model_cfg.get("modality_activation_negative_slope", 0.01))
        normalize_inputs = bool(model_cfg.get("normalize_modality_inputs", True))

        self.modality_encoders = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    *(
                        [nn.LayerNorm(int(feature_dims[modality]))]
                        if normalize_inputs
                        else []
                    ),
                    nn.Linear(int(feature_dims[modality]), projection_dim),
                    nn.LeakyReLU(negative_slope=negative_slope),
                    nn.Dropout(dropout),
                )
                for modality in self.modalities
            }
        )

        self.use_speaker_embedding = bool(model_cfg.get("speaker_embedding", True))
        fusion_input_dim = projection_dim * len(self.modalities)
        if self.use_speaker_embedding:
            self.speaker_encoder = SpeakerEncoder(
                num_speakers=num_speakers,
                embedding_dim=int(model_cfg.get("speaker_embedding_dim", 32)),
            )
            fusion_input_dim += self.speaker_encoder.output_dim
        else:
            self.speaker_encoder = None

        self.fusion = ConcatMLPFusion(
            input_dim=fusion_input_dim,
            hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
            output_dim=int(model_cfg.get("fusion_output_dim", 256)),
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Linear(self.fusion.output_dim, num_labels)
                for task_name, num_labels in task_num_labels.items()
            }
        )

    def forward(
        self,
        batch: dict,
        task_name: str | None = None,
        active_modalities: list[str] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        task_name = task_name or batch["task_name"]
        active_set = set(active_modalities or batch.get("active_modalities", self.modalities))
        mask = batch.get("modality_mask")
        modality_embeddings: dict[str, torch.Tensor] = {}
        fused_inputs: list[torch.Tensor] = []

        for modality in self.modalities:
            encoded = self.modality_encoders[modality](batch["features"][modality])
            if modality not in active_set:
                encoded = torch.zeros_like(encoded)
            elif mask is not None:
                encoded = encoded * mask[:, self.modality_to_index[modality]].unsqueeze(-1)
            modality_embeddings[modality] = encoded
            fused_inputs.append(encoded)

        if self.speaker_encoder is not None:
            fused_inputs.append(self.speaker_encoder(batch["speaker_id"]))

        embedding = self.fusion(torch.cat(fused_inputs, dim=-1))
        logits = self.heads[task_name](embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "modality_embeddings": modality_embeddings,
        }
