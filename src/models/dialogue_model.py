from __future__ import annotations

import torch
from torch import nn

from src.models.encoders import SpeakerEncoder
from src.models.fusion import ConcatMLPFusion
from src.models.stl_model import TASK_NUM_LABELS


class DialogueMultimodalSTLModel(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        num_speakers: int,
        config: dict,
        task_num_labels: dict[str, int] | None = None,
        use_dialogue_encoder: bool | None = None,
    ) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        modality_cfg = config.get("modalities", {})
        task_num_labels = task_num_labels or TASK_NUM_LABELS

        self.modalities = list(modality_cfg.get("order", ["text", "audio", "visual"]))
        self.modality_to_index = {modality: idx for idx, modality in enumerate(self.modalities)}
        projection_dim = int(model_cfg.get("modality_projection_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.3))
        negative_slope = float(model_cfg.get("modality_activation_negative_slope", 0.01))
        normalize_inputs = bool(model_cfg.get("normalize_modality_inputs", True))

        self.modality_encoders = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    *([nn.LayerNorm(int(feature_dims[modality]))] if normalize_inputs else []),
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

        self.use_dialogue_encoder = (
            bool(model_cfg.get("dialogue_encoder", True))
            if use_dialogue_encoder is None
            else use_dialogue_encoder
        )
        fusion_output_dim = self.fusion.output_dim
        if self.use_dialogue_encoder:
            hidden_dim = int(model_cfg.get("dialogue_hidden_dim", 256))
            num_layers = int(model_cfg.get("dialogue_num_layers", 2))
            self.dialogue_encoder = nn.LSTM(
                input_size=fusion_output_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            head_input_dim = hidden_dim * 2
        else:
            self.dialogue_encoder = None
            head_input_dim = fusion_output_dim

        self.heads = nn.ModuleDict(
            {
                task_name: nn.Linear(head_input_dim, num_labels)
                for task_name, num_labels in task_num_labels.items()
            }
        )

    def forward(
        self,
        batch: dict,
        task_name: str | None = None,
        active_modalities: list[str] | None = None,
        head_name: str | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        task_name = task_name or batch.get("task_name", "emotion")
        head_name = head_name or task_name
        active_set = set(active_modalities or batch.get("active_modalities", self.modalities))
        modality_mask = batch.get("modality_mask")
        fused_inputs: list[torch.Tensor] = []
        modality_embeddings: dict[str, torch.Tensor] = {}

        batch_size, max_len = batch["speaker_id"].shape
        for modality in self.modalities:
            values = batch["features"][modality]
            encoded = self.modality_encoders[modality](values.reshape(batch_size * max_len, -1))
            encoded = encoded.reshape(batch_size, max_len, -1)
            if modality not in active_set:
                encoded = torch.zeros_like(encoded)
            elif modality_mask is not None:
                mask = modality_mask[:, self.modality_to_index[modality]].view(batch_size, 1, 1)
                encoded = encoded * mask
            modality_embeddings[modality] = encoded
            fused_inputs.append(encoded)

        if self.speaker_encoder is not None:
            speaker_flat = batch["speaker_id"].reshape(batch_size * max_len)
            speaker_emb = self.speaker_encoder(speaker_flat).reshape(batch_size, max_len, -1)
            fused_inputs.append(speaker_emb)

        fused = torch.cat(fused_inputs, dim=-1).reshape(batch_size * max_len, -1)
        utterance_embedding = self.fusion(fused).reshape(batch_size, max_len, -1)
        sequence_embedding = utterance_embedding

        if self.dialogue_encoder is not None:
            lengths = batch["lengths"].detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                utterance_embedding,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            encoded, _ = self.dialogue_encoder(packed)
            sequence_embedding, _ = nn.utils.rnn.pad_packed_sequence(
                encoded,
                batch_first=True,
                total_length=max_len,
            )

        logits = self.heads[head_name](sequence_embedding)
        return {
            "logits": logits,
            "embedding": sequence_embedding,
            "utterance_embedding": utterance_embedding,
            "modality_embeddings": modality_embeddings,
        }
