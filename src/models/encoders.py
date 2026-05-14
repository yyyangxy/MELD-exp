from __future__ import annotations

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        padding_idx: int = 0,
        use_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.use_lstm = use_lstm
        if use_lstm:
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.output_dim = hidden_dim * 2
        else:
            self.encoder = None
            self.output_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        lengths = attention_mask.sum(dim=1).clamp(min=1).long().cpu()

        if self.encoder is None:
            masked = embedded * attention_mask.unsqueeze(-1)
            return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        return self.dropout(torch.cat([forward_last, backward_last], dim=-1))


class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_speakers + 1, embedding_dim, padding_idx=0)
        self.output_dim = embedding_dim

    def forward(self, speaker_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(speaker_id)

