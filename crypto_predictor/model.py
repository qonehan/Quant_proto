"""
비트코인 기울기 예측 모델.

아키텍처: LSTM + Temporal Attention + FC Head
  Input:  (batch, n, num_features)
  Output: (batch, 1) — 예측 기울기 (slope)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """시퀀스의 각 타임스텝에 대한 어텐션 가중치를 학습."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)
        return context


class SlopePredictorLSTM(nn.Module):
    """LSTM 기반 기울기 예측 모델."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(num_features)

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        effective_hidden = hidden_size * (2 if bidirectional else 1)

        self.attention = TemporalAttention(effective_hidden)

        self.head = nn.Sequential(
            nn.Linear(effective_hidden, effective_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(effective_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.head(context)
        return out.squeeze(-1)


def create_model(num_features: int, params) -> SlopePredictorLSTM:
    """HyperParams에서 모델을 생성한다."""
    return SlopePredictorLSTM(
        num_features=num_features,
        hidden_size=params.hidden_size,
        num_layers=params.num_layers,
        dropout=params.dropout,
        bidirectional=params.bidirectional,
    )
