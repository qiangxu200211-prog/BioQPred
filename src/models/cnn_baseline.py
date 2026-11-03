#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简单 1D CNN baseline。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def one_hot_encode_sequences(sequences: Sequence[str], max_length: int | None = None) -> torch.Tensor:
    """将 A/C/G/T/N 序列编码为 [batch, channels=5, length] one-hot 张量。"""

    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)
    tensor = torch.zeros((len(sequences), 5, max_length), dtype=torch.float32)
    for row_index, sequence in enumerate(sequences):
        sequence = str(sequence).upper()[:max_length]
        for col_index, base in enumerate(sequence):
            tensor[row_index, BASE_TO_INDEX.get(base, 4), col_index] = 1.0
    return tensor


class SimpleCNNBaseline(nn.Module):
    """用于普通深度学习 baseline 的小型 1D CNN。"""

    def __init__(self, num_classes: int, channels: int = 96, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        return self.classifier(features)


def cnn_collate_fn(batch: Sequence[Dict[str, object]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """DataLoader collate_fn：返回 one-hot、label、id。"""

    sequences = [str(item["sequence"]) for item in batch]
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    ids = [str(item["id"]) for item in batch]
    return one_hot_encode_sequences(sequences), labels, ids
