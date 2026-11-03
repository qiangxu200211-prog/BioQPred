#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NeuGSE-style 深度 DNA 序列编码 baseline。

该模块不是复现 NeuGSE 原论文数据集和完整内部消融，而是在本项目统一
DNA 功能预测任务上实现一个符合其核心思想的外部 baseline：
核苷酸编码 -> 位置编码 -> Conv1D 局部特征 -> 1D 双注意力 -> 空间金字塔池化 -> 分类器。
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """标准正弦位置编码。

    输入 embedding 的形状为 [batch, length, dim]，位置编码会按序列长度切片后相加。
    使用 buffer 保存编码矩阵，因此它不参与训练，也不会增加可训练参数。
    """

    def __init__(self, max_len: int, embedding_dim: int):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if embedding_dim > 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """给 token embedding 加入位置编码。"""

        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class CBAM1D(nn.Module):
    """面向一维 DNA 序列特征的 CBAM-style 双注意力模块。

    先做 channel attention，让模型选择更重要的卷积通道；再做 spatial attention，
    让模型在序列位置维度上突出更有判别力的局部区域。
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入和输出形状均为 [batch, channels, length]。"""

        avg_pool = torch.mean(x, dim=-1)
        max_pool = torch.amax(x, dim=-1)
        channel_weights = self.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)).unsqueeze(-1)
        x = x * channel_weights

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        spatial_weights = self.sigmoid(self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1)))
        return x * spatial_weights


class SpatialPyramidPooling1D(nn.Module):
    """一维空间金字塔池化。

    对序列维度做多个尺度的自适应 max/avg pooling，随后展平拼接。这样既保留
    全局信息，也保留粗粒度局部布局信息。
    """

    def __init__(self, pool_levels: Iterable[int]):
        super().__init__()
        self.pool_levels: List[int] = [int(level) for level in pool_levels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [batch, channels, length]，输出 [batch, channels * sum(levels) * 2]。"""

        features = []
        for level in self.pool_levels:
            features.append(nn.functional.adaptive_max_pool1d(x, level).flatten(start_dim=1))
            features.append(nn.functional.adaptive_avg_pool1d(x, level).flatten(start_dim=1))
        return torch.cat(features, dim=1)


class NeuGSEStyleClassifier(nn.Module):
    """NeuGSE-style DNA 序列分类器。

    该模型不使用 DNABERT-2，参数全部从当前任务的 train split 学习，用作 recent
    IEEE deep sequence encoding 类 baseline。
    """

    def __init__(
        self,
        num_classes: int,
        max_len: int,
        embedding_dim: int = 32,
        conv_channels: int = 256,
        pool_levels: Iterable[int] = (1, 2, 4),
        cbam_reduction: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        if conv_channels < 256:
            raise ValueError("conv_channels 建议不低于 256，以匹配 NeuGSE-style 默认结构。")
        self.num_classes = int(num_classes)
        self.max_len = int(max_len)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim, padding_idx=4)
        self.position = SinusoidalPositionalEncoding(max_len=max_len, embedding_dim=embedding_dim)

        # 三层 Conv1D 局部编码器：前两层 128 通道，最后一层输出 conv_channels。
        self.local_encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.attention = CBAM1D(conv_channels, reduction=cbam_reduction)
        self.spp = SpatialPyramidPooling1D(pool_levels)
        spp_dim = conv_channels * sum(int(level) for level in pool_levels) * 2
        self.classifier = nn.Sequential(
            nn.Linear(spp_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数 input_ids 为 [batch, length]，取值范围 0-4，其中 4 表示 N/unknown/padding。
        """

        x = self.embedding(input_ids)
        x = self.position(x)
        x = x.transpose(1, 2)
        x = self.local_encoder(x)
        x = self.attention(x)
        x = self.spp(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> dict[str, float | int]:
    """统计模型总参数、可训练参数和可训练比例。"""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
    }
