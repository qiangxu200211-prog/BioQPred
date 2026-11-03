#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TFBS-Finder-style baseline 模型。

参照 TFBS-Finder: Deep Learning-based Model with DNABERT and Convolutional
Networks to Predict Transcription Factor Binding Sites 的方法设计，
但不使用原论文 ENCODE 数据集，而是在本项目的统一数据集上做 method-level
reimplementation。

核心结构：
  pre-trained DNABERT embedding (DNABERT-2)
  +
  CNN module (两层 Conv1d)
  +
  Modified CBAM attention module (1D 版：Channel Attention + Spatial Attention)
  +
  Multi-Scale Convolutions with Attention (MSCA) module
  +
  output classifier

默认冻结 DNABERT-2 backbone（freeze_backbone=True），只训练 CNN/CBAM/MSCA/classifier。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import nn


class CNNModule(nn.Module):
    """CNN 模块：两层 Conv1d 对 DNABERT-2 token hidden states 做局部特征提取。

    输入：[B, L, H] DNABERT-2 hidden states
    转置为 [B, H, L] 后经过两层 Conv1d + BatchNorm + GELU + Dropout
    输出：[B, C, L] 其中 C = cnn_channels
    """

    def __init__(
        self,
        hidden_size: int = 768,
        cnn_channels: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # 第一层：kernel_size=7, padding=same(3)
        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_size, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 第二层：kernel_size=5, padding=same(2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            hidden_states: [B, L, H] DNABERT-2 输出的 token hidden states

        Returns:
            [B, C, L] CNN 精炼后的特征
        """
        # [B, L, H] → [B, H, L]
        x = hidden_states.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ChannelAttention1D(nn.Module):
    """1D 版 Channel Attention（CBAM 的通道注意力子模块）。

    对 [B, C, L] 做 AdaptiveAvgPool1d 和 AdaptiveMaxPool1d 得到 [B, C]，
    通过共享 MLP 学习通道间重要性，输出 Sigmoid 归一化的 [B, C, 1] 权重。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 1)
        # 共享 MLP：Linear → GELU → Linear
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.GELU(),
            nn.Linear(reduced, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: [B, C, L]

        Returns:
            [B, C, 1] channel attention weights
        """
        # AdaptiveAvgPool1d(1) → [B, C, 1] → squeeze → [B, C]
        avg_out = x.mean(dim=-1)
        # AdaptiveMaxPool1d(1) → [B, C, 1] → squeeze → [B, C]
        max_out = x.max(dim=-1)[0]
        # 共享 MLP，两支求和后 Sigmoid
        attn = torch.sigmoid(self.mlp(avg_out) + self.mlp(max_out))
        return attn.unsqueeze(-1)  # [B, C, 1]


class SpatialAttention1D(nn.Module):
    """1D 版 Spatial Attention（CBAM 的空间注意力子模块）。

    沿 channel 维求 avg 和 max 得到 [B, 2, L]，
    通过 Conv1d(2→1, k=7, padding=same) + Sigmoid 得到 [B, 1, L] 空间权重。
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: [B, C, L]

        Returns:
            [B, 1, L] spatial attention weights
        """
        # 沿 channel 维取 avg 和 max
        avg_out = x.mean(dim=1, keepdim=True)   # [B, 1, L]
        max_out = x.max(dim=1, keepdim=True)[0]  # [B, 1, L]
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, L]
        return torch.sigmoid(self.conv(concat))  # [B, 1, L]


class CBAM1D(nn.Module):
    """1D 版 CBAM（Convolutional Block Attention Module）。

    依次应用 Channel Attention 和 Spatial Attention，
    保留残差特性但按原论文方式顺序施加注意力。
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention1D(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention1D(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: [B, C, L]

        Returns:
            [B, C, L] 经过通道和空间注意力精炼的特征
        """
        # Channel Attention：逐通道加权
        ca = self.channel_attn(x)  # [B, C, 1]
        x = x * ca
        # Spatial Attention：逐位置加权
        sa = self.spatial_attn(x)  # [B, 1, L]
        x = x * sa
        return x


class MSCAModule(nn.Module):
    """Multi-Scale Convolutions with Attention 模块。

    使用多尺度卷积（kernel_size = 3, 5, 7, 11）提取不同粒度的局部模式，
    通过可学习的尺度注意力权重进行加权融合，
    最后用 GlobalMaxPool + GlobalAvgPool 得到序列级特征向量。
    """

    def __init__(
        self,
        in_channels: int,
        msca_channels: int = 128,
        kernel_sizes: Sequence[int] = (3, 5, 7, 11),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        # 每个尺度一个 Conv1d block
        self.convs = nn.ModuleList()
        for ks in self.kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, msca_channels, kernel_size=ks, padding=ks // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        # 尺度注意力：对每个尺度的 GlobalAvgPool 特征计算 softmax 权重
        self.scale_attn = nn.Sequential(
            nn.Linear(msca_channels * len(self.kernel_sizes), len(self.kernel_sizes)),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: [B, C, L] 来自 CBAM 的特征

        Returns:
            h_msca: [B, 2 * msca_channels] 序列级特征向量
        """
        # 各尺度卷积：每个得到 [B, msca_channels, L]
        scale_feats = [conv(x) for conv in self.convs]

        # 计算尺度注意力权重
        # 每个尺度做 GlobalAvgPool → [B, msca_channels]
        pooled = [feat.mean(dim=-1) for feat in scale_feats]
        concat_pooled = torch.cat(pooled, dim=-1)  # [B, msca_channels * num_scales]
        weights = self.scale_attn(concat_pooled)    # [B, num_scales]

        # 按尺度权重加权求和
        stacked = torch.stack(scale_feats, dim=1)  # [B, num_scales, msca_channels, L]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_scales, 1, 1]
        weighted = (weights_expanded * stacked).sum(dim=1)  # [B, msca_channels, L]

        # GlobalMaxPool + GlobalAvgPool → 拼接为序列级特征
        h_max = weighted.max(dim=-1)[0]   # [B, msca_channels]
        h_avg = weighted.mean(dim=-1)      # [B, msca_channels]
        h_msca = torch.cat([h_max, h_avg], dim=-1)  # [B, 2 * msca_channels]

        return h_msca


class TFBSFinderStyleClassifier(nn.Module):
    """TFBS-Finder-style 完整分类模型。

    结构：
      DNABERT-2 backbone → CNN module → CBAM → MSCA → output classifier

    默认冻结 DNABERT-2 backbone，只训练 CNN/CBAM/MSCA/classifier。
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = True,
        hidden_size: int = 768,
        cnn_channels: int = 128,
        msca_channels: int = 128,
        kernel_sizes: List[int] | None = None,
        cbam_reduction: int = 8,
        cnn_dropout: float = 0.2,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 11]

        self.backbone = backbone
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # 冻结 DNABERT-2 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 1. CNN module：两层 Conv1d 提取局部特征
        self.cnn_module = CNNModule(
            hidden_size=hidden_size,
            cnn_channels=cnn_channels,
            dropout=cnn_dropout,
        )

        # 2. CBAM module：通道 + 空间注意力精炼
        self.cbam = CBAM1D(
            channels=cnn_channels,
            reduction=cbam_reduction,
            spatial_kernel=7,
        )

        # 3. MSCA module：多尺度卷积 + 注意力融合
        self.msca = MSCAModule(
            in_channels=cnn_channels,
            msca_channels=msca_channels,
            kernel_sizes=kernel_sizes,
            dropout=cnn_dropout,
        )

        h_msca_dim = 2 * msca_channels

        # 4. Output classifier：Linear → GELU → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.Linear(h_msca_dim, h_msca_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(h_msca_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] attention mask

        Returns:
            logits: [B, num_classes]
        """
        # Step 1: DNABERT-2 backbone 编码
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        # hidden: [B, L, H]

        # Step 2: CNN module 局部特征提取
        cnn_out = self.cnn_module(hidden)  # [B, C, L]

        # Step 3: CBAM 注意力精炼
        cbam_out = self.cbam(cnn_out)  # [B, C, L]

        # Step 4: MSCA 多尺度融合
        h_msca = self.msca(cbam_out)  # [B, 2 * msca_channels]

        # Step 5: 分类输出
        return self.classifier(h_msca)


def count_trainable_parameters(model: nn.Module) -> Dict[str, float | int]:
    """统计总参数、可训练参数和可训练比例。

    复用 bioqpred_adapter 中同名函数的逻辑，作为独立副本避免循环引用。
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total else 0.0
    return {"total_params": total, "trainable_params": trainable, "trainable_ratio": round(ratio, 8)}
