#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DNABERT2-CAMP-style baseline 模型。

参照 DNABERT2-CAMP: A Hybrid Transformer-CNN Model for E. coli Promoter Recognition
的方法设计，但不使用原论文数据集，而是在本项目的统一数据集上做 method-level reimplementation。

核心结构：
  DNABERT-2 Transformer global sequence encoding
  +
  CAMP module: CNN-Attention-Mean Pooling local feature refinement
  +
  classifier

默认冻结 DNABERT-2 backbone（freeze_backbone=True），只训练 CAMP module 和 classifier。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn


class CAMPModule(nn.Module):
    """CNN-Attention-Mean Pooling 局部特征精炼模块。

    输入：DNABERT-2 的 token-level hidden states，形状 [B, L, H]
    输出：序列级向量 h_camp，形状 [B, C_total]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        camp_channels: int = 128,
        kernel_sizes: List[int] | None = None,
        num_attention_heads: int = 4,
        cnn_dropout: float = 0.2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        # ---- Step 1：多尺度 CNN local refinement ----
        # 每个 kernel_size 对应一个 Conv1d block，输出 camp_channels 个通道
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, camp_channels, kernel_size=ks, padding=ks // 2),
                    nn.GELU(),
                    nn.Dropout(cnn_dropout),
                )
            )

        c_total = camp_channels * len(kernel_sizes)

        # ---- Step 2：MultiheadAttention refinement ----
        self.attention = nn.MultiheadAttention(
            embed_dim=c_total,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(c_total)

        self.c_total = c_total
        self.camp_channels = camp_channels
        self.kernel_sizes = kernel_sizes

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, H] DNABERT-2 输出的 token-level hidden states

        Returns:
            h_camp: [B, C_total] 序列级向量
        """
        # Step 1：多尺度 CNN —— [B, H, L] → 各 Conv1d → concat → [B, C_total, L]
        x = hidden_states.transpose(1, 2)  # [B, H, L]
        conv_outputs = [conv(x) for conv in self.convs]  # each [B, camp_channels, L]
        x = torch.cat(conv_outputs, dim=1)  # [B, C_total, L]

        # Step 2：Attention refinement —— [B, L, C_total]
        x = x.transpose(1, 2)  # [B, L, C_total]
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)  # 残差连接 + LayerNorm

        # Step 3：Mean Pooling over token 维度
        h_camp = x.mean(dim=1)  # [B, C_total]

        return h_camp


class DNABERT2CAMPStyleClassifier(nn.Module):
    """DNABERT2-CAMP-style 完整分类模型。

    DNABERT-2 backbone → CAMP module → classifier
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = True,
        hidden_size: int = 768,
        camp_channels: int = 128,
        kernel_sizes: List[int] | None = None,
        num_attention_heads: int = 4,
        cnn_dropout: float = 0.2,
        attention_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.backbone = backbone

        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # CAMP module
        self.camp = CAMPModule(
            hidden_size=hidden_size,
            camp_channels=camp_channels,
            kernel_sizes=kernel_sizes,
            num_attention_heads=num_attention_heads,
            cnn_dropout=cnn_dropout,
            attention_dropout=attention_dropout,
        )

        c_total = camp_channels * len(kernel_sizes)

        # Step 4：Classifier
        self.classifier = nn.Sequential(
            nn.Linear(c_total, c_total),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(c_total, num_classes),
        )

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

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
        # DNABERT-2 backbone 输出
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        # hidden: [B, L, H]

        # CAMP module 精炼
        h_camp = self.camp(hidden)  # [B, C_total]

        # 分类
        return self.classifier(h_camp)


def count_trainable_parameters(model: nn.Module) -> Dict[str, float | int]:
    """统计总参数、可训练参数和可训练比例。

    复用 bioqpred_adapter 中同名函数的逻辑，作为独立副本避免循环引用。
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total else 0.0
    return {"total_params": total, "trainable_params": trainable, "trainable_ratio": round(ratio, 8)}
