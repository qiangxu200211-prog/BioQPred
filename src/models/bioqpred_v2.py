#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BioQPred-v2 模型定义。

本模块实现一个 DNABERT-compatible 的局部-语义融合模型：
1. position-aware local branch 负责学习剪接位点等任务中的局部 motif 与位置信号；
2. frozen DNABERT-2 semantic branch 保留预训练语义表征；
3. composition branch 提供 GC/CpG 等低维生物组成特征；
4. fusion head 将三类证据融合后分类。

注意：本模型不使用 test 信息做任何结构选择，所有可选分支均由训练脚本在 dev MCC 上选择。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


DNA_TO_TOKEN = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}


def encode_base_ids(sequences: Sequence[str], max_len: int) -> torch.Tensor:
    """将 DNA 序列编码为 token id，并右侧 padding。

    约定 A=1, C=2, G=3, T=4, N/未知=0。padding 也使用 0，便于 embedding 的
    ``padding_idx=0`` 自动保持为零向量。
    """

    encoded = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        seq = str(seq).upper()[:max_len]
        for col, base in enumerate(seq):
            encoded[row, col] = DNA_TO_TOKEN.get(base, 0)
    return encoded


def compute_composition_features(sequences: Sequence[str]) -> torch.Tensor:
    """提取低维 DNA composition 特征。

    这些特征只依赖单条样本本身，不需要在 train/dev/test 上 fit，因此不会引入泄漏。
    """

    rows = []
    for seq in sequences:
        s = str(seq).upper()
        length = max(1, len(s))
        a = s.count("A")
        c = s.count("C")
        g = s.count("G")
        t = s.count("T")
        n = s.count("N")
        cpg = sum(1 for i in range(max(0, len(s) - 1)) if s[i : i + 2] == "CG")
        gc = g + c
        at = a + t
        purine = a + g
        pyrimidine = c + t
        rows.append(
            [
                float(length),
                a / length,
                c / length,
                g / length,
                t / length,
                n / length,
                gc / length,
                at / length,
                float(cpg),
                cpg / length,
                purine / length,
                pyrimidine / length,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


class SEAttention1D(nn.Module):
    """轻量 Squeeze-and-Excitation 通道注意力。

    它先用全局平均池化得到通道重要性，再对卷积特征逐通道重加权。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.net(x).unsqueeze(-1)
        return x * weight


class SpatialPyramidPooling1D(nn.Module):
    """一维空间金字塔池化。

    bins=[1,2,4] 时，同时对序列维度做多尺度 max/avg pooling，兼顾全局和区域证据。
    """

    def __init__(self, bins: Iterable[int] = (1, 2, 4)):
        super().__init__()
        self.bins = tuple(int(b) for b in bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for b in self.bins:
            pieces.append(F.adaptive_max_pool1d(x, b).flatten(start_dim=1))
            pieces.append(F.adaptive_avg_pool1d(x, b).flatten(start_dim=1))
        return torch.cat(pieces, dim=1)


class PositionAwareLocalBranch(nn.Module):
    """位置感知局部模式分支。

    该分支使用可学习位置 embedding、多尺度卷积、dilated conv、SE attention 和 SPP，
    重点增强剪接任务中对中心附近局部碱基模式的捕获能力。
    """

    def __init__(
        self,
        max_len: int,
        embed_dim: int = 32,
        branch_channels: int = 128,
        kernel_sizes: Sequence[int] = (3, 5, 7, 11, 15),
        spp_bins: Sequence[int] = (1, 2, 4),
        dropout: float = 0.2,
        use_position: bool = True,
        use_multiscale: bool = True,
        use_spp: bool = True,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.use_position = bool(use_position)
        self.use_multiscale = bool(use_multiscale)
        self.use_spp = bool(use_spp)
        self.base_embedding = nn.Embedding(5, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        kernels = tuple(kernel_sizes) if use_multiscale else (7,)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, branch_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in kernels
            ]
        )
        merged_channels = branch_channels * len(kernels)
        self.dilated = nn.Sequential(
            nn.Conv1d(merged_channels, merged_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(merged_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(merged_channels, merged_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(merged_channels),
            nn.GELU(),
        )
        self.attention = SEAttention1D(merged_channels)
        self.spp = SpatialPyramidPooling1D(spp_bins)
        if self.use_spp:
            pooled_dim = merged_channels * sum(spp_bins) * 2
        else:
            pooled_dim = merged_channels * 2
        self.output_dim = 256
        self.project = nn.Sequential(
            nn.Linear(pooled_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, base_ids: torch.Tensor) -> torch.Tensor:
        batch, length = base_ids.shape
        x = self.base_embedding(base_ids)
        if self.use_position:
            pos = torch.arange(length, device=base_ids.device).unsqueeze(0).expand(batch, length)
            x = x + self.position_embedding(pos)
        x = x.transpose(1, 2)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.dilated(x)
        x = self.attention(x)
        if self.use_spp:
            pooled = self.spp(x)
        else:
            pooled = torch.cat([torch.amax(x, dim=-1), torch.mean(x, dim=-1)], dim=1)
        return self.project(pooled)


class CompositionBranch(nn.Module):
    """DNA 组成特征 MLP 分支。"""

    def __init__(self, input_dim: int = 12, dropout: float = 0.2):
        super().__init__()
        self.output_dim = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ResidualAdapter(nn.Module):
    """DNABERT pooled feature 后的轻量 residual adapter。"""

    def __init__(self, dim: int = 256, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.up(F.gelu(self.down(x))))


class BioQPredV2Classifier(nn.Module):
    """BioQPred-v2 分类器。

    DNABERT-2 backbone 默认冻结，只作为语义证据分支；局部分支和融合分类头参与训练。
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        max_len: int,
        bert_input_dim: int | None = None,
        use_comp: bool = False,
        use_adapter: bool = True,
        use_local: bool = True,
        use_bert: bool = True,
        use_position: bool = True,
        use_multiscale: bool = True,
        use_spp: bool = True,
        fusion: str = "concat",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        hidden_size = int(getattr(backbone.config, "hidden_size"))
        bert_feature_dim = int(bert_input_dim or hidden_size * 2)
        self.num_classes = int(num_classes)
        self.use_comp = bool(use_comp)
        self.use_local = bool(use_local)
        self.use_bert = bool(use_bert)
        self.fusion = fusion

        self.local_branch = (
            PositionAwareLocalBranch(
                max_len=max_len,
                use_position=use_position,
                use_multiscale=use_multiscale,
                use_spp=use_spp,
                dropout=dropout,
            )
            if self.use_local
            else None
        )
        self.bert_project = (
            nn.Sequential(
                nn.Linear(bert_feature_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(256),
            )
            if self.use_bert
            else None
        )
        self.adapter = ResidualAdapter(256, 64) if use_adapter and self.use_bert else nn.Identity()
        self.comp_branch = CompositionBranch(dropout=dropout) if self.use_comp else None

        if fusion == "gated" and self.use_local and self.use_bert:
            self.local_to_gate = nn.Linear(256, 256)
            self.bert_to_gate = nn.Linear(256, 256)
            self.gate = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())
            fuse_dim = 256 + (128 if self.use_comp else 0)
        else:
            fuse_dim = (256 if self.use_local else 0) + (256 if self.use_bert else 0) + (128 if self.use_comp else 0)
        if fuse_dim <= 0:
            raise ValueError("BioQPred-v2 至少需要启用一个输入分支。")
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, 384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def encode_bert(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """用 frozen DNABERT-2 得到 mean+max pooled 语义特征。"""

        # backbone 完全冻结，只提供语义证据；这里显式 no_grad，避免为 DNABERT-2
        # 构建反向传播图，显著降低显存与训练时间。
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if attention_mask is None:
            mean_pool = hidden.mean(dim=1)
            max_pool = torch.amax(hidden, dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            masked_hidden = hidden.masked_fill(mask == 0, torch.finfo(hidden.dtype).min)
            max_pool = torch.amax(masked_hidden, dim=1)
        return self.adapter(self.bert_project(torch.cat([mean_pool, max_pool], dim=-1)))

    def forward(
        self,
        encoded_inputs: Dict[str, torch.Tensor],
        base_ids: torch.Tensor,
        comp_features: torch.Tensor | None = None,
        bert_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_local = self.local_branch(base_ids) if self.use_local and self.local_branch is not None else None
        h_bert = None
        if self.use_bert:
            if bert_pooled is None:
                h_bert = self.encode_bert(**encoded_inputs)
            else:
                # bert_pooled 是冻结 DNABERT-2 的 mean+max pooled 原始表征缓存；
                # projection/adapter 仍然参与训练，因此语义分支没有被移除。
                h_bert = self.adapter(self.bert_project(bert_pooled))
        pieces = []
        if self.fusion == "gated" and h_local is not None and h_bert is not None:
            local = self.local_to_gate(h_local)
            bert = self.bert_to_gate(h_bert)
            gate = self.gate(torch.cat([local, bert], dim=-1))
            pieces.append(gate * local + (1.0 - gate) * bert)
        else:
            if h_local is not None:
                pieces.append(h_local)
            if h_bert is not None:
                pieces.append(h_bert)
        if self.use_comp and comp_features is not None and self.comp_branch is not None:
            pieces.append(self.comp_branch(comp_features))
        return self.classifier(torch.cat(pieces, dim=-1))


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """统计总参数、可训练参数和可训练比例。"""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": round(float(trainable / total), 8) if total else 0.0,
    }
