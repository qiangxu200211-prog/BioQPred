#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DNABERT-2 分类头封装。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DNABERT2Classifier(nn.Module):
    """DNABERT-2 backbone + 线性分类头。

    frozen=True 时冻结 backbone，只训练 classifier head；frozen=False 时全量微调。
    """

    def __init__(self, backbone: nn.Module, num_classes: int, frozen: bool = True, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DNABERT-2 backbone config 缺少 hidden_size。")
        if frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(int(hidden_size), num_classes))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # transformers 标准模型返回 ModelOutput；DNABERT-2 remote code 在部分环境下返回 tuple。
        # 两种情况的第一个元素都是最后一层 token hidden states。
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)
