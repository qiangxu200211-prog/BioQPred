"""Padding-aware adapted heads; these are not direct source-paper reproductions."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.models.baseline_dnabert2_camp_style import DNABERT2CAMPStyleClassifier
from src.models.baseline_tfbs_finder_style import TFBSFinderStyleClassifier
from src.models.cnn_baseline import SimpleCNNBaseline


def stub_backbone(hidden_size=768):
    module = nn.Module()
    module.config = SimpleNamespace(hidden_size=hidden_size)
    return module


def masked_mean(x, mask):
    return (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def masked_max(x, mask):
    return x.masked_fill(~mask.bool(), torch.finfo(x.dtype).min).amax(-1)


def masked_conv_block(block, x, mask):
    for layer in block:
        if isinstance(layer, nn.BatchNorm1d):
            valid = mask[:, 0].bool()
            values = layer(x.transpose(1, 2)[valid])
            restored = x.new_zeros(x.shape[0], x.shape[2], x.shape[1])
            restored[valid] = values
            x = restored.transpose(1, 2)
        else:
            x = layer(x)
        x = x * mask
    return x


class CNNAdapter(SimpleCNNBaseline):
    def forward(self, inputs):
        onehot = F.one_hot((inputs + 4) % 5, num_classes=5).float().transpose(1, 2)
        return super().forward(onehot)


class FrozenMeanHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, num_classes))

    def forward(self, inputs):
        return self.classifier(inputs)


class MaskedCAMPHead(DNABERT2CAMPStyleClassifier):
    def __init__(self, num_classes, hidden_size=768, channels=128):
        super().__init__(stub_backbone(hidden_size), num_classes, hidden_size=hidden_size, camp_channels=channels)

    def forward(self, inputs):
        hidden, mask = inputs["hidden"], inputs["mask"]
        channel_mask = mask.unsqueeze(1).to(hidden.dtype)
        x = hidden.transpose(1, 2) * channel_mask
        x = torch.cat([conv(x) * channel_mask for conv in self.camp.convs], 1).transpose(1, 2)
        attended, _ = self.camp.attention(x, x, x, key_padding_mask=~mask.bool(), need_weights=False)
        x = self.camp.attention_norm(x + attended).transpose(1, 2)
        return self.classifier(masked_mean(x, channel_mask))


class MaskedTFBSHead(TFBSFinderStyleClassifier):
    def __init__(self, num_classes, hidden_size=768, channels=128):
        super().__init__(stub_backbone(hidden_size), num_classes, hidden_size=hidden_size, cnn_channels=channels, msca_channels=channels)

    def forward(self, inputs):
        hidden, mask = inputs["hidden"], inputs["mask"]
        channel_mask = mask.unsqueeze(1).to(hidden.dtype)
        x = hidden.transpose(1, 2) * channel_mask
        x = masked_conv_block(self.cnn_module.conv1, x, channel_mask)
        x = masked_conv_block(self.cnn_module.conv2, x, channel_mask)
        attention = self.cbam.channel_attn
        weights = torch.sigmoid(attention.mlp(masked_mean(x, channel_mask)) + attention.mlp(masked_max(x, channel_mask)))
        x = x * weights.unsqueeze(-1)
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        x = x * torch.sigmoid(self.cbam.spatial_attn.conv(spatial)) * channel_mask
        scales = [conv(x) * channel_mask for conv in self.msca.convs]
        weights = self.msca.scale_attn(torch.cat([masked_mean(value, channel_mask) for value in scales], -1))
        x = (torch.stack(scales, 1) * weights[:, :, None, None]).sum(1)
        return self.classifier(torch.cat([masked_max(x, channel_mask), masked_mean(x, channel_mask)], -1))
