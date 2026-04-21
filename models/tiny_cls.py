from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels),
            ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1),
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2)
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(16, 16, stride=1),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 24, stride=1),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 32, stride=1),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 64, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

