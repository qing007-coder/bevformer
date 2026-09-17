import torch
import torch.nn as nn
from .bottleneck import Bottleneck


class ResNet(nn.Module):

    expansion = 4

    def __init__(self, layers):
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(mid_channels=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(mid_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(mid_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(mid_channels=512, blocks=layers[3], stride=2)

    def _make_layer(self, mid_channels, blocks, stride=1):

        out_channels = mid_channels * self.expansion

        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, mid_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5