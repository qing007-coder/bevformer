import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    自顶向下的特征金字塔：把 ResNet 的 C2 ~ C5 都融合成 out_channels 通道的 P2 ~ P5。

    做法是标准的 FPN：高层特征上采样后和低层横向连接相加，再过一个 3x3 卷积平滑。
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()

        self.lateral_c2 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(in_channels[3], out_channels, kernel_size=1)

        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p5 = self.smooth_p5(p5)

        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.smooth_p4(p4)

        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.smooth_p3(p3)

        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p2 = self.smooth_p2(p2)

        return p2, p3, p4, p5
