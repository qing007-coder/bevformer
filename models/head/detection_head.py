import torch
import torch.nn as nn


class BEVDetectionHead(nn.Module):
    """
    BEV 检测头，包含两个并行任务头：
        - 回归头 reg_head 预测每个 query 对应的 3D 边界框参数，
          输出维度为 box_dim。
        - 分类头 cls_head 预测每个 query 所属的类别，
          输出维度为 num_classes  输出的是 logits 不是概率。

    输入:
        query: 形状为 (B, N, embed_dim) 或 (N, embed_dim) 的特征张量，
               其中 B 为 batch size    N 为 query 数量   embed_dim 为特征维度。

    输出:
        cls: 分类 logits  形状为 (B, N, num_classes) 或 (N, num_classes)。
        reg: 边界框回归参数，形状为 (B, N, box_dim) 或 (N, box_dim)。
        box_dim=10: [x, y, z, w, l, h, yaw, vx, vy, vz]
        x,y,z: 中心点坐标; w,l,h: 宽度/长度/高度; yaw: 偏航角; vx,vy,vz: x/y/z方向速度
    """
    def __init__(
        self,
        embed_dim=256,
        box_dim=10,
        num_classes=10,
    ):
        super().__init__()

        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, box_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, query): 
        """
        前向传播。

        参数:
            query: 输入特征，形状为 (B, N, embed_dim) 或 (N, embed_dim)。

        返回:
            cls: 分类 logits 形状为 (B, N, num_classes) 或 (N, num_classes)。
            reg: 回归参数，形状为 (B, N, box_dim) 或 (N, box_dim)。
        """

        cls = self.cls_head(query)
        reg = self.reg_head(query)

        return cls, reg 