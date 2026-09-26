import torch
import torch.nn as nn

from utils.geometry import get_reference_points_3d, multi_camera_projection
from utils.positional_encoding import build_2d_sincos_position_embedding

from .backbone.resnet import ResNet
from .neck.fpn import FPN
from .transformer.encoder import BEVFormerEncoder
from .head.detection_head import BEVDetectionHead


class BEVFormer(nn.Module):
    """
    端到端的 BEVFormer（学习型简化复现，纯 PyTorch）。

    数据流：
        images [B, num_cams, 3, H, W]
          -> ResNet                     C2 ~ C5
          -> FPN                        P2 ~ P5（统一 256 通道，作为 4 个 level）
          -> BEV query                  可学习嵌入 + 2D 正弦位置编码
          -> BEVFormerEncoder x N       (时序自注意力 + 空间交叉注意力 + FFN)
          -> BEVDetectionHead           cls / reg

    时序：encoder 需要上一帧的 BEV 特征；把上一次 forward 返回的 bev 传回 prev_bev
    即可（第一帧传 None，此时自动跳过时序注意力）。详见 test_bevformer.py。

    和原版 BEVFormer 的差异见 README「与原版的差异」。
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_layers=6,
        feedforward_channels=1024,
        bev_h=200,
        bev_w=200,
        dropout=0.1,
        box_dim=10,
        num_classes=10,
        fpn_in_channels=(256, 512, 1024, 2048),
        backbone_layers=(3, 4, 6, 3),
        pc_range=(-51.2, -51.2, 51.2, 51.2),
        ref_z=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.ref_z = ref_z

        self.backbone = ResNet(list(backbone_layers))
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=embed_dim)

        self.encoder = BEVFormerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            feedforward_channels=feedforward_channels,
            bev_h=bev_h,
            bev_w=bev_w,
            dropout=dropout,
        )

        self.head = BEVDetectionHead(
            embed_dim=embed_dim, box_dim=box_dim, num_classes=num_classes
        )

        # BEV query = 可学习嵌入（内容）+ 固定的 2D 正弦位置编码（空间先验）
        self.bev_queries = nn.Parameter(torch.zeros(bev_h * bev_w, embed_dim))
        self.register_buffer(
            "bev_pos",
            build_2d_sincos_position_embedding(bev_h, bev_w, embed_dim),
            persistent=False,  # 纯计算得到，不用存进 checkpoint
        )

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.bev_queries, std=0.02)

    def forward(self, images, Ks, Rs, Ts, prev_bev=None):
        """
        参数:
            images:   [B, num_cams, 3, H, W]   多相机图像（已做 resize / normalize）
            Ks:       [B, num_cams, 3, 3]      相机内参
            Rs:       [B, num_cams, 3, 3]      世界 -> 相机 旋转矩阵
            Ts:       [B, num_cams, 3]         世界 -> 相机 平移向量
            prev_bev: [B, N, C] 或 None        上一帧 BEV 特征，第一帧传 None

        返回:
            cls: [B, N, num_classes]  分类 logits
            reg: [B, N, box_dim]      边界框回归量
            bev: [B, N, C]            当前帧 BEV 特征，喂给下一帧的 prev_bev
        """
        B, num_cams, _, img_h, img_w = images.shape

        # --- 1. 多相机图像特征 ---
        images = images.reshape(B * num_cams, 3, img_h, img_w)
        c2, c3, c4, c5 = self.backbone(images)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        # 每个 level 拆回 [B, num_cams, C, h, w]，给空间交叉注意力用
        image_features = [
            p.reshape(B, num_cams, self.embed_dim, p.shape[-2], p.shape[-1])
            for p in (p2, p3, p4, p5)
        ]

        # --- 2. BEV query 与相机投影参考点 ---
        bev_query = (self.bev_queries + self.bev_pos).unsqueeze(0).repeat(B, 1, 1)  # [B, N, C]

        ref_3d = get_reference_points_3d(
            self.bev_h, self.bev_w, self.pc_range,
            z=self.ref_z, device=images.device, dtype=images.dtype,
        )  # [N, 3]
        reference_points, masks = multi_camera_projection(
            ref_3d.unsqueeze(0).repeat(B, 1, 1), Ks, Rs, Ts, img_w, img_h, self.num_levels
        )  # [B, num_cams, N, num_levels, 2], [B, num_cams, N]

        # --- 3. Encoder ---
        bev = self.encoder(bev_query, prev_bev, image_features, reference_points, masks)

        # --- 4. 检测头 ---
        cls, reg = self.head(bev)

        return cls, reg, bev
