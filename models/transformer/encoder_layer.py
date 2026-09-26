import torch.nn as nn

from models.attention.temporal_self_attention import TemporalSelfAttention
from models.attention.spatial_cross_attention import SpatialCrossAttention

from .ffn import FFN


class BEVFormerEncoderLayer(nn.Module):
    """
    一层 BEVFormer encoder，按顺序做三件事（都是 pre-norm + 残差）：

        1. 时序自注意力：和上一帧 BEV 交互（第一帧没有历史，直接跳过）
        2. 空间交叉注意力：从多相机图像特征里取信息
        3. 前馈网络 FFN
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        feedforward_channels=1024,
        bev_h=200,
        bev_w=200,
        dropout=0.1,
    ):
        super().__init__()

        self.temporal_self_attention = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=2,  # 历史 BEV + 当前 BEV
            num_points=num_points,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        self.spatial_cross_attention = SpatialCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        self.ffn = FFN(
            embed_dim=embed_dim,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, history_bev, image_features, reference_points, masks):
        """
        query:            [B, N, C]
        history_bev:      [B, N, C] 或 None   上一帧 BEV 特征，None 表示第一帧
        image_features:   List[[B, num_cams, C, H_i, W_i]]
        reference_points: [B, num_cams, N, num_levels, 2]
        masks:            [B, num_cams, N]
        """
        # 1) 时序自注意力
        if history_bev is not None:
            identity = query
            attended = self.temporal_self_attention(
                query=self.norm1(query),
                history_bev=history_bev,
                current_bev=query,  # level 1 的 value 用未归一化的 query，和历史帧保持同一分布
                shift=None,  # TODO: 接入自车运动补偿（见 TemporalSelfAttention.get_shift）
            )
            query = identity + self.dropout1(attended)

        # 2) 空间交叉注意力
        identity = query
        attended = self.spatial_cross_attention(
            query=self.norm2(query),
            image_features=image_features,
            reference_points=reference_points,
            masks=masks,
        )
        query = identity + self.dropout2(attended)

        # 3) FFN
        query = query + self.ffn(self.norm3(query))

        return query
