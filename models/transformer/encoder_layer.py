import torch
import torch.nn as nn
from models.attention.temporal_self_attention import TemporalSelfAttention
from models.attention.spatial_cross_attention import SpatialCrossAttention
from .ffn import FFN


class BEVFormerEncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        feedforward_channels=1024,
        bev_h=200,
        bev_w=200,
        dropout=0.1
    ):

        super().__init__()

        self.temporal_self_attention = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=2, # 这里是历史bev_query和当前bev_query
            num_points=num_points,
            bev_h=bev_h,
            bev_w=bev_w
        )

        self.spatial_cross_attention = SpatialCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            bev_h=bev_h,
            bev_w=bev_w
        )

        self.ffn = FFN(
            embed_dim=embed_dim, 
            feedforward_channels=feedforward_channels,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, history_bev, image_features, reference_points, masks):
        identity = query

        # 时序注意力
        query = self.norm1(query)
        query = self.temporal_self_attention(
            query=query,
            history_bev=history_bev,
            shift=None # 这个暂时先不搞
        )
        query = identity + query

        # 空间注意力
        identity = query
        query = self.norm2(query)
        query = self.spatial_cross_attention(
            query=query,
            image_features=image_features,
            reference_points=reference_points,
            masks=masks
        )
        query = identity + query

        # 前馈神经网络
        identity = query
        query = self.norm3(query)
        query = self.ffn(query)
        query = identity + query

        return query
