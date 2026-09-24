import torch
import torch.nn as nn
from .encoder_layer import BEVFormerEncoderLayer


class BEVFormerEncoder(nn.Module):

    def __init__(
        self,
        num_layers=6,
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

        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            BEVFormerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                feedforward_channels=feedforward_channels,
                bev_h=bev_h,
                bev_w=bev_w,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])


    def forward(self, query, history_bev, image_features, reference_points, masks):
        for layer in self.layers:
            query = layer(
                query=query,
                history_bev=history_bev,
                image_features=image_features,
                reference_points=reference_points,
                masks=masks
            )

        return query    
            



