import torch
import torch.nn as nn
from .deformable_attention import MultiScaleDeformableAttention


class TemporalSelfAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_levels=2, num_points=4, bev_h=200, bev_w=200):
        super().__init__()

        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w

        # self.query_proj = nn.Linear(embed_dim, embed_dim)
        # self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points)


    def forward(self, query, history_bev, shift=None):
        """
        query:       [B, Nq, C]
        history_bev: [B, N_history, C]
        shift:       [B, 2] 或 None
        """
        B, N, C = query.shape

        assert N == self.bev_h * self.bev_w, \
            f"N={N}, but bev_h*bev_w={self.bev_h * self.bev_w}"

        value = torch.cat([history_bev, query], dim=1)  # [B, N_history + Nq, C]

        ref = self.get_reference_points(self.bev_h, self.bev_w, device=query.device) # [Nq, 2]
        ref = ref.unsqueeze(0) # [1, Nq, 2]
        ref = ref.unsqueeze(2) # [1, Nq, 1, 2]

        if shift is not None:
            shift = shift[:, None, None, :]
            ref = ref + shift
        
        ref = ref.repeat(B, 1, 2, 1) # [B, Nq, 2, 2]

        spatial_shapes = torch.tensor([[self.bev_h, self.bev_w], [self.bev_h, self.bev_w]], device=query.device)  # [num_levels, 2]

        output = self.attention(query, ref, value, spatial_shapes)

        return output


    def get_reference_points(self, H, W, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0.5, H, device=device),
            torch.arange(0.5, W, device=device),
            indexing="ij"
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W

        reference_points = torch.stack((ref_x, ref_y), dim=-1)  # [Nq, 2]
        return reference_points


    def get_shift(self, dx, dy, pc_range):
        """
        dx: 自车在 x 方向的位移，单位通常为米。
        dy: 自车在 y 方向的位移，单位通常为米。
        pc_range: 形如 [x_min, y_min, x_max, y_max]  BEV 特征图覆盖的真实世界范围，单位通常为米。
        """
        x_min, y_min, x_max, y_max = pc_range

        shift_x = dx / (x_max - x_min)
        shift_y = dy / (y_max - y_min)

        return torch.stack([
            shift_x,
            shift_y
        ], dim=-1)
