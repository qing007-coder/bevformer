import torch
import torch.nn as nn

from .deformable_attention import MultiScaleDeformableAttention


class TemporalSelfAttention(nn.Module):
    """
    时序自注意力（TSA）：当前帧的 BEV query 从上一帧的 BEV 特征里取信息。

    复用可形变注意力，把它当成一个 num_levels = 2 的特例：
        level 0 -> 上一帧 BEV 特征
        level 1 -> 当前帧 BEV 特征
    参考点是 BEV 网格中心，网络自己预测采样偏移，等价于让它学会补偿自车运动
    （这里没有显式传入 ego motion 的 shift，见 README「与原版的差异」）。
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=2, num_points=4, bev_h=200, bev_w=200):
        super().__init__()

        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_levels = num_levels

        self.attention = MultiScaleDeformableAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points
        )

    def forward(self, query, history_bev, current_bev=None, shift=None):
        """
        query:       [B, Nq, C]         当前帧 query，在注意力里充当 query
        history_bev: [B, Nq, C]         上一帧 BEV 特征，作为 level 0 的 value
        current_bev: [B, Nq, C] 或 None 当前帧 BEV 特征，作为 level 1 的 value；
                                        不传就复用 query
        shift:       [B, 2] 或 None     自车位移造成的网格偏移（归一化坐标）

        返回:
            output:  [B, Nq, C]
        """
        B, Nq, C = query.shape

        assert Nq == self.bev_h * self.bev_w, \
            f"Nq={Nq}, 但 bev_h*bev_w={self.bev_h * self.bev_w}"
        assert history_bev is not None, "没有历史 BEV 时应该在 encoder layer 里直接跳过时序注意力"
        assert history_bev.shape[1] == Nq, \
            f"历史 BEV 长度 {history_bev.shape[1]} 和当前 query 数量 {Nq} 不一致"

        if current_bev is None:
            current_bev = query
        value = torch.cat([history_bev, current_bev], dim=1)  # [B, 2 * Nq, C]

        ref = self.get_reference_points(self.bev_h, self.bev_w, device=query.device)  # [Nq, 2]
        ref = ref.view(1, Nq, 1, 2)  # [1, Nq, 1, 2]

        if shift is not None:
            # shift 是归一化坐标，和参考点同一个尺度
            ref = ref + shift[:, None, None, :]

        ref = ref.repeat(B, 1, self.num_levels, 1)  # [B, Nq, num_levels, 2]

        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]] * self.num_levels, dtype=torch.long, device=query.device
        )  # [num_levels, 2]

        return self.attention(query, ref, value, spatial_shapes)

    def get_reference_points(self, H, W, device):
        """BEV 网格中心，归一化到 [0, 1]，返回 [Nq, 2] 的 (x, y)"""
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0.5, H, device=device),
            torch.arange(0.5, W, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W

        return torch.stack((ref_x, ref_y), dim=-1)  # [Nq, 2]

    def get_shift(self, dx, dy, pc_range):
        """
        把自车在 x / y 方向的位移换算成归一化网格偏移。

        dx: 自车在 x 方向的位移（米）
        dy: 自车在 y 方向的位移（米）
        pc_range: [x_min, y_min, x_max, y_max]，BEV 覆盖的真实世界范围（米）
        """
        x_min, y_min, x_max, y_max = pc_range

        shift_x = dx / (x_max - x_min)
        shift_y = dy / (y_max - y_min)

        return torch.stack([shift_x, shift_y], dim=-1)
