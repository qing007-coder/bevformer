import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, query, reference_points, value, spatial_shapes):
        """
        query:            [B, Nq, C]
        reference_points: [B, Nq, num_levels, 2]  每个 query 在每个 level 上的参考点 (x, y)，归一化到 [0,1]
        value:            [B, Nvalue, C]          多尺度特征 flatten 后拼起来，顺序和 spatial_shapes 对应
        spatial_shapes:   [num_levels, 2]         每个 level 的 (H, W)，用来把归一化坐标还原到特征图上

        """

        B, Nq, C = query.shape
        value = self.value_proj(value)

        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(B, Nq, self.num_heads, self.num_levels, self.num_points)
        attention_weights = F.softmax(attention_weights.flatten(3), dim=-1).view(B, Nq, self.num_heads, self.num_levels, self.num_points)

        sampling_location = (reference_points[:, :, None, :, None, :] + sampling_offsets)

        output = self._multi_scale_sampling(value, sampling_location, attention_weights, spatial_shapes)

        output = self.output_proj(output)
        return output

    def _multi_scale_sampling(self, value, sampling_location, attention_weights, spatial_shapes):
        """
        value:              [B, Nvalue, C]
        sampling_location:  [B, Nq, num_heads, num_levels, num_points, 2]
        attention_weights:  [B, Nq, num_heads, num_levels, num_points]
        spatial_shapes:     [num_levels, 2]

        return:
            output:         [B, Nq, C]

        Nvalue是所有level的特征图flatten后拼接起来的总长度，Nq是query的数量，C是特征维度。
        """
        B, Nvalue, C = value.shape
        Nq = sampling_location.shape[1]
        output = torch.zeros(B, Nq, self.num_heads, self.head_dim, device=value.device, dtype=value.dtype)

        start = 0
        for level in range (self.num_levels):
            
            H = int(spatial_shapes[level, 0])
            W = int(spatial_shapes[level, 1])

            level_length = H * W
            value_level = value[:, start:start + level_length, :].view(B, H, W, C)
            start = start + level_length

            value_level = value_level.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            value_level = value_level.view(B, self.num_heads, self.head_dim, H, W)  # [B, num_heads, head_dim, H, W]
            value_level = value_level.view(B * self.num_heads, self.head_dim, H, W)  # [B*num_heads, head_dim, H, W]

            sampling_grid = sampling_location[:, :, :, level, :, :]  # [B, Nq, num_heads, num_points, 2]
            sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).contiguous()  # [B, num_heads, Nq, num_points, 2]
            sampling_grid = sampling_grid.view(B * self.num_heads, Nq, self.num_points, 2)  # [B*num_heads, Nq, num_points, 2]

            # 双线性采样
            sampling_grid = sampling_grid * 2 - 1  # 将采样点从 [0,1] 映射到 [-1,1]，以适应 grid_sample 的输入要求
            sampled_value = F.grid_sample(value_level, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False)  # [B*num_heads, head_dim, Nq, num_points]

            sampled_value = sampled_value.view(B, self.num_heads, self.head_dim, Nq, self.num_points)  # [B, num_heads, head_dim, Nq, num_points]
            sampled_value = sampled_value.permute(0, 3, 1, 4, 2).contiguous()  # [B, Nq, num_heads, num_points, head_dim]

            attention_weight = attention_weights[:, :, :, level, :]  # [B, Nq, num_heads, num_points]
            attention_weight = attention_weight.unsqueeze(-1)  # [B, Nq, num_heads, num_points, 1]

            sampled_value = sampled_value * attention_weight  # [B, Nq, num_heads, num_points, head_dim]
            sampled_value = sampled_value.sum(dim=3)  # [B, Nq, num_heads, head_dim]

            output += sampled_value  # 累加每个 level 的结果

        return output.view(B, Nq, C)  # [B, Nq, C]


