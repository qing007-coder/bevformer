import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):
    """
    多尺度可形变注意力（Deformable DETR / BEVFormer 的核心算子）。

    对每个 query 做的事：
        1. 用线性层预测 num_heads * num_levels * num_points 组采样偏移和注意力权重
        2. 采样点 = 参考点 + 偏移（偏移以「像素」为单位，除以该 level 的 (W, H) 后
           转成归一化坐标，这样同一个偏移在不同分辨率的 level 上含义一致）
        3. 在对应 level 的特征图上做双线性采样，按注意力权重加权求和
        4. 把各 level 的结果相加 -> [B, Nq, C]

    纯 PyTorch 实现，用 F.grid_sample 代替 mmcv 里编译的 CUDA 算子。
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.head_dim = embed_dim // num_heads

        # 每个 (head, level, point) 预测一个 (dx, dy)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        # 每个 (head, level, point) 预测一个注意力权重，softmax 在 (level, point) 上做
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        # 采样偏移：权重置 0，bias 初始化为「每个 head 朝不同方向、第 i 个采样点距离 i + 1」。
        # 这样训练一开始，每个 query 就会在参考点周围小范围散开采样，而不是被随机权重
        # 甩到特征图外面（和 mmcv 官方实现一致）。
        # 注意：这里的前提是偏移的单位是「像素」，见 forward 里的归一化。
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [num_heads, 2]
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1  # [num_heads, num_levels, num_points, 2]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

        # 注意力权重置 0 -> softmax 之后每个 (level, point) 的权重相同，不偏袒任何一层
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, reference_points, value, spatial_shapes):
        """
        query:            [B, Nq, C]
        reference_points: [B, Nq, num_levels, 2]  每个 query 在每个 level 上的参考点，
                          归一化到 [0, 1]
        value:            [B, Nvalue, C]         多尺度特征 flatten 后拼起来，
                          顺序和 spatial_shapes 对应
        spatial_shapes:   [num_levels, 2]        每个 level 的 (H, W)，
                          用来把归一化坐标还原到特征图上

        返回:
            output:       [B, Nq, C]
        """
        B, Nq, C = query.shape

        assert reference_points.shape[-2] == self.num_levels, \
            f"reference_points 的 level 维是 {reference_points.shape[-2]}，期望 {self.num_levels}"

        value = self.value_proj(value)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Nq, self.num_heads, self.num_levels, self.num_points, 2
        )

        attention_weights = self.attention_weights(query).view(
            B, Nq, self.num_heads, self.num_levels, self.num_points
        )
        # softmax 在所有 level 和采样点上做：一个 query 的所有采样点权重和为 1
        attention_weights = F.softmax(attention_weights.flatten(3), dim=-1).view(
            B, Nq, self.num_heads, self.num_levels, self.num_points
        )

        # 采样点 = 参考点 + 偏移；偏移单位是像素，先除以 level 的 (W, H) 转回归一化坐标
        offset_normalizer = torch.stack(
            [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1  # (W, H)
        ).to(device=query.device, dtype=query.dtype).view(1, 1, 1, self.num_levels, 1, 2)

        sampling_location = (
            reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer
        )  # [B, Nq, num_heads, num_levels, num_points, 2]

        output = self._multi_scale_sampling(value, sampling_location, attention_weights, spatial_shapes)

        return self.output_proj(output)

    def _multi_scale_sampling(self, value, sampling_location, attention_weights, spatial_shapes):
        """
        value:              [B, Nvalue, C]
        sampling_location:  [B, Nq, num_heads, num_levels, num_points, 2]
        attention_weights:  [B, Nq, num_heads, num_levels, num_points]
        spatial_shapes:     [num_levels, 2]

        返回:
            output:         [B, Nq, C]

        Nvalue 是所有 level 的特征图 flatten 后拼起来的总长度，
        Nq 是 query 的数量，C 是特征维度。
        """
        B, Nvalue, C = value.shape
        Nq = sampling_location.shape[1]
        num_levels = spatial_shapes.shape[0]

        total = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
        assert total == Nvalue, \
            f"spatial_shapes 展开后共 {total} 个 token，但 value 的长度是 {Nvalue}"

        output = torch.zeros(B, Nq, self.num_heads, self.head_dim, device=value.device, dtype=value.dtype)

        start = 0
        for level in range(num_levels):
            H = int(spatial_shapes[level, 0])
            W = int(spatial_shapes[level, 1])

            level_length = H * W
            value_level = value[:, start:start + level_length, :].reshape(B, H, W, C)
            start = start + level_length

            # [B, H*W, C] -> [B, num_heads, head_dim, H, W] -> [B*num_heads, head_dim, H, W]
            value_level = value_level.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            value_level = value_level.view(B, self.num_heads, self.head_dim, H, W)
            value_level = value_level.reshape(B * self.num_heads, self.head_dim, H, W)

            sampling_grid = sampling_location[:, :, :, level, :, :]  # [B, Nq, num_heads, num_points, 2]
            sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).contiguous()  # [B, num_heads, Nq, num_points, 2]
            sampling_grid = sampling_grid.reshape(B * self.num_heads, Nq, self.num_points, 2)

            # 双线性采样：grid_sample 要求坐标在 [-1, 1]，所以把 [0, 1] 映射过去
            sampling_grid = sampling_grid * 2 - 1
            sampled_value = F.grid_sample(
                value_level, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )  # [B*num_heads, head_dim, Nq, num_points]

            sampled_value = sampled_value.view(B, self.num_heads, self.head_dim, Nq, self.num_points)
            sampled_value = sampled_value.permute(0, 3, 1, 4, 2).contiguous()  # [B, Nq, num_heads, num_points, head_dim]

            attention_weight = attention_weights[:, :, :, level, :].unsqueeze(-1)  # [B, Nq, num_heads, num_points, 1]
            sampled_value = (sampled_value * attention_weight).sum(dim=3)  # [B, Nq, num_heads, head_dim]

            output += sampled_value  # 各 level 加权累加

        return output.reshape(B, Nq, C)  # [B, Nq, C]
