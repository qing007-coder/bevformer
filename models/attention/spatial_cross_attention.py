import torch
import torch.nn as nn

from .deformable_attention import MultiScaleDeformableAttention


class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力（SCA）：每个 BEV query 只和它「看得见」的相机特征交互。

    本模块只负责把张量整理成可形变注意力需要的形状：
        1. 多相机、多尺度的图像特征 flatten 成一条 token 序列
        2. query 复制 num_cams 份（每个相机一份，配一份投影好的参考点）
        3. 跑可形变注意力，把投影不到（不在画面内 / 在相机后方）的相机结果置 0
        4. 对有效相机做平均

    参考点由外部用 utils.geometry.multi_camera_projection 算好传进来。
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, bev_h=200, bev_w=200):
        super().__init__()

        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.attention = MultiScaleDeformableAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points
        )

    def forward(self, query, image_features, reference_points, masks):
        """
        参数:
            query: Tensor, [B, N, C]
                N: BEV query 数量（bev_h * bev_w）
            image_features: List[Tensor]，长度 num_levels
                每个元素形状 [B, num_cams, C, H_i, W_i]
            reference_points: Tensor, [B, num_cams, N, num_levels, 2]
                每个 BEV query 在每个相机、每个 level 上的归一化参考点 (u, v)
            masks: Tensor, [B, num_cams, N]
                bool，该 query 在该相机上的投影是否有效

        返回:
            output: Tensor, [B, N, C]
        """
        B, N, C = query.shape
        assert N == self.bev_h * self.bev_w, \
            f"N={N}, 但 bev_h*bev_w={self.bev_h * self.bev_w}"
        assert len(image_features) == self.num_levels, \
            f"image_features 有 {len(image_features)} 层，期望 {self.num_levels}"

        flatten_features, spatial_shapes = self.flatten_image_features(
            image_features
        )  # [B, num_cams, sum_i(H_i*W_i), C], [num_levels, 2]

        _, num_cams, total_len, _ = flatten_features.shape

        # 把 (B, num_cams) 合并进 batch 维，让可形变注意力一次处理完所有相机
        flatten_features = flatten_features.reshape(B * num_cams, total_len, C)
        query_camera = query.unsqueeze(1).repeat(1, num_cams, 1, 1).view(B * num_cams, N, C)

        reference_points = reference_points.view(B * num_cams, N, self.num_levels, 2)

        output = self.attention(
            query=query_camera,
            reference_points=reference_points,
            value=flatten_features,
            spatial_shapes=spatial_shapes,
        )  # [B * num_cams, N, C]
        output = output.view(B, num_cams, N, C)

        # 无效相机（投影不在画面内 / 在相机后方）的特征清零
        masks = masks.unsqueeze(-1).to(output.dtype)  # [B, num_cams, N, 1]
        output = output * masks

        # 对有效相机求平均；一个有效相机都没有时用 clamp 兜底，避免除 0
        valid_count = masks.sum(dim=1)  # [B, N, 1]
        output = output.sum(dim=1) / valid_count.clamp(min=1.0)

        return output  # [B, N, C]

    def flatten_image_features(self, image_features):
        """
        把多尺度、多相机的特征图展平拼成统一的 token 序列。

        参数:
            image_features: List[Tensor]，长度 num_levels，每个 [B, num_cams, C, H_i, W_i]

        返回:
            flatten_features: [B, num_cams, sum_i(H_i * W_i), C]
            spatial_shapes:   [num_levels, 2]，每行 [H_i, W_i]
        """
        flatten_features = []
        spatial_shapes = []

        for feature in image_features:
            B, num_cams, C, H, W = feature.shape

            feature = feature.flatten(3)  # [B, num_cams, C, H*W]
            feature = feature.transpose(2, 3)  # [B, num_cams, H*W, C]

            flatten_features.append(feature)
            spatial_shapes.append([H, W])

        flatten_features = torch.cat(flatten_features, dim=2)
        spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.long, device=flatten_features.device
        )

        return flatten_features, spatial_shapes
