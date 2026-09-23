import torch
import torch.nn as nn
from .deformable_attention import MultiScaleDeformableAttention


class SpatialCrossAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, bev_h=200, bev_w=200):
        super().__init__()

        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.attention = MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points)

    def forward(self, query, image_features, reference_points, masks):
        """
        参数:
            query: Tensor, [B, N, C]
                B: batch size
                N: BEV 查询数量 (bev_h * bev_w)
                C: 特征维度 (embed_dim)
            image_features: List[Tensor]
                多尺度多相机图像特征列表，长度 num_levels
                每个元素形状 [B, num_cams, C, H_i, W_i]
            reference_points: Tensor, [B, num_cams, N, num_levels, 2]
                每个 BEV 查询在每个相机、每个特征层上的归一化 2D 参考点坐标 (u, v)
                其中 num_levels 对应 image_features 的层数（代码中 reshape 为 4)
            masks: Tensor, [B, cam_nums, N]

        返回:
            output: Tensor, [B, N, C]
                经过多相机 SpatialCrossAttention 融合后的 BEV 查询特征。
        """

        B, N, C = query.shape
        flatten_features, spatial_shapes = self.flatten_image_features(image_features=image_features)  # [B, num_cams, sum_i(H_i * W_i), C]  [num_levels, 2]

        _, num_cams, total_len, _ = flatten_features.shape
        flatten_features = flatten_features.view(B * num_cams, total_len, C) # [B*num_cams, total_len, C]

        query_camera = query.unsqueeze(1).repeat(1, num_cams, 1, 1) # [B, num_cams, N, C]
        query_camera = query_camera.view(B * num_cams, N, C) # [B * num_cams, N, C]

        reference_points = reference_points.view(B * num_cams, N, self.num_levels, 2) # [B * num_cams, N, num_levels, 2]

        output = self.attention(
            query=query_camera,
            reference_points=reference_points,
            value=flatten_features,
            spatial_shapes=spatial_shapes,
        ) # [B * num_cams, N, C]

        output = output.view(B, num_cams, N, C) # [B, num_cams, N, C]

        masks = masks.unsqueeze(-1).to(output.dtype) # [B, cam_nums, N, 1]  把bool变成可以做乘法的浮点数
        output = output * masks # [B, num_cams, N, C] 无效特征清零

        valid_count = masks.sum(dim=1) # [B, N, 1] 一个query的有效相机数
        output = output.sum(dim=1) / valid_count.clamp(min=1.0) # 对维度做均值 对有效相机为零的维度做兜底（除1而不是0）

        return output # [B, N, C]

    def get_reference_points_3d(self, bev_h, bev_w, pc_range, z=1.0):   
        x_min, y_min, x_max, y_max = pc_range

        # BEV网格中心
        ref_y, ref_x = torch.meshgrid(torch.arange(0.5, bev_h, device="cpu"), torch.arange(0.5, bev_w, device="cpu"), indexing="ij")

        # 归一化到 [0,1]
        ref_x = ref_x.reshape(-1) / bev_w
        ref_y = ref_y.reshape(-1) / bev_h

        # 转成真实世界坐标
        x = x_min + ref_x * (x_max - x_min)
        y = y_min + ref_y * (y_max - y_min)

        z = torch.full_like(x, z)

        reference_points = torch.stack(
            [x, y, z],
            dim=-1
        ) # [bev_h * bev_w, 3]  (x, y ,1)

        return reference_points

    def project_to_image(self, points_3d, K, R, T, img_w, img_h):
        """
            将世界坐标系下的 3D 点投影到图像像素坐标。
            投影公式：
                P_cam = R @ P_world + T
                u = fx * X_c / Z_c + cx
                v = fy * Y_c / Z_c + cy

            参数:
                points_3d: Tensor, 形状 [N, 3]
                    N 个世界坐标系下的 3D 点，每行为 (X_w, Y_w, Z_w)。
                K: Tensor, 形状 [3, 3]
                    相机内参矩阵。
                R: Tensor, 形状 [3, 3]
                    相机外参旋转矩阵，方向为 世界 -> 相机。
                T: Tensor, 形状 [3] 或 [3, 1]
                    相机外参平移向量，方向为 世界 -> 相机。
                img_w: int  # 图像宽度（像素列数），投影点 u 需满足 0 <= u < img_w
                img_h: int  # 图像高度（像素行数），投影点 v 需满足 0 <= v < img_h

            返回:
                points_2d: Tensor, 形状 [N, 2]
                    N 个像素坐标，每行为 (u, v)。
        """

        # 世界坐标 -> 相机坐标: P_cam = R @ P_world + T
        points_camera = points_3d @ R.T + T  # [N, 3]

        X = points_camera[:, 0]  # [N]
        Y = points_camera[:, 1]  # [N]
        Z = points_camera[:, 2]  # [N]

        # 防止除 0，数值保护（并不能处理点在相机后面的情况）
        safe_Z = torch.clamp(Z, min=1e-5)  # [N]

        # 从内参矩阵中取出焦距和主点
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # 透视除法 + 内参投影: u = fx * X / Z + cx, v = fy * Y / Z + cy
        u = fx * X / safe_Z + cx  # [N]
        v = fy * Y / safe_Z + cy  # [N]

        valid_x = (u >= 0) & (u < img_w)
        valid_y = (v >= 0) & (v < img_h)
        valid_depth = (Z > 1e-5)


        valid = valid_x & valid_y & valid_depth

        # 拼成像素坐标 [N, 2]
        points_2d = torch.stack([u, v], dim=-1)

        return points_2d, Z, valid


    def project_to_image(self, points_3d, K, R, T, img_w, img_h):
        """
        将世界坐标系下的 3D 点投影到图像像素坐标。
        投影公式：
            P_cam = R @ P_world + T
            u = fx * X_c / Z_c + cx
            v = fy * Y_c / Z_c + cy
    
        参数:
            points_3d: [B, N, 3]
                B: batch size
                N: BEV 查询点数量
    
            K: [3, 3]
                相机内参矩阵
    
            R: [3, 3]
                世界坐标系 -> 相机坐标系的旋转矩阵
    
            T: [3]
                世界坐标系 -> 相机坐标系的平移向量
    
            img_w: 图像宽度
            img_h: 图像高度
    
        返回:
            points_2d: [B, N, 2]
                图像像素坐标 (u, v)
    
            Z: [B, N]
                相机坐标系下的深度
    
            valid: [B, N]
                有效投影点 mask
        """
    
        # 世界坐标系 -> 相机坐标系
        points_camera = points_3d @ R.T + T
    
        X = points_camera[..., 0]
        Y = points_camera[..., 1]
        Z = points_camera[..., 2]
    
        # 防止 Z=0 导致除零
        safe_Z = torch.clamp(Z, min=1e-5)
    
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    
        # 针孔相机投影
        u = fx * X / safe_Z + cx
        v = fy * Y / safe_Z + cy
    
        # 判断投影点是否在图像范围内
        valid_x = (u >= 0) & (u < img_w)
        valid_y = (v >= 0) & (v < img_h)
    
        # 判断点是否在相机前方
        valid_depth = Z > 1e-5
    
        valid = valid_x & valid_y & valid_depth
    
        points_2d = torch.stack([u, v], dim=-1)
    
        return points_2d, Z, valid


    def multi_camera_projection(
            self,
            points_3d,
            Ks,
            Rs,
            Ts,
            img_w,
            img_h
    ):
        """
        将 3D BEV 参考点投影到多个相机。
    
        参数:
            points_3d: [B, N, 3]
                B: batch size
                N: BEV 查询点数量
    
            Ks: [num_cams, 3, 3]
                所有相机的内参矩阵
    
            Rs: [num_cams, 3, 3]
                所有相机的旋转矩阵
    
            Ts: [num_cams, 3]
                所有相机的平移向量
    
            img_w: 图像宽度
            img_h: 图像高度
    
        返回:
            reference_points: [B, num_cams, N, num_levels, 2]
                每个 BEV 查询点在每个相机、
                每个 FPN level 上对应的归一化 2D 参考点坐标。
    
            masks: [B, num_cams, N]
                每个 BEV 查询点在对应相机中是否有效。
        """
    
        B, N, _ = points_3d.shape
        num_cams = Ks.shape[0]
    
        points_2d_list = []
        mask_list = []
    
        for cam_id in range(num_cams):
            points_2d, depth, valid = self.project_to_image(
                points_3d=points_3d,
                K=Ks[cam_id],
                R=Rs[cam_id],
                T=Ts[cam_id],
                img_w=img_w,
                img_h=img_h
            )
    
            # 像素坐标归一化到 [0, 1]
            points_2d[..., 0] /= img_w
            points_2d[..., 1] /= img_h
    
            points_2d_list.append(points_2d)
            mask_list.append(valid)
    
        points_2d = torch.stack(points_2d_list, dim=1)
        # [B, num_cams, N, 2]
    
        masks = torch.stack(mask_list, dim=1)
        # [B, num_cams, N]
    
        reference_points = points_2d.unsqueeze(3).repeat(
            1, 1, 1, self.num_levels, 1
        )
        # [B, num_cams, N, num_levels, 2]
    
        return reference_points, masks

    def flatten_image_features(self, image_features):
        """
            将多尺度、多相机特征图展平并拼接成统一的 token 序列。

            Args:
                image_features (List[Tensor]): 多尺度特征图列表，长度为 num_levels。
                    每个元素形状为 [B, num_cams, C, H_i, W_i]，其中 i 为 level 索引。
                    - B: batch size
                    - num_cams: 相机数量
                    - C: 通道数（通常各 level 相同）
                    - H_i, W_i: 第 i 个 level 的空间高宽（不同 level 不同）

            Returns:
                flatten_features (Tensor): 展平并拼接后的特征，形状为
                    [B, num_cams, sum_i(H_i * W_i), C]。
                    其中 dim=2 是所有 level 空间 token 拼接后的序列长度，
                    dim=3 为通道维。
                spatial_shapes (Tensor): 每个 level 的空间尺寸，形状为
                    [num_levels, 2]，每行为 [H_i, W_i] dtype=torch.long
                    device 与 flatten_features 相同。
                    用于后续将 1D token 索引还原到对应 level 的 2D 坐标。
        """

        flatten_features = []
        spatial_shapes = []

        for feature in image_features:

            B, num_cams, C, H, W = feature.shape

            feature = feature.flatten(3)  # [B, num_cams, C, H*W]
            feature = feature.transpose(2, 3) # [B, num_cams, H*W, C]

            flatten_features.append(feature)

            spatial_shapes.append([H, W])

        flatten_features = torch.cat(flatten_features, dim=2)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=flatten_features.device)

        return flatten_features, spatial_shapes
