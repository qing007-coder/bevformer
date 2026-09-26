import torch


def get_reference_points_3d(bev_h, bev_w, pc_range, z=0.0, device="cpu", dtype=torch.float32):
    """
    生成 BEV 网格中心在世界坐标系下的 3D 坐标 (x, y, z)。

    参数:
        bev_h, bev_w: BEV 特征图的高 / 宽，query 数量 N = bev_h * bev_w
        pc_range:     [x_min, y_min, x_max, y_max]，BEV 网格覆盖的真实世界范围（米）
        z:            参考平面的高度（米）。原版 BEVFormer 会在每个 pillar 上取
                      num_points_in_pillar=4 个不同高度来缓解"高度歧义"，
                      这里简化为单层，见 README「与原版的差异」。

    返回:
        [N, 3]  每行为一个 BEV 网格中心的世界坐标 (x, y, z)
    """
    x_min, y_min, x_max, y_max = pc_range

    # 网格中心：行方向对应 bev_h(y)，列方向对应 bev_w(x)
    ref_y, ref_x = torch.meshgrid(
        torch.arange(0.5, bev_h, dtype=dtype, device=device),
        torch.arange(0.5, bev_w, dtype=dtype, device=device),
        indexing="ij",
    )
    ref_x = ref_x.reshape(-1) / bev_w  # [N]，归一化到 (0, 1)
    ref_y = ref_y.reshape(-1) / bev_h

    x = x_min + ref_x * (x_max - x_min)  # 归一化坐标 -> 真实世界坐标
    y = y_min + ref_y * (y_max - y_min)
    z = torch.full_like(x, float(z))

    return torch.stack([x, y, z], dim=-1)  # [N, 3]


def project_to_image(points_3d, K, R, T, img_w, img_h):
    """
    针孔相机投影：世界坐标系 3D 点 -> 图像像素坐标。

        P_cam = R @ P_world + T
        u = fx * X_c / Z_c + cx
        v = fy * Y_c / Z_c + cy

    参数:
        points_3d: [B, N, 3]
        K:         [B, num_cams, 3, 3]  内参
        R:         [B, num_cams, 3, 3]  世界 -> 相机 旋转
        T:         [B, num_cams, 3]     世界 -> 相机 平移
        img_w, img_h: 图像宽高（像素），用于判断投影点是否落在画面内

    返回:
        points_2d: [B, num_cams, N, 2]  像素坐标 (u, v)
        depth:     [B, num_cams, N]     相机坐标系下的 Z
        valid:     [B, num_cams, N]     bool，投影点在画面内且在相机前方
    """
    B, N, _ = points_3d.shape
    num_cams = K.shape[1]

    pts = points_3d[:, None].expand(B, num_cams, N, 3)  # [B, num_cams, N, 3]
    pts_cam = pts @ R.transpose(-1, -2) + T[:, :, None, :]  # [B, num_cams, N, 3]

    X, Y, Z = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]

    # 防止除零：相机后方的点由 valid 里的 Z > 1e-5 过滤掉
    safe_Z = torch.clamp(Z, min=1e-5)

    fx, fy = K[..., 0, 0], K[..., 1, 1]  # [B, num_cams]
    cx, cy = K[..., 0, 2], K[..., 1, 2]

    u = fx[:, :, None] * X / safe_Z + cx[:, :, None]
    v = fy[:, :, None] * Y / safe_Z + cy[:, :, None]

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (Z > 1e-5)

    points_2d = torch.stack([u, v], dim=-1)  # [B, num_cams, N, 2]
    return points_2d, Z, valid


def multi_camera_projection(points_3d, Ks, Rs, Ts, img_w, img_h, num_levels):
    """
    把 BEV 参考点投影到所有相机，得到可形变注意力要用的参考点和有效性 mask。

    参数:
        points_3d:  [B, N, 3]                 BEV 网格中心的世界坐标
        Ks / Rs / Ts: [B, num_cams, ...]      相机内外参（见 project_to_image）
        img_w, img_h: 图像宽高
        num_levels: 特征层数（每个 level 共用同一个归一化参考点）

    返回:
        reference_points: [B, num_cams, N, num_levels, 2]  归一化到 [0, 1] 的 (u, v)
        masks:            [B, num_cams, N]                 bool，该 query 在该相机上是否可见
    """
    points_2d, _, masks = project_to_image(points_3d, Ks, Rs, Ts, img_w, img_h)

    # 像素 -> 归一化坐标，和可形变注意力里参考点的约定一致
    scale = torch.tensor([img_w, img_h], dtype=points_2d.dtype, device=points_2d.device)
    reference_points = points_2d / scale  # [B, num_cams, N, 2]

    # 各 level 共用同一个参考点：level 之间的尺度差异由 deformable attention 内部的
    # offset_normalizer 处理（原版也是这么做的）
    reference_points = reference_points.unsqueeze(3).repeat(1, 1, 1, num_levels, 1)

    return reference_points, masks
