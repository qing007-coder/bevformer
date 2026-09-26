import torch


def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000.0):
    """
    生成 BEV 网格的 2D 正弦位置编码（DETR 风格的固定编码，不需要训练）。

    编码方式：
        embed_dim 均分两半 —— 前半编码行方向 (y)，后半编码列方向 (x)；
        每一半内部再均分为 sin / cos 两部分（所以 embed_dim 必须是 4 的倍数）。

    参数:
        h, w:       BEV 特征图的高宽
        embed_dim:  编码维度
        temperature: 频率底数

    返回:
        [h * w, embed_dim]，第 (y * w + x) 行对应网格位置 (y, x)
        —— 与 reference_points 的行优先展开顺序一致
    """
    assert embed_dim % 4 == 0, f"embed_dim 需要能被 4 整除，当前为 {embed_dim}"

    half = embed_dim // 2  # y 和 x 各占一半
    quarter = half // 2  # 一半再分给 sin / cos

    # 1 / (temperature ^ (i / quarter))，i = 0..quarter-1
    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (temperature ** omega)  # [quarter]

    grid_y = torch.arange(h, dtype=torch.float32).unsqueeze(1)  # [h, 1]
    grid_x = torch.arange(w, dtype=torch.float32).unsqueeze(1)  # [w, 1]

    pe_y = torch.cat([torch.sin(grid_y * omega), torch.cos(grid_y * omega)], dim=1)  # [h, half]
    pe_x = torch.cat([torch.sin(grid_x * omega), torch.cos(grid_x * omega)], dim=1)  # [w, half]

    pe_y = pe_y.unsqueeze(1).expand(h, w, half)  # [h, w, half]
    pe_x = pe_x.unsqueeze(0).expand(h, w, half)  # [h, w, half]

    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, embed_dim)
