"""
BEVFormer 端到端连通性测试。

全部输入都是随机生成的（图像、相机内外参、回归目标），
只验证「数据流是不是通的 / 形状对不对 / 梯度能不能回传 / 能不能过拟合」，
不验证精度。

用法:
    python test_bevformer.py                    # 默认小尺寸，CPU 上一两分钟跑完
    python test_bevformer.py --bev 50 --cams 6  # 放大一点
    python test_bevformer.py --steps 0          # 只看前向和反向，不跑训练
"""

import argparse
import math
import sys
import time

import torch

from models.bevformer import BEVFormer

# Windows 控制台默认 GBK，输出符号会崩，这里强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_cameras(num_cams, img_w, img_h, radius=1.5, height=1.5):
    """
    造一圈水平朝向的虚拟相机，环绕在自车（世界坐标原点）周围。

    世界坐标系约定：x / y 是 BEV 平面，z 朝上；相机坐标系：x 右、y 下、z 前。

    返回:
        Ks: [num_cams, 3, 3]  内参
        Rs: [num_cams, 3, 3]  世界 -> 相机 旋转
        Ts: [num_cams, 3]     世界 -> 相机 平移
    """
    fx = fy = img_w * 0.8
    cx, cy = img_w / 2.0, img_h / 2.0

    Ks, Rs, Ts = [], [], []
    for cam_id in range(num_cams):
        yaw = 2 * math.pi * cam_id / num_cams
        s, c = math.sin(yaw), math.cos(yaw)

        forward = torch.tensor([s, c, 0.0])  # 光轴：水平指出去
        up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(forward, up, dim=0)
        right = right / right.norm()
        down = torch.cross(forward, right, dim=0)

        R_c2w = torch.stack([right, down, forward], dim=1)  # 三列分别是 x/y/z 轴
        R_w2c = R_c2w.transpose(0, 1)

        cam_pos = torch.tensor([radius * s, radius * c, height])
        T_w2c = -R_w2c @ cam_pos

        Ks.append(torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))
        Rs.append(R_w2c)
        Ts.append(T_w2c)

    return torch.stack(Ks), torch.stack(Rs), torch.stack(Ts)


def to_batch(t, B):
    """给不带 batch 维的张量补上 batch 维并复制 B 份"""
    return t.unsqueeze(0).repeat(B, *([1] * t.dim()))


def check_shape(name, tensor, expected):
    assert tuple(tensor.shape) == tuple(expected), \
        f"{name} 形状错误: {tuple(tensor.shape)}, 期望 {tuple(expected)}"
    print(f"  {name:22s} {tuple(tensor.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", type=int, default=6, help="相机数量")
    parser.add_argument("--img", type=int, default=64, help="输入图像边长")
    parser.add_argument("--bev", type=int, default=20, help="BEV 特征图边长")
    parser.add_argument("--layers", type=int, default=2, help="encoder 层数")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--steps", type=int, default=40, help="过拟合 demo 的训练步数，0 表示跳过")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    B, cams, img, bev = args.batch, args.cams, args.img, args.bev
    N = bev * bev

    model = BEVFormer(
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_layers=args.layers,
        feedforward_channels=1024,
        bev_h=bev,
        bev_w=bev,
        dropout=0.1,
        box_dim=10,
        num_classes=10,
    )

    print("=" * 64)
    print("配置：batch={} cams={} img={}x{} BEV={}x{} layers={}".format(
        B, cams, img, img, bev, bev, args.layers))
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 随机输入：图像 + 相机内外参
    images = torch.randn(B, cams, 3, img, img)
    Ks, Rs, Ts = make_cameras(cams, img, img)
    Ks, Rs, Ts = to_batch(Ks, B), to_batch(Rs, B), to_batch(Ts, B)

    # ---------------------------------------------------------------- 1
    print("=" * 64)
    print("[1] 前向：第一帧（prev_bev=None，跳过时序注意力）")

    from utils.geometry import get_reference_points_3d, multi_camera_projection

    ref_3d = get_reference_points_3d(bev, bev, model.pc_range, z=model.ref_z)
    _, masks = multi_camera_projection(
        ref_3d.unsqueeze(0).repeat(B, 1, 1), Ks, Rs, Ts, img, img, 4
    )
    print(f"  有效投影比例 {masks.float().mean().item():.3f}，"
          f"每个 query 平均可见相机数 {masks.float().sum(dim=1).mean().item():.2f}")

    t0 = time.time()
    cls, reg, bev_feat = model(images, Ks, Rs, Ts)
    print(f"  前向耗时 {time.time() - t0:.2f}s")

    check_shape("cls", cls, (B, N, 10))
    check_shape("reg", reg, (B, N, 10))
    check_shape("bev", bev_feat, (B, N, 256))
    assert torch.isfinite(cls).all() and torch.isfinite(reg).all(), "前向输出里有 nan / inf"
    print("  ✓ 通过")

    # ---------------------------------------------------------------- 2
    print("=" * 64)
    print("[2] 反向：第二帧（带 prev_bev，时序注意力生效）+ 全参数梯度检查")

    cls2, reg2, bev2 = model(images, Ks, Rs, Ts, prev_bev=bev_feat.detach())
    check_shape("cls", cls2, (B, N, 10))
    check_shape("reg", reg2, (B, N, 10))
    check_shape("bev", bev2, (B, N, 256))

    loss = cls2.pow(2).mean() + reg2.pow(2).mean()
    loss.backward()

    no_grad, bad_grad = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            bad_grad.append(name)

    total_norm = torch.sqrt(sum(
        p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None
    ))
    print(f"  loss = {loss.item():.4f}, 梯度总范数 = {total_norm.item():.4f}")
    assert not no_grad, f"这些参数没拿到梯度: {no_grad}"
    assert not bad_grad, f"这些参数的梯度不是有限值: {bad_grad}"
    print(f"  全部 {sum(1 for _ in model.parameters())} 个参数都拿到了有限梯度")
    print("  ✓ 通过")

    # ---------------------------------------------------------------- 3
    if args.steps > 0:
        print("=" * 64)
        print(f"[3] 过拟合 demo：同一个 batch 上训练 {args.steps} 步，loss 应该下降")
        print("    （目标是随机张量，只为了证明梯度能穿透整个网络；lr=1e-3，调大了会炸）")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        target_cls = torch.randn(B, N, 10)
        target_reg = torch.randn(B, N, 10)

        model.train()  # dropout 打开，loss 会有抖动，看趋势就行
        losses = []
        t0 = time.time()
        for step in range(args.steps):
            opt.zero_grad()
            out_cls, out_reg, _ = model(images, Ks, Rs, Ts, prev_bev=bev_feat.detach())
            train_loss = (out_cls - target_cls).pow(2).mean() + (out_reg - target_reg).pow(2).mean()
            train_loss.backward()
            opt.step()
            losses.append(train_loss.item())
            if (step + 1) % max(1, args.steps // 4) == 0:
                print(f"  step {step + 1:3d}/{args.steps}  loss = {losses[-1]:.4f}")
        print(f"  {args.steps} 步耗时 {time.time() - t0:.2f}s")

        head = sum(losses[:5]) / 5
        tail = sum(losses[-5:]) / 5
        print(f"  前 5 步平均 loss = {head:.4f} -> 后 5 步平均 loss = {tail:.4f}")
        assert tail < head, f"loss 没有下降: {head:.4f} -> {tail:.4f}"
        print("  ✓ 通过")

    print("=" * 64)
    print("全部测试完成：模型整体是通的")


if __name__ == "__main__":
    main()
