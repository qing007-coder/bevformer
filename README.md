# BEVFormer 复现

从零手写的 BEVFormer 核心复现，**纯 PyTorch**：不依赖 mmcv / mmdet3d，也不需要编译自定义
CUDA 算子（可形变注意力用 `F.grid_sample` 实现）。目的是把 BEVFormer 的每个模块都拆开自己
写一遍，所以代码优先保证**可读**而不是极致性能。

当前状态：**模型部分完成，前向 / 反向已端到端验证通过**；数据集、loss、训练循环还没写。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 组件级测试：ResNet + FPN 的形状
python test.py

# 2) 单元测试：可形变注意力（形状 / 梯度 / 与手写 grid_sample 对拍）
python test_msdeform_attn.py

# 3) 端到端连通性测试：随机图像 + 随机相机内外参，跑完整前向 + 反向
python test_bevformer.py
python test_bevformer.py --bev 50 --batch 2      # 换个大一点的配置
python test_bevformer.py --steps 0               # 只测前向反向，跳过训练 demo
```

`test_bevformer.py` 不需要任何数据集，全部输入都是随机生成的，CPU 上一分钟左右跑完，
输出形如：

```
配置：batch=1 cams=6 img=64x64 BEV=20x20 layers=2
参数量: 28.97 M
[1] 前向：第一帧（prev_bev=None，跳过时序注意力）
  有效投影比例 0.167，每个 query 平均可见相机数 1.00
  cls (1, 400, 10)   reg (1, 400, 10)   bev (1, 400, 256)      ✓
[2] 反向：第二帧（带 prev_bev，时序注意力生效）+ 全参数梯度检查
  全部 236 个参数都拿到了有限梯度                              ✓
[3] 过拟合 demo：同一个 batch 上训练 40 步
  前 5 步平均 loss = 2.2707 -> 后 5 步平均 loss = 1.7592       ✓
```

---

## 目录结构

```
bevformer/
├── models/
│   ├── backbone/
│   │   ├── resnet.py               # ResNet-50（只取 C2~C5 四层输出）
│   │   └── bottleneck.py           # Bottleneck 残差块
│   ├── neck/
│   │   └── fpn.py                  # 自顶向下 FPN，输出统一 256 通道的 P2~P5
│   ├── attention/
│   │   ├── deformable_attention.py # 多尺度可形变注意力（核心算子）
│   │   ├── temporal_self_attention.py   # 时序自注意力 TSA
│   │   └── spatial_cross_attention.py   # 空间交叉注意力 SCA
│   ├── transformer/
│   │   ├── ffn.py                  # 前馈网络
│   │   ├── encoder_layer.py        # 一层 = TSA + SCA + FFN
│   │   └── encoder.py              # 堆叠 num_layers 层
│   ├── head/
│   │   └── detection_head.py       # 分类头 + 回归头（两个 MLP）
│   └── bevformer.py                # 端到端模型
├── utils/
│   ├── geometry.py                 # BEV 参考点生成 + 相机投影
│   └── positional_encoding.py      # BEV 网格的 2D 正弦位置编码
├── test.py                         # ResNet + FPN 形状测试
├── test_msdeform_attn.py           # 可形变注意力单元测试
├── test_bevformer.py               # 端到端连通性测试
└── requirements.txt
```

---

## 模型数据流

```
images [B, num_cams, 3, H, W]
   │
   ├─ reshape -> [B*num_cams, 3, H, W]
   │
   ▼
ResNet-50 ─────► C2(1/4) C3(1/8) C4(1/16) C5(1/32)
   │
   ▼
FPN ───────────► P2 P3 P4 P5           # 都是 256 通道，作为 4 个 level
   │  reshape -> [B, num_cams, 256, h_i, w_i] × 4
   │
   │                    BEV query = bev_queries(可学习) + bev_pos(2D 正弦编码)
   │                                        [B, bev_h*bev_w, 256]
   │                                              │
   │                    BEV 3D 参考点 ──► 相机投影（K/R/T）──► 参考点 + mask
   │                                              │
   ▼                                              ▼
BEVFormerEncoder × num_layers
   │  每层： 时序自注意力（和上一帧 BEV）→ 空间交叉注意力（多相机图像特征）→ FFN
   ▼
bev [B, N, 256]
   │
   ├─► BEVDetectionHead ─► cls [B, N, num_classes], reg [B, N, box_dim]
   └─► 作为下一帧的 prev_bev（自回归时序建模）
```

**时序用法**：把上一次 forward 返回的 `bev` 传回 `prev_bev` 即可；第一帧传 `None`，
此时会自动跳过时序注意力（原版也是这个逻辑）：

```python
cls, reg, bev = model(images, Ks, Rs, Ts)                      # 第一帧
cls, reg, bev = model(images, Ks, Rs, Ts, prev_bev=bev.detach())   # 第二帧
```

---

## 已实现 / 未实现

| 模块 | 状态 |
| --- | --- |
| ResNet-50 backbone | ✅ |
| FPN neck | ✅ |
| 多尺度可形变注意力（纯 PyTorch） | ✅ |
| 时序自注意力 TSA | ✅ 不含 ego-motion shift |
| 空间交叉注意力 SCA | ✅ 单层 z 参考点 |
| Encoder（pre-norm + 残差 + dropout） | ✅ |
| 端到端 BEVFormer + 前向/反向 | ✅ |
| 检测头 | ⚠️ 只有两个 MLP，缺 box coder / 正负样本匹配 |
| 位置编码 | ✅ BEV 网格正弦编码；⚠️ 图像特征没有位置编码 |
| 数据集（nuScenes） | ❌ |
| Loss / 训练循环 / 推理 | ❌ |
| 预训练权重加载 | ❌ |

---

## 关键实现说明

### 1. 可形变注意力的偏移要先归一化

参考点是归一化到 `[0, 1]` 的，而偏移的单位是「像素」，所以必须除以该 level 的 `(W, H)`
才是同一个尺度：

```python
offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1)  # (W, H)
sampling_location = reference_points[...] + sampling_offsets / offset_normalizer[...]
```

少了这一步，同一个偏移权重在 200×200 的 level 上等于 200 像素、在 13×13 上只等于 13 像素，
训练初期采样点会被甩到特征图外面。

同时 `sampling_offsets` 的初始化也做了处理：权重置 0，bias 初始化成「每个 head 朝不同方向、
第 i 个采样点距离 i+1」，让每个 query 开局就在参考点周围小范围散开采样（和 mmcv 一致）。

### 2. 时序自注意力 = num_levels 为 2 的可形变注意力

把「上一帧 BEV」和「当前帧 BEV」拼起来当成两个 level 的 value，参考点是 BEV 网格中心，
由网络自己预测采样偏移 —— 等价于让它学会补偿自车运动。没有显式传入 ego motion 的 `shift`。

### 3. 空间交叉注意力按有效相机求平均

每个 BEV query 只和投影落在画面内、且在相机前方的那些相机交互：无效相机的特征先乘 0，
再除以有效相机数（`clamp(min=1)` 兜底，避免除 0）。这保证了「一个有效相机都没有」的 query
不会爆出 nan。

### 4. 位置编码

BEV query = 可学习的 `bev_queries`（内容）+ 固定的 2D 正弦编码 `bev_pos`（空间先验）。
正弦编码让不同网格位置在训练一开始就有区分度，而不是所有 query 完全相同。
