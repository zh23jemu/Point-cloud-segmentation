# CLAUDE.md — daima 项目 AI 上下文

## 项目概述

基于 **Point Transformer V3 (PTv3)** 的三维点云语义分割项目，针对 S3DIS 室内场景数据集，执行 **3 类自定义语义分割**任务。核心创新是将 2D EMA（Efficient Multi-scale Attention）模块改造为 1D 并插入 PTv3 解码器。

代码从 [Pointcept](https://github.com/Pointcept/Pointcept) 开源框架中抽取并二次改造，`tmp/Pointcept/` 保留原仓库作为参考，不参与本项目编译和训练。

---

## 目录结构

```
C:\Coding\daima\
├── config/
│   └── seg/
│       └── seg_pointtransformer_v3_EMA.yaml   # 主训练配置
├── data/
│   └── s3dis/
│       └── s3dis_names.txt                     # 3 个类别名称
├── exp/
│   └── seg/pointtransformer_v3_EMA/model/
│       ├── model_best.pth                      # 验证集最优权重
│       └── model_last.pth                      # 最后一轮权重
├── lib/
│   └── pointops/                               # 自定义 CUDA 算子（需编译）
├── model/
│   ├── pointtransformer_v3.py                  # 核心模型定义（含 EMA 改造）
│   └── serialization/                          # 空间填充曲线序列化
│       ├── default.py                          # Z-order / Hilbert 编解码
│       ├── hilbert.py
│       └── z_order.py
├── tmp/Pointcept/                              # 上游参考仓库（只读，勿修改）
├── tool/
│   ├── train.py                                # 训练主入口（支持 DDP）
│   ├── test.py                                 # 评测（混淆矩阵、per-class IoU）
│   ├── inference.py                            # 单文件推理
│   ├── plot_curves.py                          # 离线绘制训练曲线
│   ├── train.sh / test.sh / inference.sh       # Shell 启动脚本
└── util/
    ├── s3dis.py                                # S3DIS Dataset 类
    ├── data_util.py                            # 数据预处理、collate_fn
    ├── transform.py                            # 数据增强变换
    ├── lovasz_loss.py                          # Lovász-Softmax Loss
    ├── voxelize.py                             # 体素化（当前被注释未使用）
    ├── config.py                               # YAML 配置加载
    └── common_util.py                          # AverageMeter、IoU 计算等工具
```

---

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.x，CUDA C++（自定义算子） |
| 深度学习框架 | PyTorch，DDP 分布式训练 |
| 点云核心库 | spconv（稀疏卷积）、torch_scatter、pointops（自研 CUDA 算子） |
| Transformer 组件 | timm（DropPath），flash-attn（可选） |
| 辅助库 | addict、SharedArray |
| 监控 | TensorboardX、matplotlib |
| 配置 | YAML + 自研 CfgNode（支持命令行 key=value 覆盖） |

---

## 模型架构要点

### PTv3 主体

- **编码器**：5 stage，通道 `32→64→128→256→512`，深度 `2,2,2,6,2`
- **解码器**：4 stage，通道 `256→128→64→64`，深度 `2,2,2,2`
- **注意力头数**：`2,4,8,16,32`（编码器各 stage）
- **序列化**：Z-order、Z-trans、Hilbert、Hilbert-trans 四种轮换

### EMA 模块改造（核心创新）

- 将 2D 图像 EMA 适配为 **1D 点云序列** 处理
- 逐 Batch 处理（避免 padding 零值污染 GAP 统计）
- 加残差连接 + 可学习融合权重（`fusion_weight=0.1`）
- 通过 `ema_stages` 参数控制插入哪些解码器 stage
- 入口类：`EMAPointAdapter`，位于 `model/pointtransformer_v3.py`

---

## 数据流

```
原始点云 .txt（X Y Z R G B Label）
  → S3DIS Dataset：球形邻域裁剪（voxel_max=30000）+ 数据增强
  → collate_fn：batch 拼接 + offset 计数向量
  → PointTransformerV3.forward([coord, feat, offset])
      → Point.serialization()：空间填充曲线排序
      → Point.sparsify()：转 spconv SparseConvTensor
      → Embedding（SubMConv3d 5×5×5）
      → Encoder × 5 stage（SerializedAttention + Pooling）
      → Decoder × 4 stage（Unpooling + SerializedAttention + EMAPointAdapter）
      → seg_head（Linear → BN → ReLU → Linear(3)）
  → CrossEntropy Loss（权重 [10,10,1]）+ Lovász-Softmax Loss
  → SGD（lr=0.001）+ MultiStepLR（60%/80% epoch 各降 10x）
```

---

## 训练配置摘要

| 参数 | 值 |
|---|---|
| 数据集 | S3DIS，3 类，test_area=6 |
| 体素大小 | 0.04 |
| 最大点数 | 30,000 |
| batch_size | 4 |
| epochs | 100 |
| optimizer | SGD，momentum=0.9 |
| lr | 0.001，MultiStepLR |
| 损失函数 | CE（class_weights=[10,10,1]）+ Lovász，各权重 1.0 |
| GPU | 单卡（train_gpu=[0]），nccl 后端 |

---

## 注意事项

1. **CUDA 算子需编译**：使用前需在 `lib/pointops/` 下执行 `python setup.py install`
2. **体素化已注释**：`data_util.py` 中体素化代码被注释，当前仅球形裁剪
3. **tmp/ 只读**：`tmp/Pointcept/` 是上游参考仓库，不参与本项目运行，勿在其中修改
4. **已有训练结果**：`exp/` 目录下存在训练好的权重，可直接用于推理
5. **类别极不平衡**：前两类赋权重 10，第三类权重 1

---

## 常用命令

```bash
# 训练
bash tool/train.sh

# 测试评估
bash tool/test.sh

# 单文件推理
bash tool/inference.sh <input_file> <output_file>

# 绘制训练曲线
python tool/plot_curves.py
```
