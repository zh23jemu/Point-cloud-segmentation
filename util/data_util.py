import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def _sample_crop_center_index(label, split='train', crop_bias_classes=None, crop_bias_prob=0.0, crop_bias_min_points=1):
    """
    为训练裁剪选择中心点。

    设计说明：
    1. 验证/测试阶段保持原有确定性中心，避免评估结果被随机性污染。
    2. 训练阶段在指定概率下，优先从目标类别中选择中心点。
    3. 若目标类别点数过少，则自动回退到普通随机中心，避免过度偏置。
    """
    if 'train' not in split:
        return label.shape[0] // 2

    if crop_bias_classes and crop_bias_prob > 0 and np.random.rand() < crop_bias_prob:
        priority_mask = np.zeros(label.shape[0], dtype=bool)
        for cls_id in crop_bias_classes:
            priority_mask |= (label == cls_id)
        priority_index = np.where(priority_mask)[0]
        if priority_index.size >= crop_bias_min_points:
            return np.random.choice(priority_index)

    return np.random.randint(label.shape[0])


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None,
                 shuffle_index=False, crop_bias_classes=None, crop_bias_prob=0.0, crop_bias_min_points=1):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    # if voxel_size:
    #     coord_min = np.min(coord, 0)
    #     coord -= coord_min
    #     uniq_idx = voxelize(coord, voxel_size)
    #     coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        # 训练时支持“难类优先”的裁剪中心采样，让 class1 更频繁地处于局部窗口核心区域，
        # 从而提升模型对该类局部上下文和边界的学习强度。
        init_idx = _sample_crop_center_index(
            label,
            split=split,
            crop_bias_classes=crop_bias_classes,
            crop_bias_prob=crop_bias_prob,
            crop_bias_min_points=crop_bias_min_points
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
