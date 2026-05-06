import os

import numpy as np
from torch.utils.data import Dataset
from util.data_util import data_prepare


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None,
                 transform=None, shuffle_index=False, loop=1, crop_bias_classes=None, crop_bias_prob=0.0,
                 crop_bias_min_points=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max = split, voxel_size, transform, voxel_max
        self.shuffle_index, self.loop = shuffle_index, loop
        # 这些参数只在训练裁剪时生效，用于提升目标难类在局部窗口中的覆盖率。
        self.crop_bias_classes = crop_bias_classes
        self.crop_bias_prob = crop_bias_prob
        self.crop_bias_min_points = crop_bias_min_points
        self.data_list = sorted([f for f in os.listdir(os.path.join(data_root, split)) if f.endswith('.txt')])
        self.data_root = data_root + '/' + split
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = np.loadtxt(self.data_root + '/' + self.data_list[data_idx])
        coord, feat = data[:, 0:3], data[:, 3:6]
        # 在数据集入口统一转换为从 0 开始的标签，避免后续“关注 class1”时再做额外映射。
        label = data[:, 7].astype(np.int64) - 1
        coord, feat, label = data_prepare(
            coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index,
            crop_bias_classes=self.crop_bias_classes,
            crop_bias_prob=self.crop_bias_prob,
            crop_bias_min_points=self.crop_bias_min_points
        )
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
