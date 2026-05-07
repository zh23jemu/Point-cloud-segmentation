import os

import numpy as np
from torch.utils.data import Dataset
from util.data_util import data_prepare


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        self.data_list = sorted([f for f in os.listdir(os.path.join(data_root, split)) if f.endswith('.txt')])
        self.data_root = data_root + '/' + split
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = np.loadtxt(self.data_root + '/' + self.data_list[data_idx])
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 7]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label-1

    def __len__(self):
        return len(self.data_idx) * self.loop
