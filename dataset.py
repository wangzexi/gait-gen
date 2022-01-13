from typing import Tuple
import re
import os
import functools

import torch
from torch.utils.data import DataLoader, Dataset, Subset, dataloader, dataset


import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


@functools.lru_cache(maxsize=6666)
def load_bvh_from_file(file_path: str) -> np.ndarray:
    cache_dir = os.path.join(os.path.dirname(file_path), '_cache')
    cache_npy = os.path.join(cache_dir, f'{os.path.basename(file_path)}.npy')

    if os.path.exists(cache_npy):
        return np.load(cache_npy, allow_pickle=True)
    elif not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    with open(file_path, 'r') as file:
        bvh = file.read()

    # 骨架信息 [93]
    skeleton = list(map(float, ' '.join(re.findall(
        r'OFFSET\s(.*?)\n\s*CHANNELS', bvh)).split(' ')))
    skeleton = np.array(skeleton, dtype=np.float32)

    # 运动帧信息 [96, 240]
    bvh = bvh[bvh.find('Frame Time'):]
    bvh = bvh[bvh.find('\n') + 1:]
    bvh = bvh.strip()
    # [240, 96]
    sequence = list(map(lambda f: [float(x)
                                   for x in f.split(' ')], bvh.split('\n')))
    sequence = np.array(sequence, dtype=np.float32)

    # 身份标签 ID
    label = int(os.path.basename(
        file_path).split('_')[0])  # 文件名第一个数字作为标签

    sample = np.array([skeleton, sequence, label], dtype=object)
    np.save(cache_npy, sample)
    return sample


def save_bvh_to_file(file_path: str, skeleton: np.ndarray, sequence: np.ndarray, frame_time: float = 0.025):
    # skeleton.shape: [93]
    # sequence.shape: [n, 96]

    with open('dataset/CMU/template.bvh', 'r') as f:
        bvh_template = f.read()

    split_index = bvh_template.find('MOTION')
    bvh_hierarchy = bvh_template[:split_index]
    bvh_motion = bvh_template[split_index:]

    bvh_hierarchy = bvh_hierarchy.format(*skeleton.tolist())

    n, _ = sequence.shape
    motion_data = '\n'.join([' '.join(['{:.6f}'.format(x) for x in frame])
                             for frame in sequence])
    bvh_motion = bvh_motion.format(n, frame_time, motion_data)

    output_dir = os.path.join(os.path.dirname(file_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'w') as f:
        f.write('\n'.join([bvh_hierarchy, bvh_motion]))


class CMU_Dataset(Dataset):
    def __init__(self, data_root: str = './dataset/CMU/walk'):
        self.sequence_file_path = sorted([os.path.join(data_root, f) for f in os.listdir(data_root)
                                          if not f.startswith('_')])
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.sequence_file_path)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        if idx < 0 or idx >= self.__len__():
            raise IndexError

        sample = load_bvh_from_file(self.sequence_file_path[idx])
        skeleton, sequence, label = sample

        # transform
        sequence = sequence[::3, 3:]  # 下采样帧率、去除根节点绝对位置

        return (
            skeleton,
            sequence,
            label,
        )


def collate_fn(
    random_clip: bool = False,
    min_clip_ratio: float = 0.05,  # 样本最小被裁剪到百分之几
    amplify_factor: float = 1,  # 样本扩增期望数量
):
    def _collate_fn(samples: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # samples: [(skeleton, sequence, label), ...]
        # sequence.shape: [sequence_num, keypoints_num]

        seq_length = len(samples[0][1])
        clip_length = seq_length
        if random_clip:
            clip_length = np.random.randint(
                seq_length*min_clip_ratio, seq_length+1)

        # 随机裁剪数据、扩增样本

        skeletons = []
        sequences = []
        labels = []

        for skeleton, sequence, label in samples:
            # sequence: [sequence_num, keypoints_x_y_z_num]
            t = 1
            if amplify_factor > 1:
                t = int((amplify_factor**0.5)*np.random.randn() +
                        amplify_factor)  # 正态分布样本扩增
            for _ in range(t):
                start_index = 0
                if random_clip and seq_length-clip_length > 0:
                    start_index = np.random.randint(
                        0, seq_length - clip_length)
                end_index = start_index + clip_length

                skeletons.append(torch.tensor(skeleton, dtype=torch.float32))
                sequences.append(torch.tensor(
                    sequence[start_index:end_index], dtype=torch.float32))
                labels.append(torch.tensor(label, dtype=torch.int32))

        return (
            torch.stack(skeletons),
            torch.stack(sequences),
            torch.stack(labels),
        )
    return _collate_fn


class CMU_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        dataset = CMU_Dataset()

        self.train = Subset(
            dataset=dataset,
            indices=range(0, 96)
        )
        self.val = Subset(
            dataset=dataset,
            indices=range(96, 154)
        )
        self.test = Subset(
            dataset=dataset,
            # indices=range(154, 161)
            indices=range(114, 161)
        )

    def train_dataloader(self) -> 'torch.utils.data.Dataloader':
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn(
                random_clip=True, min_clip_ratio=0.05, amplify_factor=3),
        )

    def val_dataloader(self) -> 'torch.utils.data.Dataloader':
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn(random_clip=False, amplify_factor=3),
        )

    def test_dataloader(self) -> 'torch.utils.data.Dataloader':
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn(random_clip=False, amplify_factor=1),
        )


if __name__ == '__main__':
    datamodule = CMU_DataModule()
    datamodule.prepare_data()
    train_loader = datamodule.train_dataloader()

    for epoch in range(10):
        for batch in tqdm(train_loader):
            skeleton, sequence, label = batch
            print(skeleton.shape, sequence.shape, label.shape)
            exit()
