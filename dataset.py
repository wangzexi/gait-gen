from typing import Tuple

import os
import json
import functools

import torch
from torch.utils.data import DataLoader, Dataset, Subset


import pytorch_lightning as pl
import torchmetrics

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import multiprocessing


def get_keypoint_index(keypoint: str) -> int:
    keypoints = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
                 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar']
    return keypoints.index(keypoint)


def read_frame_file(file_path: str) -> np.ndarray:
    # print(file_path)
    with open(file_path, 'r') as file:
        data = json.loads(file.read())
    frame = np.array(data['people'][0]['pose_keypoints_2d']).reshape((18, 3))
    # frame.shape: [(keypoints_num), (x, y, confidence)]
    return frame[:, :2]


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    i_neck = get_keypoint_index('Neck')
    i_rhip = get_keypoint_index('RHip')
    i_lhip = get_keypoint_index('LHip')

    kp_neck = np.array([frame[i_neck, 0], frame[i_neck, 1]])
    kp_rhip = np.array([frame[i_rhip, 0], frame[i_rhip, 1]])
    kp_lhip = np.array([frame[i_lhip, 0], frame[i_lhip, 1]])
    kp_mhip = (kp_rhip + kp_lhip) / 2
    kp_center = (kp_neck + kp_mhip) / 2
    scale = np.linalg.norm(kp_center - kp_mhip)
    return (frame - kp_center) / scale  # align center and scale


def read_sequence_dir(dir_path: str) -> np.ndarray:
    sequence = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        frame = read_frame_file(file_path)
        frame = normalize_frame(frame)
        sequence.append(frame)
    # sequence.shape: [(sequence_num), (keypoints_num), (x, y)]
    return np.array(sequence)


@functools.lru_cache(maxsize=6666)
def read_subject_dir(subject_dir: str) -> np.ndarray:
    subject_id = int(os.path.basename(subject_dir))

    cache_dir = os.path.join(subject_dir, '..', '_cache')
    cache_npy = os.path.join(cache_dir, f'{subject_id}.npy')

    if os.path.exists(cache_npy):
        return np.load(cache_npy, allow_pickle=True)
    elif not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    sequences = []
    for sequence_dir in os.listdir(subject_dir):
        camera_angle, sequence_no = sequence_dir.split('_')
        camera_angle, sequence_no = int(camera_angle), int(sequence_no)
        sequence_dir_path = os.path.join(subject_dir, sequence_dir)
        sequence = read_sequence_dir(sequence_dir_path)
        sequences.append((subject_id, camera_angle, sequence_no, sequence))
    sequences = np.array(sequences, dtype=object)

    cache_npy = os.path.join(
        cache_dir, f'{subject_id}_{sequences.shape[0]}.npy')
    np.save(cache_npy, sequences)
    # np_sequences.shape: [sequence_num, (subject_id, camera_angle, sequence_no, sequence)]
    return sequences


class OU_MVLP_POSE_Dataset(Dataset):
    def __init__(self, data_root: str = './dataset/OU-MVLP-POSE/alphapose'):
        # there should be 10307 subject folders in the data_root folder

        cache_npy = os.path.join(
            data_root, '_cache', '_subject_sequence_num.npy')
        if os.path.exists(cache_npy):
            subject_sequences_num = np.load(cache_npy)
        else:
            subject_dirs = sorted([f for f in os.listdir(data_root)
                                   if not f.startswith('_')])

            subject_sequences_num = []
            for subject_dir in subject_dirs:
                print(subject_dir)
                sequences = read_subject_dir(
                    os.path.join(data_root, subject_dir))
                print(sequences.shape)
                subject_sequences_num.append(sequences.shape[0])
            subject_sequences_num = np.array(subject_sequences_num)
            np.save(cache_npy, subject_sequences_num)
        # subject_sequence_num: [subject_1_num, subject_2_num, ...]

        subject_sequences_range = [0]
        for x in subject_sequences_num:
            subject_sequences_range.append(subject_sequences_range[-1] + x)
        # subject_sequences_range: [0, subject_1_num, subject_1_num + subject_2_num, ...]
        subject_sequences_range = np.array(subject_sequences_range)
        self.subject_sequences_range = subject_sequences_range

        self.data_root = data_root

    def __len__(self) -> int:
        return self.subject_sequences_range[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if idx < 0 or idx >= self.__len__():
            raise IndexError

        subject_id = np.searchsorted(
            self.subject_sequences_range, idx, side='right')
        seq_local_index = idx - self.subject_sequences_range[subject_id - 1]

        subject_dir = os.path.join(self.data_root, f'{subject_id:05d}')
        sequences = read_subject_dir(subject_dir)
        subject_id, camera_angle, sequence_no, sequence = sequences[seq_local_index]
        return sequence, subject_id, camera_angle, sequence_no


def collate_fn(random_clip: bool = False):
    def _collate_fn(samples: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # samples: [(sequence, subject_id, camera_angle, sequence_no), ...]
        # sequence.shape: [sequence_num, (keypoints_num), (x, y)]

        # find the min sequence length
        min_sequence_num = min([x[0].shape[0] for x in samples])

        # all sequences should have the same length
        sequences = [torch.tensor(x[0], dtype=torch.float32)
                     for x in samples]  # ragged
        for i, s in enumerate(sequences):
            # s: [sequence_num, (keypoints_num), (x, y)]
            n = s.shape[0]

            startIndex = 0
            if random_clip and n > min_sequence_num:
                startIndex = np.random.randint(0, n - min_sequence_num)

            endIndex = startIndex + min_sequence_num
            sequences[i] = s[startIndex:endIndex]

        return (
            torch.stack(sequences),  # clip sequence
            torch.stack([torch.tensor(x[1], dtype=torch.float32)
                        for x in samples]),  # subject_id
            torch.stack([torch.tensor(x[2], dtype=torch.float32)
                        for x in samples]),  # camera_angle
            torch.stack([torch.tensor(x[3], dtype=torch.float32)
                        for x in samples]),  # sequence_no
        )
    return _collate_fn


class OU_MVLP_POSE_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0
        # self.num_workers = multiprocessing.cpu_count()

    def prepare_data(self) -> None:
        dataset = OU_MVLP_POSE_Dataset()

        split_index = [0, 3607, 5153, 10307]
        split_seq_index = [dataset.subject_sequences_range[x]
                           for x in split_index]

        self.train = Subset(
            dataset=dataset,
            indices=range(split_seq_index[0], split_seq_index[1])
        )
        self.val = Subset(
            dataset=dataset,
            indices=range(split_seq_index[1], split_seq_index[2])
        )
        self.test = Subset(
            dataset=dataset,
            indices=range(split_seq_index[2], split_seq_index[3])
        )

    def train_dataloader(self) -> 'torch.utils.data.Dataloader':
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn(random_clip=True),
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> 'torch.utils.data.Dataloader':
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn(random_clip=False)
        )

    def test_dataloader(self) -> 'torch.utils.data.Dataloader':
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn(random_clip=False)
        )


# with torch.no_grad():
#     model = Transformer(dim=128, depth=3, heads=6,
#                         dim_head=64, mlp_dim=256, dropout=0.1)
#     x = torch.randn(1, 15, 128)  # [b, n, c]
#     y = model(x)
#     print(y.shape)

# root = './dataset/OU-MVLP-POSE/alphapose/_cache'
# for x in os.listdir(root):
#     if x.startswith('_'):
#         continue
#     arr = x.split('_')
#     oldname = x
#     newname = f'{arr[0]}.npy'
#     os.rename(os.path.join(root, oldname), os.path.join(root, newname))
#     print(newname)


if __name__ == '__main__':
    datamodule = OU_MVLP_POSE_DataModule()
    datamodule.prepare_data()
    train_loader = datamodule.train_dataloader()

    for epoch in range(10):
        for batch in tqdm(train_loader):
            sequence, subject_id, camera_angle, sequence_no = batch
            continue

    # d = OU_MVLP_POSE_Dataset()
    # s = Subset(d, range(0, 100000))
    # dl = DataLoader(s, batch_size=64, shuffle=True, num_workers=0)

    # for epoch in range(10):
    #     for batch in tqdm(dl):
    #         pass

    # dataset = OU_MVLP_POSE_Dataset()

    # print(len(dataset))
    # s = dataset[len(dataset)-1]

    # sequences = read_subject_dir(os.path.join(DATA_ROOT, '00001'))
    # sequence = sequences[0][3]
    # frame = sequence[0]

    # print(sequences.shape)
    # print(sequence.shape)
    # print(frame.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_aspect('equal')
    # ax.invert_yaxis()

    # aligned = align_center(frame)
    # scatter = ax.scatter(aligned[:, 0], aligned[:, 1], c='gray')

    # plt.show()

    # sequence = read_sequence_dir(os.path.join(
    #     './dataset/OU-MVLP-POSE/alphapose', '03607', '000_00'))
    # print(sequence.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_aspect('equal')
    # ax.invert_yaxis()

    # for frame in sequence:
    #     scatter = ax.scatter(frame[:, 0], frame[:, 1], c='gray')

    # plt.savefig('test.png')

    # exit()
