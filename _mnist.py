from logging import log
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms


import pytorch_lightning as pl

import torchvision

import numpy as np
import matplotlib.pyplot as plt

import torchmetrics
import multiprocessing

from circle_loss import convert_label_to_similarity, CircleLoss

# 数据集


class MyDataModule(pl.LightningDataModule):
    def prepare_data(self):
        dataset = MNIST(
            root='./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.train, self.val = random_split(dataset, [55000, 5000])
        self.test = MNIST(
            root='./dataset', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        self.num_workers = multiprocessing.cpu_count()

    def train_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self, batch_size=64, shuffle=False):
        return DataLoader(self.val, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def test_dataloader(self, batch_size=64, shuffle=False):
        return DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)


# 模型
class MyModule(LightningModule):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 10)

        metrics = torchmetrics.MetricCollection([
            # torchmetrics.Accuracy(compute_on_step=False),
            # torchmetrics.Precision(compute_on_step=False),
            # torchmetrics.Recall(),
            # torchmetrics.F1(),
            torchmetrics.AUROC(num_classes=10),
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='valid/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.criterion = CircleLoss(m=0.25, gamma=80)

    def forward(self, x):  # [64, 1, 28, 28]
        batch_size, channels, height, width = x.size()

        x = x.view(batch_size, -1)  # [64, 784]
        x = F.relu(self.layer_1(x))  # [64, 128]
        x = F.relu(self.layer_2(x))  # [64, 64]
        x = self.layer_3(x)  # [64, 10]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # logits = nn.functional.normalize(self.forward(x))
        # loss = self.criterion(*convert_label_to_similarity(logits, y))

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        output = self.train_metrics(logits, y)
        output['train/loss'] = loss
        self.log_dict(output)

        return loss

    def training_epoch_end(self, outputs):

        # print(outputs)
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        output = self.valid_metrics(logits, y)
        self.log('valid/loss', loss)
        self.log_dict(output)

        return loss

    def test_step(self, batch, batch_idx):
        return
        x, y = batch

        logits = self(x)
        loss = F.nll_loss(logits, y)

        output = self.test_metrics(logits, y)
        self.log('test/loss', loss)
        self.log_dict(output)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


logger = TensorBoardLogger(save_dir='tb_logs', name='my_model')
trainer = Trainer(
    gpus=1,
    logger=logger,
    max_epochs=20,
)

datamodule = MyDataModule()


# module = MyModule.load_from_checkpoint(
#     'tb_logs/my_model/version_1/checkpoints/epoch=19-step=17199.ckpt')

module = MyModule()

trainer.fit(
    model=module,
    datamodule=datamodule,
    # ckpt_path='tb_logs/my_model/version_0/checkpoints/epoch=9-step=8599.ckpt'
)
