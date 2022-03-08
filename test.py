import pytorch_lightning as pl
from model import MaskGait
from dataset import CMU_DataModule
import einops
import torch
import numpy as np
from dataset import save_bvh_to_file

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MaskGait')
)

model = MaskGait.load_from_checkpoint(
    'pre_training/loss=41.02-epoch=161-step=1619.ckpt')
# model = MaskGait()
data_module = CMU_DataModule(batch_size=100)

# 测试细节请实现在 model.py 中 MaskGait 类的 test_step 函数中
trainer.test(model=model, datamodule=data_module)
