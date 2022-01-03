import pytorch_lightning as pl
from model import MaskGait
from dataset import OU_MVLP_POSE_DataModule

logger = pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MaskGait')
trainer = pl.Trainer(
    gpus=1,
    logger=logger,
    max_epochs=100,
    log_every_n_steps=5,
    # val_check_interval=1,
    # limit_val_batches=0.5
)

model = MaskGait()
data_module = OU_MVLP_POSE_DataModule(batch_size=4096)

trainer.fit(
    model=model,
    datamodule=data_module,
)
