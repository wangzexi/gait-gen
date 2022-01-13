import pytorch_lightning as pl
from pytorch_lightning import callbacks
from model import MaskGait
from dataset import CMU_DataModule

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MaskGait'),
    max_epochs=5000,
    log_every_n_steps=50,
    # val_check_interval=1,
    # limit_val_batches=0.5

    callbacks=[pl.callbacks.ModelCheckpoint(
        filename='{val/loss:.2f}-{epoch}-{step}',
        monitor='val/loss',
        save_top_k=3,
        mode='min',
    )]
)


model = MaskGait()
data_module = CMU_DataModule(batch_size=32)

trainer.fit(
    model=model,
    datamodule=data_module,
)
