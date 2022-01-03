
grid = torchvision.utils.make_grid(x[:8], normalize=True)
self.logger.experiment.add_image('generated_images', grid, self.global_step)

self.log_dict({"val_acc": 88, "val_loss": loss})
self.logger.log_metrics({"loss/train": loss}, step=self.global_step)


self.logger.experiment.add_scalar(
    "loss/train", loss, self.global_step)

module = MyModule.load_from_checkpoint(
    'tb_logs/my_model/version_0/checkpoints/epoch=9-step=8599.ckpt')

trainer.test(
    model=module,
    datamodule=datamodule)
