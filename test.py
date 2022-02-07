import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from model import MaskGait
from dataset import CMU_DataModule
import einops
import torch
import numpy as np
from dataset import save_bvh_to_file
import torch.utils.tensorboard as tb

writer = tb.SummaryWriter('tb_logs')

# logger = pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MaskGait')


model = MaskGait()
model = model.load_from_checkpoint(
    'pre_training/loss=119.99-epoch=81-step=491.ckpt')

data_module = CMU_DataModule(batch_size=100)
data_module.prepare_data()

test_dataloader = data_module.test_dataloader()

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        skeleton, sequence, label = batch
        b, n, c = sequence.shape

        # 拿到feature
        feature, embedding = model.encoder(sequence)

        hybrid_feature = feature[:2, :].clone()
        hybrid_feature[0, :128] = feature[5, :128].clone()
        hybrid_feature[1, :128] = feature[40, :128].clone()

        hybrid_label = label[:2].clone()
        hybrid_label[0] = 77
        hybrid_label[1] = 78

        mat = torch.cat([feature, hybrid_feature], dim=0)
        metadata = torch.cat([label, hybrid_label], dim=0)

        print(mat.shape)
        print(metadata.shape)

        writer.add_embedding(
            mat=mat,
            metadata=metadata,
            global_step=1
        )
        writer.flush()

        # 输出一个序列
        recon_seq = model.decoder(hybrid_feature, embedding[:2, :, :])

        i = 0
        np_label = hybrid_label[i].cpu().numpy()
        np_recon_sequence = recon_seq[i].detach().cpu().numpy()

        np_skeleton = skeleton[0].detach().cpu().numpy()

        n, _ = np_recon_sequence.shape
        np_recon_sequence = np.concatenate(  # 设置根节点xyz
            [np.zeros([n, 3], dtype=np.float32), np_recon_sequence], axis=1)

        save_bvh_to_file(
            'tb_logs/output/gen_hybrid_{}_{}.bvh'.format(np_label, n), np_skeleton, np_recon_sequence, frame_time=0.025)
