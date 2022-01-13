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
    'tb_logs/MaskGait/version_0/checkpoints/val/loss=34.315491-epoch=875-step=2627.ckpt')

data_module = CMU_DataModule(batch_size=100)
data_module.prepare_data()
# test_dataloader = data_module.test_dataloader()
test_dataloader = data_module.test_dataloader()

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        skeleton, sequence, label = batch
        b, n, c = sequence.shape

        # 拿到feature
        feature, embedding = model.encoder(sequence)

        # 染色体互换
        # hybrid_skeleton = skeleton.clone()
        # hybrid_sequence = sequence.clone()
        # hybrid_label = label.clone()

        # hybrid_sequence[:, ]

        writer.add_embedding(feature, metadata=label, global_step=0)
        writer.flush()
        print(b, n, c)
        exit()

        embedding = embedding[:, :1, :]  # 是否给e1

        i = 0
        j = 5
        print(f'main:{label[i]} from:{label[j]}')
        embedding[i, 0, :] = embedding[j, 0, :]  # 偷妻换子

        gen_seq = torch.zeros(b, 0, c)

        for _ in range(80):
            new_embedding = einops.repeat(
                model.decoder.placeholder, '() () d -> b n d', b=b, n=1)
            t = embedding.shape[1]+1
            new_embedding += model.decoder.pos_embedding[:, t:(t+1), :]
            embedding = torch.cat([embedding, new_embedding], dim=1)

            recon_seq = model.decoder(feature, embedding)
            gen_seq = torch.cat([gen_seq, recon_seq[:, -1:, :]], dim=1)

            _, embedding = model.encoder(gen_seq)
        print(gen_seq.shape)

        # embedding = torch.zeros(b, 0, model.encoder.proj.out_features)
        # gen_seq = torch.zeros(b, 0, c)
        # print(embedding.shape)
        # print(gen_seq.shape)

        # for t in range(1+embedding.shape[1], 50):
        #     # 制造一个新帧的embedding
        #     new_embedding = einops.repeat(
        #         model.decoder.placeholder, '() () d -> b n d', b=b, n=1)
        #     new_embedding += torch.randn_like(new_embedding)  # 噪声
        #     new_embedding += model.decoder.pos_embedding[:, t:(t+1), :]

        #     embedding = torch.cat([embedding, new_embedding], dim=1)
        #     reconstruction = model.decoder(feature, embedding)
        #     gen_seq = torch.cat(
        #         [gen_seq, reconstruction[:, -1:, :]], dim=1)  # 拿到生成的一帧
        #     print(gen_seq.shape)

        #     _, embedding = model.encoder(gen_seq)

        # 输出一个序列
        i = 0
        np_label = label[i].cpu().numpy()
        np_skeleton = skeleton[i].detach().cpu().numpy()
        np_recon_sequence = gen_seq[i].detach().cpu().numpy()

        n, _ = np_recon_sequence.shape
        np_recon_sequence = np.concatenate(  # 设置根节点xyz
            [np.zeros([n, 3], dtype=np.float32), np_recon_sequence], axis=1)

        save_bvh_to_file(
            'tb_logs/output/gen_{}_{}.bvh'.format(np_label, n), np_skeleton, np_recon_sequence, frame_time=0.025)

        break
