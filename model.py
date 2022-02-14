from typing import Tuple

import itertools
import functools

import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

from einops import rearrange, repeat

from circle_loss import convert_label_to_similarity, CircleLoss

from dataset import save_bvh_to_file


@functools.lru_cache(maxsize=16)
def get_sinusoid_encoding_table(n_position, d_hid, device: str = 'cpu'):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.tensor(sinusoid_table, dtype=torch.float32, device=device).unsqueeze(0)


def gen_mask(n: int, mask_ratio: float = 0.2, device: str = 'cpu') -> torch.Tensor:
    mask = torch.ones(n, device=device)
    mask[:int(n * mask_ratio)] = 0
    mask = mask[torch.randperm(n)]
    mask = mask.eq(1)
    # False: masked, True: unmasked
    return mask  # [n]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Encoder(pl.LightningModule):
    def __init__(self, patch_dim, feature_identify_dim, feature_motion_dim, z_dim, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.pos_embedding = get_sinusoid_encoding_table(128, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.proj = nn.Linear(patch_dim, dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, feature_identify_dim+feature_motion_dim),
        )

        self.feature_identify_dim = feature_identify_dim
        self.feature_motion_dim = feature_motion_dim

        self.fc_mu = nn.Linear(feature_identify_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_identify_dim, z_dim)

    def forward(self, x, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.shape: [b, n, patch_dim]
        b, n, d = x.shape

        if mask is not None:
            x = x.masked_select(
                repeat(mask, 'n -> b n d', b=b, d=d)).reshape(b, -1, d)

        if self.device != self.pos_embedding.device:  # hotfix
            self.pos_embedding = self.pos_embedding.to(self.device)

        x_pos_embedding = self.pos_embedding[:, 1:(n + 1), :]

        x = self.proj(x)  # [b, n, dim]

        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        cls_tokens = cls_tokens + self.pos_embedding[:, 0:1, :]

        if mask is not None:
            x_pos_embedding = x_pos_embedding.masked_select(
                repeat(mask, 'n -> b n d', b=b, d=d)).reshape(b, -1, d)
        x = x + x_pos_embedding

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        feature = self.mlp(x[:, 0])
        feature_identify = feature[:, :self.feature_identify_dim]
        feature_motion = feature[:, self.feature_identify_dim:]

        embedding = x[:, 1:]

        # vae
        mu = self.fc_mu(feature_identify)
        logvar = self.fc_logvar(feature_identify)

        # [b, feature_dim], [b, n, dim]
        return (feature_identify, feature_motion), (mu, logvar),  embedding


class Decoder(pl.LightningModule):
    def __init__(self, feature_identify_dim, feature_motion_dim, z_dim, patch_dim, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        super().__init__()

        self.pos_embedding = get_sinusoid_encoding_table(128, dim)
        self.placeholder = nn.Parameter(torch.zeros(1, 1, dim))

        self.feature_proj = nn.Linear(
            z_dim + feature_motion_dim, dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.out_proj = nn.Linear(dim, patch_dim)

    def forward(self, feature_identify_z, feature_motion, seq_len: int) -> torch.Tensor:
        # feature_identify_z.shape: [b, feature_identify_dim]
        # feature_motion.shape: [b, feature_motion_dim]
        # seq_len: 生成序列长度

        if self.device != self.pos_embedding.device:  # hotfix
            self.pos_embedding = self.pos_embedding.to(self.device)

        b, n = feature_identify_z.shape[0], seq_len

        placeholder = repeat(self.placeholder, '() () d -> b n d', b=b, n=n)
        placeholder = placeholder + \
            self.pos_embedding[:, 1:(n + 1), :]  # 加位置编码

        feature = torch.cat((feature_identify_z, feature_motion), dim=1)
        feature = rearrange(self.feature_proj(feature), 'b d -> b 1 d')

        x = self.transformer(torch.cat([feature, placeholder], dim=1))
        x = self.out_proj(x[:, 1:, :])
        return x


class MaskGait(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(
            patch_dim=93,
            feature_identify_dim=64,
            feature_motion_dim=64,
            z_dim=64,
            dim=256,
            depth=4,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.decoder = Decoder(
            feature_identify_dim=64,
            feature_motion_dim=64,
            z_dim=64,
            patch_dim=93,
            dim=256,
            depth=3,
            heads=8,
            mlp_dim=512,
        )

        self.circle_loss_identify = CircleLoss(m=0.25, gamma=80)
        self.circle_loss_motion = CircleLoss(m=0.25, gamma=80)
        self.mse_loss = nn.MSELoss()

        def kld_loss(mu, logvar):
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.kld_loss = kld_loss

        self.pos_embedding = get_sinusoid_encoding_table(128, 93)

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(),
                            self.decoder.parameters()),
            lr=1e-3)
        opt_decoder = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        return [opt_ae, opt_decoder], []

    def forward(self, input_seq, mask: torch.Tensor = None):
        (fi, fm), (mu, logvar), embedding = self.encoder(input_seq, mask)
        return fi, fm

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def training_step(self, batch, batch_idx, optimizer_idx):
        skeleton, sequence, label, motion_label = batch

        if self.global_step % 2 == 0:  # 样本自交训练

            input_seq = sequence[:, :, :]
            target_seq = sequence[:, :, :]

            b, n, c = input_seq.shape

            if optimizer_idx == 0:  # 训练编码器
                mask = gen_mask(n, mask_ratio=0.8, device=self.device)

                (feature_identify, feature_motion), (mu, logvar), embedding = self.encoder(
                    input_seq, mask)
                z = self.reparameterize(mu, logvar)
                recon_seq = self.decoder(z, feature_motion, seq_len=n)

                loss_cls_identify = self.circle_loss_identify(
                    *convert_label_to_similarity(feature_identify, label))
                loss_identify_kld = self.kld_loss(mu, logvar)

                loss_cls_motion = self.circle_loss_motion(
                    *convert_label_to_similarity(feature_motion, motion_label))

                loss_recon = self.mse_loss(recon_seq, target_seq)

                loss = loss_cls_identify + loss_identify_kld + loss_cls_motion + loss_recon

                self.log_dict({'train/loss_cls_identify': loss_cls_identify,
                               'train/loss_cls_motion': loss_cls_motion,
                               'train/loss_recon': loss_recon})
                return loss

            if optimizer_idx == 1:  # 训练解码器，让生成的序列与原始序列feature相似度更高
                (feature_identify, feature_motion), (mu, logvar), embedding = self.encoder(
                    input_seq)
                z = self.reparameterize(mu, logvar)
                recon_seq = self.decoder(z, feature_motion, seq_len=n)
                (recon_feature_identify, recon_feature_motion), (mu, logvar), _ = self.encoder(
                    recon_seq)

                loss_recon_cls_identify = self.circle_loss_identify(
                    *convert_label_to_similarity(torch.cat([feature_identify, recon_feature_identify]), torch.cat([label, label])))
                loss_recon_cls_motion = self.circle_loss_motion(
                    *convert_label_to_similarity(torch.cat([feature_motion, recon_feature_motion]), torch.cat([motion_label, motion_label])))

                loss = loss_recon_cls_identify + loss_recon_cls_motion

                self.log_dict({'train/loss_recon_cls_identify': loss_recon_cls_identify,
                               'train/loss_recon_cls_motion': loss_recon_cls_motion})

                return loss

        else:  # 样本杂交训练
            b, n, c = sequence.shape

            idx = torch.randperm(b)

            a_input_seq = sequence[:, :, :]
            a_target_seq = sequence[:, :, :]
            a_label = label[:]
            a_motion_label = motion_label[:]

            b_input_seq = sequence[idx, :, :]
            b_target_seq = sequence[idx, :, :]
            b_label = label[idx]
            b_motion_label = motion_label[idx]

            if optimizer_idx == 0:  # 训练编码器
                mask = gen_mask(n, mask_ratio=0.8, device=self.device)

                (a_feature_identify, a_feature_motion), (a_mu, a_logvar), a_embedding = self.encoder(
                    a_input_seq, mask)
                (b_feature_identify, b_feature_motion), (b_mu, b_logvar), b_embedding = self.encoder(
                    b_input_seq, mask)

                a_z = self.reparameterize(a_mu, a_logvar)
                recon_seq = self.decoder(a_z, b_feature_motion, seq_len=n)

                loss_cls_identify = self.circle_loss_identify(
                    *convert_label_to_similarity(a_feature_identify, a_label))
                loss_identify_kld = self.kld_loss(a_mu, b_logvar)

                loss_cls_motion = self.circle_loss_motion(
                    *convert_label_to_similarity(b_feature_motion, b_motion_label))

                loss_recon = self.mse_loss(recon_seq, b_target_seq)  # 运动来自b

                loss = loss_cls_identify + loss_identify_kld + loss_cls_motion + 0.5*loss_recon

                self.log_dict({'train/loss_cls_identify': loss_cls_identify,
                               'train/loss_cls_motion': loss_cls_motion,
                               'train/loss_recon': loss_recon})
                return loss

            if optimizer_idx == 1:  # 训练解码器，让生成的序列与原始序列feature相似度更高
                (a_feature_identify, a_feature_motion), (a_mu, a_logvar), a_embedding = self.encoder(
                    a_input_seq)
                (b_feature_identify, b_feature_motion), (b_mu, b_logvar), b_embedding = self.encoder(
                    b_input_seq)

                a_z = self.reparameterize(a_mu, a_logvar)
                recon_seq = self.decoder(a_z, b_feature_motion, seq_len=n)

                (recon_feature_identify, recon_feature_motion), (_, _), _ = self.encoder(
                    recon_seq)

                loss_recon_cls_identify = self.circle_loss_identify(
                    *convert_label_to_similarity(torch.cat([a_feature_identify, recon_feature_identify]), torch.cat([a_label, a_label])))
                loss_recon_cls_motion = self.circle_loss_motion(
                    *convert_label_to_similarity(torch.cat([b_feature_motion, recon_feature_motion]), torch.cat([b_motion_label, b_motion_label])))

                loss = loss_recon_cls_identify + loss_recon_cls_motion

                self.log_dict({'train/loss_recon_cls_identify': loss_recon_cls_identify,
                               'train/loss_recon_cls_motion': loss_recon_cls_motion})

                return loss

    def validation_step(self, batch, batch_idx):
        skeleton, sequence, label, motion_label = batch

        input_seq = sequence[:, :, :]
        target_seq = sequence[:, :, :]

        b, n, c = input_seq.shape

        (feature_identify, feature_motion), (mean,
                                             logvar), embedding = self.encoder(input_seq)
        # z = self.reparameterize(mean, logvar)
        z = torch.randn_like(mean)
        recon_seq = self.decoder(z, feature_motion, seq_len=n)

        (recon_feature_identify, recon_feature_motion), (mu, logvar), _ = self.encoder(
            recon_seq)

        loss_cls_identify = self.circle_loss_identify(
            *convert_label_to_similarity(feature_identify, label))
        loss_cls_motion = self.circle_loss_motion(
            *convert_label_to_similarity(feature_motion, motion_label))

        loss_recon = self.mse_loss(recon_seq, target_seq)

        loss_recon_cls_identify = self.circle_loss_identify(
            *convert_label_to_similarity(torch.cat([feature_identify, recon_feature_identify]), torch.cat([label, label])))
        loss_recon_cls_motion = self.circle_loss_motion(
            *convert_label_to_similarity(torch.cat([feature_motion, recon_feature_motion]), torch.cat([motion_label, motion_label])))

        loss = loss_cls_identify + loss_cls_motion + loss_recon + \
            loss_recon_cls_identify + loss_recon_cls_motion

        self.log_dict({'val/loss_cls_identify': loss_cls_identify,
                       'val/loss_cls_motion': loss_cls_motion,
                      'val/loss_recon': loss_recon,
                       'val/loss_recon_cls_identify': loss_recon_cls_identify,
                       'val/loss_recon_cls_motion': loss_recon_cls_motion,
                       'val/loss': loss})

        # if self.trainer.logger_connector.should_update_logs:
        i = np.random.randint(0, len(label))
        np_label = label[i].cpu().numpy()
        np_motion_label = motion_label[i].cpu().numpy()
        np_skeleton = skeleton[i].detach().cpu().numpy()
        np_recon_seq = recon_seq[i].detach().cpu().numpy()
        n, _ = np_recon_seq.shape
        np_recon_seq = np.concatenate(  # 设置根节点xyz为0
            [np.zeros([n, 3], dtype=np.float32), np_recon_seq], axis=1)

        save_bvh_to_file(
            'tb_logs/output/recon_{}_{}_{}_{}.bvh'.format(self.current_epoch, np_label, np_motion_label, n), np_skeleton, np_recon_seq, frame_time=0.025)

        if self.global_step % 2 == 0:
            self.logger.experiment.add_embedding(
                feature_identify, metadata=label, global_step=self.global_step)
        else:
            self.logger.experiment.add_embedding(
                feature_motion, metadata=motion_label, global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        skeleton, sequence, label, motion_label = batch

        input_seq = sequence[:, :, :]
        target_seq = sequence[:, :, :]

        b, n, c = input_seq.shape

        print('测试代码实现在此处')


if __name__ == '__main__':
    model = MaskGait()

    x = torch.randn(6, 80, 93)
    fi, ft = model(x)
    print(fi.shape, ft.shape)
