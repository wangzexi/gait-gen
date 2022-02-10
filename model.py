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
    def __init__(self, patch_dim, feature_identify_dim, feature_motion_dim, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
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
        cls_tokens += self.pos_embedding[:, 0:1, :]

        if mask is not None:
            x_pos_embedding = x_pos_embedding.masked_select(
                repeat(mask, 'n -> b n d', b=b, d=d)).reshape(b, -1, d)
        x += x_pos_embedding

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        feature = self.mlp(x[:, 0])
        feature_identify = feature[:, :self.feature_identify_dim]
        feature_motion = feature[:, self.feature_identify_dim:]

        embedding = x[:, 1:]

        # [b, feature_dim], [b, n, dim]
        return feature_identify, feature_motion, embedding


class Decoder(pl.LightningModule):
    def __init__(self, feature_identify_dim, feature_motion_dim, patch_dim, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        super().__init__()

        self.pos_embedding = get_sinusoid_encoding_table(128, dim)
        self.placeholder = nn.Parameter(torch.zeros(1, 1, dim))

        self.feature_proj = nn.Linear(
            feature_identify_dim + feature_motion_dim, dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.out_proj = nn.Linear(dim, patch_dim)

    def forward(self, feature_identify, feature_motion, seq_len: int) -> torch.Tensor:
        # feature.shape: [b, dim]
        # seq_len: 生成序列长度

        if self.device != self.pos_embedding.device:  # hotfix
            self.pos_embedding = self.pos_embedding.to(self.device)

        b, n = feature_identify.shape[0], seq_len

        placeholder = repeat(self.placeholder, '() () d -> b n d', b=b, n=n)
        placeholder += self.pos_embedding[:, 1:(n + 1), :]  # 加位置编码

        feature = torch.cat((feature_identify, feature_motion), dim=1)
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
            patch_dim=93,
            dim=256,
            depth=3,
            heads=8,
            mlp_dim=512,
        )

        self.circle_loss_identify = CircleLoss(m=0.25, gamma=80)
        self.circle_loss_motion = CircleLoss(m=0.25, gamma=80)
        self.mse_loss = nn.MSELoss()

        self.pos_embedding = get_sinusoid_encoding_table(128, 93)

    def forward(self, input_seq, mask: torch.Tensor = None):
        fi, fm, embedding = self.encoder(input_seq, mask)
        recon_seq = self.decoder(fi, fm, seq_len=embedding.shape[1])
        return fi, fm, recon_seq

    def training_step(self, batch, batch_idx, optimizer_idx):
        skeleton, sequence, label, motion_label = batch

        input_seq = sequence[:, :, :]
        target_seq = sequence[:, :, :]

        b, n, c = input_seq.shape

        if optimizer_idx == 0:  # 训练自编码器

            mask = gen_mask(n, mask_ratio=0.8, device=self.device)

            feature_identify, feature_motion, embedding = self.encoder(
                input_seq, mask)
            recon_seq = self.decoder(
                feature_identify, feature_motion, seq_len=n)

            loss_cls_identify = self.circle_loss_identify(
                *convert_label_to_similarity(feature_identify, label))
            loss_cls_motion = self.circle_loss_motion(
                *convert_label_to_similarity(feature_motion, motion_label))

            loss_recon = self.mse_loss(recon_seq, target_seq)

            loss = loss_cls_identify + loss_cls_motion + loss_recon

            self.log_dict({'train/loss_cls_identify': loss_cls_identify,
                           'train/loss_cls_motion': loss_cls_motion,
                          'train/loss_recon': loss_recon})
            return loss

        if optimizer_idx == 1:  # 训练解码器，让它生成的序列与原始序列feature相似度更高
            feature_identify, feature_motion, embedding = self.encoder(
                input_seq)
            recon_seq = self.decoder(
                feature_identify, feature_motion, seq_len=n)
            recon_feature_identify, recon_feature_motion, _ = self.encoder(
                recon_seq)

            loss_recon_cls_identify = self.circle_loss_identify(
                *convert_label_to_similarity(torch.cat([feature_identify, recon_feature_identify]), torch.cat([label, label])))
            loss_recon_cls_motion = self.circle_loss_motion(
                *convert_label_to_similarity(torch.cat([feature_motion, recon_feature_motion]), torch.cat([motion_label, motion_label])))

            loss = loss_recon_cls_identify + loss_recon_cls_motion

            self.log_dict({'train/loss_recon_cls_identify': loss_recon_cls_identify,
                           'train/loss_recon_cls_motion': loss_recon_cls_motion})

            return loss

    def validation_step(self, batch, batch_idx):
        skeleton, sequence, label, motion_label = batch

        input_seq = sequence[:, :, :]
        target_seq = sequence[:, :, :]

        b, n, c = input_seq.shape

        feature_identify, feature_motion, embedding = self.encoder(input_seq)
        recon_seq = self.decoder(
            feature_identify, feature_motion, seq_len=n)

        recon_feature_identify, recon_feature_motion, _ = self.encoder(
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
        return loss

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(),
                            self.decoder.parameters()),
            lr=1e-3)
        opt_decoder = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        return [opt_ae, opt_decoder], []


if __name__ == '__main__':
    model = MaskGait()

    x = torch.randn(6, 80, 93)
    fi, ft, recon = model(x)
    print(fi.shape, ft.shape, recon.shape)
