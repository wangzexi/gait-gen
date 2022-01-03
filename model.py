from typing import Tuple

import functools

import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch._C import device

from circle_loss import convert_label_to_similarity, CircleLoss


# helpers

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


# classes

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
    def __init__(self, patch_dim, feature_dim, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.proj = nn.Linear(patch_dim, dim)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # precompute position embedding table
        self.pos_embedding = get_sinusoid_encoding_table(128, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, feature_dim)
        )

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

        # x = self.dropout(x)
        x = self.transformer(x)

        feature = self.mlp_head(x[:, 0])
        embedding = x[:, 1:]

        return feature, embedding  # [b, feature_dim], [b, n, dim]


class Decoder(pl.LightningModule):
    def __init__(self, patch_dim, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        super().__init__()

        # self.placeholder = nn.Parameter(torch.randn(1, 1, dim))
        self.placeholder = nn.Parameter(torch.zeros(1, 1, dim))

        # precompute position embedding table
        self.pos_embedding = get_sinusoid_encoding_table(128, dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.proj = nn.Linear(dim, patch_dim)

    def forward(self, x, mask: torch.Tensor = None) -> torch.Tensor:
        # x.shape: [b, n, dim]

        if self.device != self.pos_embedding.device:  # hotfix
            self.pos_embedding = self.pos_embedding.to(self.device)

        if mask is not None:
            # recovery masked embedding
            b, n, d = x.shape[0], mask.shape[0], x.shape[2]
            recovery_x = repeat(self.placeholder, '() () d -> b n d', b=b, n=n)
            recovery_x += self.pos_embedding[:, 1:(n + 1), :]
            x = recovery_x.masked_scatter(
                repeat(mask, 'n -> b n d', b=b, n=n, d=d), x)

        x = self.transformer(x)
        x = self.proj(x)
        return x


class MaskGait(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(
            patch_dim=18*2,
            feature_dim=256,
            dim=256,
            depth=4,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.decoder = Decoder(
            patch_dim=18*2,
            dim=256,
            depth=3,
            heads=8,
            mlp_dim=512,
        )

        self.cricle_loss = CircleLoss(m=0.25, gamma=80)
        self.mse_loss = nn.MSELoss()

    def forward(self, x, mask: torch.Tensor = None):
        feature, embedding = self.encoder(x, mask)
        reconstruction = self.decoder(embedding, mask)
        return feature, reconstruction

    def training_step(self, batch, batch_idx):
        sequence, subject_id, camera_angle, sequence_no = batch

        b, n, _, _ = sequence.shape
        sequence = sequence.reshape(b, n, -1)  # [b, n, 18*2]

        mask_ratio = 0.2
        mask = gen_mask(n, mask_ratio, device=self.device)

        feature, reconstruction = self.forward(sequence, mask)

        loss_cls = self.cricle_loss(
            *convert_label_to_similarity(feature, subject_id))
        loss_recon = self.mse_loss(reconstruction, sequence)
        loss = loss_cls + loss_recon

        self.log_dict({'train/loss_cls': loss_cls,
                      'train/loss_recon': loss_recon, 'train/loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        sequence, subject_id, camera_angle, sequence_no = batch

        b, n, _, _ = sequence.shape
        sequence = sequence.reshape(b, n, -1)  # [b, n, 18*2]

        # mask_ratio = 0
        # mask = gen_mask(n, mask_ratio, device=self.device)

        feature, reconstruction = self.forward(sequence)

        loss_cls = self.cricle_loss(
            *convert_label_to_similarity(feature, subject_id))
        loss_recon = self.mse_loss(reconstruction, sequence)
        loss = loss_cls + loss_recon

        self.log_dict({'val/loss_cls': loss_cls,
                      'val/loss_recon': loss_recon, 'val/loss': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    model = MaskGait()
    x = torch.randn(6, 14, 36)
    model(x)
