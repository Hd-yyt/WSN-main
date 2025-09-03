import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
from lib.utils import read_meta1, read_meta2,discretize_altitude
from geoclip import LocationEncoder



class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.weight_n = nn.Parameter(torch.tensor(0.5))
        self.weight_s = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, T, _, D = x.shape
        P, N = self.num, self.size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        x_s = x
        qkv = self.snorm1(x_s.reshape(B * T * P, N, D))
        attn_s = self.sattn(qkv).reshape(B, T, P, N, D)
        x_s = x_s + attn_s
        x_s = x_s + self.smlp(self.snorm2(x_s))

        x_n = x
        qkv = self.nnorm1(x_n.transpose(2, 3).reshape(B * T * N, P, D))
        attn_n = self.nattn(qkv).reshape(B, T, N, P, D).transpose(2, 3)
        x_n = x_n + attn_n
        x_n = x_n + self.nmlp(self.nnorm2(x_n))

        weights = torch.softmax(torch.stack([self.weight_n, self.weight_s]), dim=0)
        weight_n, weight_s = weights[0], weights[1]

        x = weight_n * x_n + weight_s * x_s
        return x.reshape(B, T, -1, D)

class PatchSTG(nn.Module):
    def __init__(self, tem_patchsize, tem_patchnum,
                 node_num, spa_patchsize, spa_patchnum,
                 layers, factors,
                 input_dims, node_dims,
                 ori_parts_idx, reo_parts_idx, reo_all_idx, meta_file):
        super(PatchSTG, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx

        # 更新 dims，包含海拔嵌入
        dims = input_dims + node_dims + node_dims

        # spatio-temporal embedding
        self.input_st_fc = nn.Conv2d(in_channels=1, out_channels=input_dims, kernel_size=(1, tem_patchsize),
                                     stride=(1, tem_patchsize), bias=True)

        self.gps_encoder = LocationEncoder()
        self.locations = torch.from_numpy(read_meta1(meta_file)).float().to(self.device)
        self.locations = self.locations.transpose(0, 1)
        self.node_emb_projection = nn.Linear(512, node_dims)

        # 海拔嵌入
        altitudes = read_meta2(meta_file)
        num_bins = 50
        altitude_indices = discretize_altitude(altitudes, num_bins)
        self.altitude_indices = torch.tensor(altitude_indices, dtype=torch.long).to(self.device)
        self.altitude_embedding = nn.Embedding(num_embeddings=num_bins, embedding_dim=node_dims)

        # dual attention encoder
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, spa_patchnum // factors, spa_patchsize * factors, mlp_ratio=1) for _ in
            range(layers)
        ])

        # projection decoder
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum * dims,
                                         out_channels=tem_patchsize,
                                         kernel_size=(1, 1), bias=True)

    def forward(self, x):
        embeded_x = self.embedding(x)
        rex = embeded_x[:, :, self.reo_all_idx, :]  # select patched points

        for block in self.spa_encoder:
            rex = block(rex)

        orginal = torch.zeros(rex.shape[0], rex.shape[1], self.node_num, rex.shape[-1]).to(x.device)
        orginal[:, :, self.ori_parts_idx, :] = rex[:, :, self.reo_parts_idx, :]
        pred_y = self.regression_conv(orginal.transpose(2, 3).reshape(orginal.shape[0], -1, orginal.shape[-2], 1))

        return pred_y  # [B,T,N,1]

    def embedding(self, x):
        b, t, n, _ = x.shape
        input_data = self.input_st_fc(x.transpose(1, 3)).transpose(1, 3)
        t, d = input_data.shape[1], input_data.shape[-1]

        # GPS 嵌入
        gps_embeddings = self.gps_encoder(self.locations)
        node_emb1 = self.node_emb_projection(gps_embeddings)  # 形状: (node_num, node_dims)
        node_emb1 = node_emb1.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)

        # 海拔嵌入
        altitude_emb = self.altitude_embedding(self.altitude_indices)  # 形状: (node_num, node_dims)
        altitude_emb = altitude_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)

        # 拼接所有嵌入
        input_data = torch.cat([input_data, node_emb1, altitude_emb], dim=-1)

        return input_data