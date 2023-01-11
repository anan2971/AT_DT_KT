import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


def o_distance(a, b):
    x = a - b
    y = torch.einsum("bhle,bhke->bhle", x, x) / a.shape[2]
    x = y.sum(axis=3, keepdim=True)  # keepdim=True 表示保留维度
    y = x / a.shape[3]
    x = torch.sqrt(y)
    return x

class TriangularCausalMask():
    def __init__(self, B, L, SR, device="cpu"):
        mask_shape = [B, 1, L, L, SR]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property  # 它把方法包装成属性，让方法可以以属性的形式被访问和调用。
    def mask(self):
        return self._mask   

class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        # self.distances = torch.zeros((window_size, window_size)).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distances = torch.zeros((window_size, window_size)).to(self.device)  # （100，100）的数据
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, x, query, key, value, sigma, attn_mask, retrieval, search, q_plus):
        B, L, H, E = query.shape  # H为头数，B为batch_size:256，D和E均为64
        _, S, _, D = value.shape  # LS值均为100，_为8

        scale = self.scale or 1. / sqrt(E)

        # compositional attention
        Search = torch.softmax(torch.einsum("blxe,bsxe->bxls", query, key) * scale, dim=-1)
        Retrieval = torch.einsum("bxls,bsye->blexy", Search, value)

        key_plus_projuction = nn.Linear(E, E).to(self.device)
        key_plus = key_plus_projuction(Retrieval.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        scores = torch.einsum("blxe,bmexy->blmxy", q_plus, key_plus)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, search * retrieval, device=query.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        series = torch.softmax(scores * scale, dim=-1)

        sigma = sigma.transpose(1, 2)  # B L H(search) ->  B H(search) L
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1)  # .repeat(1, 1, 1, L)  # B H(search) L L, L就是window_size

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)
        #prior = torch.einsum("bhle,bhke->bhlk", x, x)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(series)
        catt = torch.einsum("blsxy,bsexy->blexy", series, Retrieval)

        # final result
        catt = list(catt.chunk(retrieval, dim=-1))
        l = []
        for k in range(retrieval):
            l.append(catt[k].squeeze(-1))
        V = sum(l)  # CATTi -> blhx
        V_list = V.contiguous().view(B, L, search, -1)  # B L H(search) E
        V_projuction = nn.Linear(E, E).to(self.device)
        V = V_projuction(V_list)

        # conculate average sereis:because series has on more dimension than prior
        series = series.permute(0, 3, 1, 2, 4)
        series = list(series.chunk(retrieval, dim=-1))
        l2 = []
        for k in range(retrieval):
            l2.append(series[k].squeeze(-1))
        series = sum(l2) / retrieval

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, search=None, retrieval=None):
        super(AttentionLayer, self).__init__()

        self.search = search
        self.retrieval = retrieval
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * self.search)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * self.search)
        self.value_projection = nn.Linear(d_model,
                                          d_values * self.retrieval)
        self.x_projection = nn.Linear(d_model,
                                      d_values * self.search)
        self.sigma_projection = nn.Linear(d_model,
                                          self.search)
        self.out_projection = nn.Linear(d_values * self.search, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        query = self.query_projection(queries).view(B, L, self.search, -1)
        key = self.key_projection(keys).view(B, S, self.search, -1)
        value = self.value_projection(values).view(B, S, self.retrieval, -1)
        q_plus = self.x_projection(values).view(B, S, self.search, -1)
        sigma = self.sigma_projection(queries).view(B, L, self.search)
        x = query.transpose(1, 2)

        out, series, prior, sigma = self.inner_attention(
            x,
            query,
            key,
            value,
            sigma,
            attn_mask,
            self.retrieval,
            self.search,
            q_plus
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
