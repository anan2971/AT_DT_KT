import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
import os
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property  # 它把方法包装成属性，让方法可以以属性的形式被访问和调用。
    def mask(self):
        return self._mask


def o_distance(a, b):
    x = a - b
    y = torch.einsum("bhle,bhke->bhle", x, x)/a.shape[2]
    x = y.sum(axis=3, keepdim=True)  # keepdim=True 表示保留维度
    y = x/a.shape[3]
    x = torch.sqrt(y)
    return x


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
        #self.params = nn.Parameter(torch.ones(1), requires_grad=False).to(self.device)

    def forward(self, queries, keys, values, attn_mask, sigma, x, k=40):
        B, L, H, E = queries.shape  # H为头数8，B为batch_size:256，D和E均为64
        _, S, _, D = values.shape  # LS值均为100，_为8

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1)
        sigma_den = sigma.repeat(1, 1, 1, 64)

        # # 固定sigma
        # sigma = torch.tensor(np.zeros([B, H, L, 1]), dtype=torch.float)
        # sigma += 0.1

        x = x.transpose(1, 2).detach()  # B L H E ->  B H L E
        p_density = torch.zeros([B, H, L, E], dtype=torch.float, requires_grad=False).cuda().detach()

        p_density += 1.0 / (math.sqrt(2 * math.pi) * sigma_den) *torch.exp(-x ** 2 / 2 / (sigma_den ** 2)) 
        p_density += 1e-5
        p_density = torch.einsum("bhle,bhke->bhlk", p_density, p_density)
        
#         p_distance = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)
#         p_distance = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-p_distance ** 2 / 2 / (sigma ** 2))

        prior = p_density

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

#         B, L, H, E = queries.shape  # H为头数8，B为batch_size:256，D和E均为64
#         _, S, _, D = values.shape  # LS值均为100，_为8

#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)
#         attn = scale * scores

#         sigma = sigma.transpose(1, 2)  # B L H ->  B H L
#         sigma = torch.sigmoid(sigma * 5) + 1e-5
#         sigma = torch.pow(3, sigma) - 1
#         sigma = sigma.unsqueeze(-1)

#         # # 固定sigma
#         # sigma = torch.tensor(np.zeros([B, H, L, 1]), dtype=torch.float)
#         # sigma += 0.1

#         x = x.transpose(1, 2).detach()  # B L H E ->  B H L E
#         p_density = torch.zeros([B, H, L, 1], dtype=torch.float, requires_grad=False).cuda().detach()
#         z = torch.zeros([B, H, L, 1], dtype=torch.float, requires_grad=False).cuda().detach()
#         neighbours = [0 for i in range(L)]

#         # 往后找临近点
#         for i in range(1, k // 2 + 1, 1):
#             y = x
#             for j in range(L):  # l=window_size
#                 if (i + j) < L:
#                     z[:, :, [j], :] = o_distance(y[:, :, [j], :], x[:, :, [i + j], :])
#                     neighbours[j] += 1
#             p_density += 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(
#                 -z ** 2 / 2 / (sigma ** 2))  # 1.0 / (math.sqrt(2 * math.pi) * sigma) *

#         # 往前找临近点
#         for i in range(1, (k - k // 2) + 1, 1):
#             y = x
#             for j in range(L):
#                 if (j - i) > 0:
#                     z[:, :, [j], :] = o_distance(y[:, :, [j], :], x[:, :, [j - i], :])
#                     neighbours[j] += 1
#             p_density += 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(
#                 -z ** 2 / 2 / (sigma ** 2))  # 1.0 / (math.sqrt(2 * math.pi) * sigma) *

#         for i in range(L):
#             p_density[:, :, [i], :] /= (neighbours[i] + 1)
#         p_density += 1e-5

#         p_density = torch.einsum("bhle,bhke->bhlk", p_density, p_density)

# #         p_distance = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(
# #             self.device)
# #         p_distance = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-p_distance ** 2 / 2 / (sigma ** 2))

#         prior = p_density

#         series = self.dropout(torch.softmax(attn, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", series, values)

#         if self.output_attention:
#             return (V.contiguous(), series, prior, sigma)
#         else:
#             return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        # self.x_projection = nn.Linear(d_model,
        #                               d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        x = x.view(B, S, H, -1)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            sigma,
            x
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
